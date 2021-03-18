import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, kl_divergence

class GAILDiscriminator(nn.Module):
    def __init__(self, writer, device, state_dim, action_dim, hidden_dim,discriminator_lr):
        super(GAILDiscriminator, self).__init__()
        self.writer = writer
        self.device = device
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.criterion = nn.BCELoss()

        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_() 
        self.optimizer = optim.Adam(self.parameters(), lr=discriminator_lr)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        prob = torch.sigmoid(self.fc3(x))
        return prob
    def get_reward(self,state,action):
        x = torch.cat((state,action),-1)
        x = self.forward(x)
        return -torch.log(x).detach()
    def train(self,n_epi,agent_s,agent_a,expert_s,expert_a):
        
        
        expert_cat = torch.cat((torch.tensor(expert_s),torch.tensor(expert_a)),-1)
        expert_preds = self.forward(expert_cat.float().to(self.device))
        
        expert_loss = self.criterion(expert_preds,torch.zeros(expert_preds.shape[0],1).to(self.device))
        
        agent_cat = torch.cat((agent_s,agent_a),-1)
        agent_preds = self.forward(agent_cat.float().to(self.device))
        agent_loss = self.criterion(agent_preds,torch.ones(agent_preds.shape[0],1).to(self.device))
        
        loss = expert_loss+agent_loss
        expert_acc = ((expert_preds < 0.5).float()).mean()
        learner_acc = ((agent_preds > 0.5).float()).mean()
        if self.writer != None:
            self.writer.add_scalar("loss/discriminator_loss", loss.item(), n_epi)
        if (expert_acc > 0.8) and (learner_acc > 0.8):
            return 
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        
class VAILDiscriminator(nn.Module):
    def __init__(self, writer, device, state_dim, action_dim, hidden_dim,z_dim,discriminator_lr,dual_stepsize=1e-5,mutual_info_constraint=0.5):
        super(VAILDiscriminator, self).__init__()
        self.writer = writer
        self.device = device
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, z_dim) #128
        self.sigma = nn.Linear(hidden_dim, z_dim) #128
        
        self.fc2 = nn.Linear(z_dim,hidden_dim)
        self.fc3 = nn.Linear(hidden_dim,1)
        self.dual_stepsize = dual_stepsize
        self.mutual_info_constraint = mutual_info_constraint
        self.beta = 0
        self.r = Normal(0,1)
        self.criterion = nn.BCELoss()
        
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_() 
        self.optimizer = optim.Adam(self.parameters(), lr=discriminator_lr)

    def forward(self, x,get_dist = False):
        z,mu,std = self.get_z(x)
        x = torch.tanh(self.fc2(z))
        x = torch.sigmoid(self.fc3(x))
        if get_dist == False:
            return x
        else:
            return x,mu,std
    def get_z(self,x):
        x = torch.tanh(self.fc1(x))
        mu = self.mu(x)
        sigma = self.sigma(x)
        #std = torch.exp(0.5 * sigma)
        #eps = torch.randn_like(std)
        #return  eps * std + mu,mu,std
        normal = Normal(mu,sigma)
        
        return normal.rsample(),mu,sigma
        
    def get_reward(self,state,action):
        x = torch.cat((state,action),-1)
        x = self.forward(x)
        return -torch.log(x).detach()
    def train(self,n_epi,agent_s,agent_a,expert_s,expert_a):
        expert_cat = torch.cat((torch.tensor(expert_s),torch.tensor(expert_a)),-1)
        expert_preds,expert_mu,expert_std = self.forward(expert_cat.float().to(self.device),get_dist = True)
        
        expert_loss = self.criterion(expert_preds,torch.zeros(expert_preds.shape[0],1).to(self.device))
        
        agent_cat = torch.cat((agent_s,agent_a),-1)
        agent_preds,agent_mu,agent_std = self.forward(agent_cat.float().to(self.device),get_dist = True)
        agent_loss = self.criterion(agent_preds,torch.ones(agent_preds.shape[0],1).to(self.device))
        
        expert_bottleneck_loss = torch.sum(kl_divergence(Normal(expert_mu,expert_std),self.r))
        agent_bottleneck_loss = torch.sum(kl_divergence(Normal(agent_mu,agent_std),self.r))
        
        
        bottleneck_loss = 0.5 * (expert_bottleneck_loss + agent_bottleneck_loss) #-self.mutual_info_constraint
        self.beta = max(0,self.beta + self.dual_stepsize * (bottleneck_loss.detach()-self.mutual_info_constraint))
        loss = expert_loss+agent_loss + (bottleneck_loss - self.mutual_info_constraint) *self.beta
        
        
        expert_acc = ((expert_preds < 0.5).float()).mean()
        learner_acc = ((agent_preds > 0.5).float()).mean()
        if self.writer != None:
            self.writer.add_scalar("loss/discriminator_loss", loss.item(), n_epi)
        if (expert_acc > 0.8) and (learner_acc > 0.8):
            return 
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()