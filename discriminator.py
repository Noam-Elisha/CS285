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
    def train(self,writer,n_epi,agent_s,agent_a,expert_s,expert_a):
        
        
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
        self.fc2 = nn.Linear(hidden_dim,hidden_dim)
        self.mu = nn.Linear(hidden_dim, z_dim) 
        self.sigma = nn.Linear(hidden_dim, z_dim) 
        
        
        self.fc3 = nn.Linear(z_dim,1)
        self.dual_stepsize = dual_stepsize
        self.mutual_info_constraint = mutual_info_constraint
        self.beta = 0
        self.r = Normal(torch.zeros(z_dim),torch.ones(z_dim))
        self.criterion = nn.BCELoss()
        
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_() 
        self.optimizer = optim.Adam(self.parameters(), lr=discriminator_lr)

    def forward(self, x,get_dist = False):
        z,mu,std = self.get_z(x)
        x = torch.sigmoid(self.fc3(z))
        if get_dist == False:
            return x
        else:
            return x,mu,std
    def get_z(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mu = self.mu(x)
        sigma = self.sigma(x)
        std = torch.exp(sigma/2)
        eps = torch.randn_like(std)
        return  mu + std * eps,mu,sigma
        
        
    def get_reward(self,state,action):
        x = torch.cat((state,action),-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mu = self.mu(x)
        x = torch.sigmoid(self.fc3(torch.relu(mu)))+ 1e-8
        
        return -torch.log(x).detach()
    def get_latent_kl_div(self,mu,logvar):
        return torch.mean(-logvar+(torch.square(mu)+torch.square(torch.exp(logvar))-1.)/2.)
    def train(self,writer,n_epi,agent_s,agent_a,expert_s,expert_a):
        for i in range(3):
            expert_cat = torch.cat((torch.tensor(expert_s),torch.tensor(expert_a)),-1)
            expert_preds,expert_mu,expert_std = self.forward(expert_cat.float().to(self.device),get_dist = True)

            expert_loss = self.criterion(expert_preds,torch.zeros(expert_preds.shape[0],1).to(self.device))

            agent_cat = torch.cat((agent_s,agent_a),-1)
            agent_preds,agent_mu,agent_std = self.forward(agent_cat.float().to(self.device),get_dist = True)
            agent_loss = self.criterion(agent_preds,torch.ones(agent_preds.shape[0],1).to(self.device))
            
            expert_bottleneck_loss = self.get_latent_kl_div(expert_mu,expert_std)
            
            agent_bottleneck_loss = self.get_latent_kl_div(agent_mu,agent_std)
            
            bottleneck_loss = 0.5 * (expert_bottleneck_loss + agent_bottleneck_loss)
            bottleneck_loss = bottleneck_loss -  self.mutual_info_constraint
            
            
            self.beta = max(0,self.beta + self.dual_stepsize * bottleneck_loss.detach())
            loss = expert_loss + agent_loss + (bottleneck_loss) * self.beta


            expert_acc = ((expert_preds < 0.5).float()).mean()
            learner_acc = ((agent_preds > 0.5).float()).mean()
            if self.writer != None:
                self.writer.add_scalar("loss/discriminator_loss", loss.item(), n_epi)
            if (expert_acc > 0.8) and (learner_acc > 0.8):
                return 
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()
            
class G(nn.Module):
    def __init__(self,device,state_dim,action_dim,hidden_dim,state_only = True):
        super(G,self).__init__()
        self.device = device
        self.state_only = state_only
        if state_only :
            self.fc1 = nn.Linear(state_dim, hidden_dim)
        else :
            self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_() 
    def forward(self,state,action):
        if self.state_only:
            x = state
        else:
            x = torch.cat((state,action),-1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = (self.fc3(x))
        return x
class H(nn.Module):
    def __init__(self,device,state_dim,hidden_dim):
        super(H,self).__init__()
        self.device = device
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_() 
    def forward(self,state):
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        x = (self.fc3(x))
        return x
    
class AIRLDiscriminator(nn.Module):
    def __init__(self, writer, device, state_dim, action_dim, hidden_dim,discriminator_lr,gamma,state_only):
        super(AIRLDiscriminator, self).__init__()
        self.writer = writer
        self.device = device
        self.gamma = gamma
        self.g = G(device,state_dim,action_dim,hidden_dim,state_only = state_only)
        self.h = H(device,state_dim,hidden_dim)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=discriminator_lr)
        
    def get_f(self,state,action,next_state,done):
        return self.g(state,action) + (1-done.float()) * self.gamma * self.h(next_state) - self.h(state)
    def get_d(self,prob,state,action,next_state,done):
        exp_f = torch.exp(self.get_f(state,action,next_state,done))
        return (exp_f/(exp_f + prob))
    def get_reward(self,prob,state,action,next_state,done):
        d = self.get_d(prob,state,action,next_state,done)
        #return (torch.log(d+1e-3) - torch.log(1-d)+1e-3).detach()
        #return (- torch.log(1-d)+1e-3).detach()
        return -torch.log(d).detach()
    def forward(self,prob,state,action,next_state,done):
        d = (self.get_d(prob,state,action,next_state,done))
        return d
        
    def train(self,writer,n_epi,agent_s,agent_a,agent_next_s,agent_prob,agent_done,expert_s,expert_a,expert_next_s,expert_prob,expert_done):
        
        expert_preds = self.forward(expert_prob,expert_s,expert_a,expert_next_s,expert_done)
        expert_loss = self.criterion(expert_preds,torch.zeros(expert_preds.shape[0],1).to(self.device)) 
        
        agent_preds = self.forward(agent_prob,agent_s,agent_a,agent_next_s,agent_done)
        agent_loss = self.criterion(agent_preds,torch.ones(agent_preds.shape[0],1).to(self.device))
        
        loss = expert_loss+agent_loss
        expert_acc = ((expert_preds < 0.5).float()).mean()
        learner_acc = ((agent_preds > 0.5).float()).mean()
        #print("expert_acc : ",expert_acc)
        #print("learner_acc : ",learner_acc)
        if self.writer != None:
            self.writer.add_scalar("loss/discriminator_loss", loss.item(), n_epi)
        #if (expert_acc > 0.8) and (learner_acc > 0.8):
        #    return 
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()