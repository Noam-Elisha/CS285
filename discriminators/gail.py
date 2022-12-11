from discriminators.base import Discriminator
from networks.base import Network
from networks.discriminator_transformer import Transformer_Network
import torch
import torch.nn as nn

class GAIL(Discriminator):
    def __init__(self, writer, device, state_dim, action_dim, args, transformer):
        super(GAIL, self).__init__()
        self.writer = writer
        self.device = device
        self.args = args
        if transformer:
            self.network = Transformer_Network( state_dim +
                                           action_dim, args.layer_num,args.hidden_dim)
        else:
            self.network = Network(args.layer_num, state_dim+action_dim, 1, args.hidden_dim, args.activation_function,args.last_activation)
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr)

    def forward(self, x):
        x= x.unsqueeze(0)
        
        prob = self.network.forward(x)
        
        return prob.squeeze(0)


    def get_reward(self,state,action):
        x = torch.cat((state,action),-1)
        x = self.network.forward(x)
        return -torch.log(x).detach()
    def train_network(self,writer,n_epi,agent_s,agent_a,expert_s,expert_a):
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