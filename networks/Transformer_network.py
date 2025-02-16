import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.discriminator_transformer import Transformer_Network
from networks.utils_network import Block, PositionalEncoding



    
class VDB_transformer(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, layer_num, z_dim):
        super(VDB_transformer, self).__init__()
        input_dim = state_dim + action_dim
        
        self.encoder_input_layer = torch.nn.Linear(input_dim, hidden_dim)
        self.pos_encod = PositionalEncoding(hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4)
        self.attention = nn.TransformerEncoder(encoder_layer, num_layers=layer_num)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, z_dim) 
        self.sigma = nn.Linear(hidden_dim, z_dim) 
        # self.ln1 = nn.LayerNorm(input_dim)

    def get_z(self,x):
        x_encoded = nn.ReLU()(self.encoder_input_layer(x)) 
        
        # Pass through the positional encoding layer
        x = self.pos_encod(x_encoded)
        x = self.attention(x)
        # x = self.ln1(x)
        x = nn.ReLU()(self.fc1(x))
        # x = torch.relu(self.fc2(x))
        mu = self.mu(x)
        sigma = self.sigma(x)
        std = torch.exp(sigma/2)
        eps = torch.randn_like(std)
        return  mu + std * eps,mu,sigma
    
    def get_mean(self,x):
        
        x_encoded = nn.ReLU()(self.encoder_input_layer(x)) 
        
        # Pass through the positional encoding layer
        x = self.pos_encod(x_encoded)
        x = self.attention(x)
        # x = self.ln1(x)
        x = nn.ReLU()(self.fc1(x))
        # x = torch.relu(self.fc2(x))
        mu = self.mu(x)
        return mu
    


    
class G_transformer(Transformer_Network):
    def __init__(self,state_only, layer_num, input_dim, output_dim, hidden_dim,activation_function = torch.relu,last_activation = None , drop_p=0.8):
        if state_only :
            super(G_transformer, self).__init__(input_dim,layer_num, hidden_dim )
        else:
            super(G_transformer, self).__init__(input_dim+output_dim, layer_num, hidden_dim)
        self.state_only = state_only
    def forward(self, state, action):
        if self.state_only:
            x = state
        else:
            x = torch.cat((state,action),-1)
        return self._forward(x)
    
    
class H_transformer(Transformer_Network):
    def __init__(self, layer_num, input_dim, output_dim, hidden_dim,activation_function = torch.relu,last_activation = None , drop_p=0.8):
        super(H_transformer, self).__init__(input_dim,layer_num, hidden_dim)
        
    def forward(self, x):
        return self._forward(x)
    
    
class Reward_transformer(Transformer_Network):
    def __init__(self,layer_num, input_dim, output_dim, hidden_dim):
        super(Reward_transformer,self).__init__(input_dim+output_dim,layer_num,hidden_dim, 1)
        '''
        self.reward
        input : s,a
        output : scalar
        '''
    def forward(self,state,action):
        x = torch.cat((state,action),-1)
        return self._forward(x)