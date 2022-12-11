import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.utils_network import Block, PositionalEncoding

            
class Transformer_Network(nn.Module):
    def __init__(self, input_dim, n_blocks, h_dim,output_dim = 1, context_len=10, 
                 n_heads=1, drop_p=0.5, max_timestep=4096):
        super().__init__()

        self.input_dim = input_dim
        self.h_dim = h_dim
        
        
        
        # continuous actions
        self.encoder_input_layer = torch.nn.Linear(input_dim, h_dim)
        
        self.encod = PositionalEncoding(h_dim)
        ### transformer blocks
        input_seq_len = 3 * context_len
        blocks = [Block(h_dim, input_seq_len, n_heads, drop_p) for _ in range(n_blocks)]
        self.transformer = nn.Sequential(*blocks)

        ### projection heads (project to embedding)
        self.embed_ln = nn.LayerNorm(h_dim)


        # use_action_tanh = True # True for continuous actions
        
        ### prediction heads
        self.predict_rtg = torch.nn.Linear(h_dim, 1)



    def forward(self,x):
        # time embeddings are treated similar to positional embeddings
        
        src = self.encoder_input_layer(x) 
        
        # Pass through the positional encoding layer
        h = self.positional_encoding_layer(src)
        
        
        # transformer and prediction
        h = self.transformer(h)
        
        h = self.embed_ln(h)
        # get predictions
        return_preds = nn.Sigmoid()(self.predict_rtg(h)) 
        return return_preds


    
    