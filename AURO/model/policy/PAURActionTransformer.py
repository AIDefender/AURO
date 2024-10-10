from model.policy.ActionTransformer import ActionTransformer
import datetime
from model.components import DNN
import os
import numpy as np
import torch
from model.general import BaseModel


    
class PAURActionTransformer(ActionTransformer):
    @staticmethod
    def parse_model_args(parser):
        parser = ActionTransformer.parse_model_args(parser)
        parser.add_argument('--abstraction_dim', type=int, default=8, 
                            help='state abstraction size')
        parser.add_argument('--abstraction_hidden_dims', type=int, nargs='+', default=[128,128], 
                            help='hidden dimensions')        
        return parser
    
    def __init__(self, args, env, device):
        reader_stats = env.reader.get_statistics()
        self.abstraction_dim = args.abstraction_dim
        self.abstraction_hidden_dims = args.abstraction_hidden_dims
        self.user_latent_dim = args.user_latent_dim
        self.item_latent_dim = args.item_latent_dim
        self.enc_dim = args.enc_dim
        self.state_dim = 3*args.enc_dim + self.abstraction_dim
        self.attn_n_head = args.attn_n_head
        self.action_hidden_dims = args.action_hidden_dims
        self.dropout_rate = args.dropout_rate
        BaseModel.__init__(self, args, reader_stats, device)
        
    def _define_params(self, args):
        super()._define_params(args)
        self.state_abstraction = DNN(3*self.enc_dim, self.abstraction_hidden_dims, self.abstraction_dim,
                                   dropout_rate=args.dropout_rate, do_batch_norm = True)

    def encode_state(self, feed_dict, B):
        return_dict = super().encode_state(feed_dict, B)
        # (B, 3*enc_dim)
        return_state = return_dict['state']
        # (B, abstraction_dim)
        # with torch.no_grad():
        abstraction_emb = self.state_abstraction(return_state)

        state = torch.cat([return_state, abstraction_emb], dim=1)
        return_dict['state'] = state
        
        return return_dict