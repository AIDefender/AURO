# from model.simulator.KRUserRetention import KRUserRetention
from model.general import BaseModel
from model.components import DNN
import torch
import torch.nn as nn

class KRUserRetention_NoSeq_Im(BaseModel):
    '''
    KuaiRand user retention model trained from immediate response
    '''
    
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - sess_enc_dim
        - attn_n_head
        - transformer_d_forward
        - transformer_n_layer
        - output_hidden_dims
        - dropout_rate
        - from BaseModel:
            - model_path
            - loss
            - l2_coef
        '''
        parser = BaseModel.parse_model_args(parser)
        
        parser.add_argument('--enc_dim', type=int, default=32, 
                            help='session encoding size')
        parser.add_argument('--attn_n_head', type=int, default=4, 
                            help='number of attention heads in transformer')
        parser.add_argument('--transformer_d_forward', type=int, default=64, 
                            help='forward layer dimension in transformer')
        parser.add_argument('--transformer_n_layer', type=int, default=2, 
                            help='number of encoder layers in transformer')
        parser.add_argument('--output_hidden_dims', type=int, nargs='+', default=[128], 
                            help='hidden dimensions')
        parser.add_argument('--dropout_rate', type=float, default=0.1, 
                            help='dropout rate in deep layers')
        return parser
        
    def __init__(self, args, reader_stats, device):
        self.enc_dim = args.enc_dim
        self.attn_n_head = args.attn_n_head
        self.transformer_d_forward = args.transformer_d_forward
        self.transformer_n_layer = args.transformer_n_layer
        self.output_hidden_dims = args.output_hidden_dims
        self.dropout_rate = args.dropout_rate
        super().__init__(args, reader_stats, device)
        self.ce_loss = nn.CrossEntropyLoss(reduction = 'none')
        
    def to(self, device):
        new_self = super(KRUserRetention_NoSeq_Im, self).to(device)
        new_self.attn_mask = new_self.attn_mask.to(device)
        self.dayBias = self.dayBias.to(device)
        return new_self

    def _define_params(self, args):
        stats = self.reader_stats
        
#         self.n_user = stats['n_user']
        # self.sess_emb_dim = stats['enc_dim']
        self.state_dim = args.enc_dim
        # self.output_dim = stats['max_return_day']
        self.output_dim = 10
        
        # session embedding kernel encoder
        # self.sessFeatureKernel = nn.Linear(self.sess_emb_dim, args.enc_dim)
        self.encDropout = nn.Dropout(self.dropout_rate)
        self.encNorm = nn.LayerNorm(args.enc_dim)
        
        # output DNN
        self.outputModule = DNN(self.state_dim, args.output_hidden_dims, self.output_dim, 
                                dropout_rate = args.dropout_rate, do_batch_norm = True)
        
        self.dayBias = torch.FloatTensor([-0.155-0.523*i for i in range(self.output_dim)])
        self.dayBias.requires_grad = False

        # response embedding

        encoder_layer = nn.TransformerEncoderLayer(d_model=args.enc_dim, dim_feedforward = args.transformer_d_forward, 
                                                   nhead=args.attn_n_head, dropout = args.dropout_rate, 
                                                   batch_first = True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=args.transformer_n_layer)
        
        
        self.max_len = stats['max_seq_len']
        self.feedback_types = stats['feedback_type']
        self.feedback_dim = stats['feedback_size']
        self.feedbackEncoder = nn.Linear(self.feedback_dim, args.enc_dim)
        self.attn_mask = ~torch.tril(torch.ones((self.max_len,self.max_len), dtype=torch.bool))

    def get_forward(self, feed_dict: dict):
        '''
        @input:
        - feed_dict: {
            'user_id': (B,)
            'return_day': (B,)
            'history_encoding': (B,max_H,enc_dim)
            'history_response': (B,max_H)
            ... (irrelevant input)
        }
        @output:
        - out_dict: {'preds': (B,output_dim), 'state': (B,state_dim), 'reg': scalar}
        '''
        # (B,)
        users = feed_dict['user_id']
        B = users.shape[0]
        
        # user encoding
        state_encoder_output = self.encode_state(feed_dict, B)
        # (B, state_dim)
        user_state = state_encoder_output['state'].view(B,self.state_dim)

        # output
        # (B, output_dim)
        preds = self.outputModule(user_state)
        preds = preds + self.dayBias.view(1,self.output_dim)

        # regularization terms
        reg = self.get_regularization(self.transformer, self.outputModule)
        reg = reg + state_encoder_output['reg']
        
        return {'preds': preds, 'state': user_state, 'reg': reg}
    
    def encode_state(self, feed_dict, B):
        '''
        @input:
        - feed_dict: {
            'user_id': (B,)
            'sess_encoding': (B,sess_emb_dim)
            'history_encoding': (B,max_H,sess_emb_dim)
            'history_response': (B,max_H)
            ... (irrelevant input)
        }
        - B: batch size
        @output:
        - out_dict:{
            'state': (B,enc_dim)
            'reg': scalar
        }
        '''
        # user history
        # (B, sess_emb_dim)
        # sess_emb = feed_dict['sess_encoding'].view(B, self.sess_emb_dim)
        # print(feed_dict['history_response'].shape)
        # print(torch.mean(feed_dict['history_response'].to(dtype=torch.float64), axis=0))
        
        # (B, enc_dim)
        # sess_enc = self.sessFeatureKernel(sess_emb)
        resp_emb = self.get_response_embedding(feed_dict['history_response'], B)
        output_seq = self.transformer(resp_emb, mask = self.attn_mask)
        state = output_seq[:,-1,:].view(B,self.enc_dim)

        # state = self.encNorm(self.encDropout(output_seq))
        return {'state': state, 'reg': 0.}

    def get_response_embedding(self, feed_dict, B):
        resp_list = []
        for f in self.feedback_types:
            # (B, max_H)
            resp = feed_dict[f'history_{f}'].view(B, self.max_len)
            resp_list.append(resp)
        # (B, max_H, n_feedback)
        combined_resp = torch.cat(resp_list, -1).view(B,self.max_len,self.feedback_dim)
        # (B, max_H, i_latent_dim)
        resp_emb = self.feedbackEncoder(combined_resp)
        return resp_emb 
    
    def get_loss(self, feed_dict: dict, out_dict: dict):
        """
        @input:
        - feed_dict: {...}
        - out_dict: {"preds":, "reg":}
        
        Loss terms implemented:
        - BCE
        """
        B = feed_dict['user_id'].shape[0]
        # (B, output_dim)
        preds = out_dict['preds'].view(B,self.output_dim)
        # (B,)
        targets = feed_dict['return_day']
        
        if self.loss_type == 'ce':
            loss = self.ce_loss(torch.softmax(preds, dim = -1), targets)
        else:
            raise NotImplemented
        out_dict['loss'] = torch.mean(loss) + self.l2_coef * out_dict['reg']
        return out_dict