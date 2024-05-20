import numpy as np
import pandas as pd
from tqdm import tqdm

from reader.KRMBSeqReader import KRMBSeqReader
from reader.RetentionDataReader import RetentionDataReader
from utils import padding_and_clip, get_onehot_vocab, get_multihot_vocab

class MBRetentionDataReader(KRMBSeqReader, RetentionDataReader):
    
    @staticmethod
    def parse_data_args(parser):
        parser = KRMBSeqReader.parse_data_args(parser)
        # Parsers from the retention data reader
        parser.add_argument('--max_sess_seq_len', type=int, default=100, 
                            help='maximum history length in the input sequence')
        parser.add_argument('--max_return_day', type=int, default=10, 
                            help='number of possible return_day for classification')
        # The specific training file for obtaining return time
        parser.add_argument('--retention_file', type=str, required=True,
                            help='path to the retention data file (csv)')

        return parser
    def __init__(self, args):
        super().__init__(args)
        
    def _read_data(self, args):
        KRMBSeqReader._read_data(self, args)
        
        print(f"Loading data files")
        self.log_data_ret = pd.read_csv(args.retention_file)
        self.log_data_ret[self.log_data_ret['return_day'] > 10] = 10
        
        self.users_ret = list(self.log_data_ret['user_id'].unique())
        self.user_id_vocab_ret = {uid: i+1 for i,uid in enumerate(self.users_ret)}
        
        self.user_history_ret = {uid: list(self.log_data_ret[self.log_data_ret['user_id'] == uid].index) for uid in self.users_ret}
        
        self.enc_dim = len([col for col in self.log_data_ret if 'session_enc_' in col])
        self.padding_return_day = 10
        
        # {'train': [row_id], 'val': [row_id], 'test': [row_id]}
        self.data_ret = self._sequence_holdout_ret(args)

    def _sequence_holdout_ret(self, args):
        print(f"sequence holdout for users (-1, {args.val_holdout_per_user}, {args.test_holdout_per_user})")
        data = {"train": [], "val": [], "test": []}
        for u in tqdm(self.users_ret):
            sub_df = self.log_data_ret[self.log_data_ret['user_id'] == u]
            n_train = len(sub_df) - args.val_holdout_per_user - args.test_holdout_per_user
            if n_train < 0.8 * len(sub_df):
                continue
            data['train'].append(list(sub_df.index[:n_train]))
            data['val'].append(list(sub_df.index[n_train:n_train+args.val_holdout_per_user]))
            data['test'].append(list(sub_df.index[-args.test_holdout_per_user:]))
        for k,v in data.items():
            data[k] = np.concatenate(v)
        return data
            
    def __len__(self):
        return min(len(self.data[self.phase]), len(self.data_ret[self.phase]))

    def __getitem__(self, idx):
        # history encoding is the same as session encoding
        # which is not used now.
        record = KRMBSeqReader.__getitem__(self, idx)
        
        row_id = self.data_ret[self.phase][idx]
        row = self.log_data_ret.iloc[row_id]
        
        user_id = row['user_id'] # raw user ID
        
        # (max_H,)
        H_rowIDs = [rid for rid in self.user_history_ret[user_id] if rid < row_id][-self.max_sess_seq_len:]
        # (max_H, enc_dim), scalar, (max_H,)
        hist_enc, hist_length, hist_response = self.get_user_history_ret(H_rowIDs)
        # (enc_dim,)
        current_sess_enc = np.array([row[f'session_enc_{dim}'] for dim in range(self.enc_dim)])
        
        record.update({
            'user_id': self.user_id_vocab_ret[row['user_id']], # encoded user ID
            'sess_encoding': current_sess_enc,
            'return_day': int(row['return_day'])-1,
            'history_encoding': hist_enc,
            'history_response': hist_response,
            'history_length': hist_length
        })
        return record
    
    def get_user_history_ret(self, H_rowIDs):
        L = len(H_rowIDs)
        if L == 0:
            # (max_H, enc_dim)
            history_encoding = np.zeros((self.max_sess_seq_len, self.enc_dim))
            # {resp_type: (max_H)}
            history_response = np.array([self.padding_return_day] * self.max_sess_seq_len)
        else:
            H = self.log_data_ret.iloc[H_rowIDs]
            pad_hist_encoding = np.zeros((self.max_sess_seq_len - L, self.enc_dim))
            real_hist_encoding = [np.array(H[f'session_enc_{dim}']).reshape((-1,1)) for dim in range(self.enc_dim)]
            real_hist_encoding = np.concatenate(real_hist_encoding, axis = 1)
            history_encoding = np.concatenate((pad_hist_encoding, real_hist_encoding), axis = 0)
            pad_hist_response = np.array([self.padding_return_day] * (self.max_sess_seq_len - L))
            real_hist_response = np.array(H['return_day'])-1
            history_response = np.concatenate((pad_hist_response, real_hist_response), axis = 0)
        return history_encoding, L, history_response.astype(int)

    def get_statistics(self):
        stats1 = KRMBSeqReader.get_statistics(self)
        stats2 = RetentionDataReader.get_statistics(self)
        stats1.update(stats2)
        
        return stats1