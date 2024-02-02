import torch
import torch.nn.functional as F
import random
import numpy as np

import utils
from model.facade.BaseFacade import BaseFacade
from model.facade.reward_func import *

class TwoRewardFacade(BaseFacade):
    '''
    The general interface for one-stage RL policies.
    Key components:
    - replay buffer
    - environment
    - actor
    - critic
    '''
    
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - from BaseFacade:
            - buffer_size
            - start_train_at_step
            - noise_var
            - noise_clip
            - reward_func
        '''
        parser = BaseFacade.parse_model_args(parser)
        parser.add_argument('--ret_reward_func', type=str, default='get_retention_reward',
                            help='retention reward function name')
        return parser
        
    def __init__(self, args, environment, actor, critics):
        '''
        self.buffer_size
        self.start_train_at_step
        self.noise_var
        self.noise_clip
        self.reward_func
        self.device
        self.env
        self.actor
        self.critics
        self.noise_decay
        self.immediate_response_weight
        self.leave_response_weight
        self.reward_history
        self.max_return_day
        '''
        super().__init__(args, environment, actor, critics)
        self.ret_reward_func = eval(args.ret_reward_func)    # retention reward function
#         self.ret_reward_func = reward_func.get_power_retention_reward    # retention reward function
        self.im_reward_func = get_immediate_reward   # immediate reward function
        
    def initialize_train(self):
        '''
        Procedures before training
        '''
        super().initialize_train()
        # ['reward'] for retention, ['im_reward'] for immediate responses
        self.buffer['im_reward'] = torch.zeros_like(self.buffer['reward'])
    
    def env_step(self, policy_output):
        action_dict = {'action': policy_output['action']}
        new_observation, user_feedback, updated_observation = self.env.step(action_dict)
        user_feedback['immediate_response_weight'] = self.immediate_response_weight
        user_feedback['leave_weight'] = self.leave_response_weight
        user_feedback['reward'] = self.ret_reward_func(user_feedback)
        user_feedback['im_reward'] = self.im_reward_func(user_feedback)
        return new_observation, user_feedback, updated_observation
    
    def sample_buffer(self, batch_size, buffer=None, buffer_size=None):
        '''
        Batch sample is organized as a tuple of (observation, policy_output, reward, done_mask, next_observation)
        
        Buffer:   {'observation': {'user_profile': {'user_id': (L,), 
                                                    'uf_{feature_name}': (L, feature_dim)}, 
                                   'user_history': {'history': (L, max_H), 
                                                    'history_if_{feature_name}': (L, max_H * feature_dim), 
                                                    'history_{response}': (L, max_H), 
                                                    'history_length': (L,)}}
                   'policy_output': {'state': (L, state_dim), 'action': (L, action_dim)}, 
                   'next_observation': same format as @output-buffer['observation'], 
                   'reward': (L,),
                   'im_reward': (L,)
                   'user_response': {'done': (L,), 'retention':, (L,)}}
        '''
        if buffer is None:
            buffer = self.buffer
        if buffer_size is None:
            buffer_size = self.current_buffer_size
        indices = np.random.randint(0, buffer_size, size = batch_size)
        profile = {k:v[indices] for k,v in buffer["observation"]["user_profile"].items()}
        history = {k:v[indices] for k,v in buffer["observation"]["user_history"].items()}
        observation = {"user_profile": profile, "user_history": history}
        profile = {k:v[indices] for k,v in buffer["next_observation"]["user_profile"].items()}
        history = {k:v[indices] for k,v in buffer["next_observation"]["user_history"].items()}
        next_observation = {"user_profile": profile, "user_history": history}
        policy_output = {"state": buffer["policy_output"]["state"][indices], 
                         "action": buffer["policy_output"]["action"][indices]}
        reward = buffer["reward"][indices]
        im_reward = buffer["im_reward"][indices]
        done_mask = buffer["user_response"]["done"][indices]
        return observation, policy_output, {"reward": reward, "im_reward": im_reward}, done_mask, next_observation
    
    def update_buffer(self, observation, policy_output, user_response, next_observation):
        '''
        @input:
        - observation: {'user_profile': {'user_id': (B,), 
                                         'uf_{feature_name}': (B, feature_dim)}, 
                        'user_history': {'history': (B, max_H), 
                                         'history_if_{feature_name}': (B, max_H * feature_dim), 
                                         'history_{response}': (B, max_H), 
                                         'history_length': (B,)}}
        - policy_output: {'state': (B, state_dim), 'action': (B, action_dim)}
        - user_response: {'done': (B,), 'retention':, (B,), 
                          'reward': (B,), 'im_reward': (B,)}}
        - next_observation: same format as @input-observation
        '''
        B = len(user_response['retention'])
        if self.buffer_head + B >= self.buffer_size:
            tail = self.buffer_size - self.buffer_head
            indices = [self.buffer_head + i for i in range(tail)] + \
                        [i for i in range(B - tail)]
        else:
            indices = [self.buffer_head + i for i in range(B)]
        indices = torch.tensor(indices).to(torch.long).to(self.device)
        # update buffer
        for k,v in observation['user_profile'].items():
#             print(k, v.shape, self.buffer['observation']['user_profile'][k].shape)
            self.buffer['observation']['user_profile'][k][indices] = v
        for k,v in observation['user_history'].items():
#             print(k, v.shape, self.buffer['observation']['user_history'][k].shape)
            self.buffer['observation']['user_history'][k][indices] = v
        for k,v in next_observation['user_profile'].items():
#             print(k, v.shape, self.buffer['next_observation']['user_profile'][k].shape)
            self.buffer['next_observation']['user_profile'][k][indices] = v
        for k,v in next_observation['user_history'].items():
#             print(k, v.shape, self.buffer['next_observation']['user_history'][k].shape)
            self.buffer['next_observation']['user_history'][k][indices] = v
#         input()
        self.buffer['policy_output']['state'][indices] = policy_output['state']
        self.buffer['policy_output']['action'][indices] = policy_output['action']
        self.buffer['user_response']['done'][indices] = user_response['done']
        self.buffer['user_response']['retention'][indices] = user_response['retention']
        self.buffer['reward'][indices] = user_response['reward']
        self.buffer['im_reward'][indices] = user_response['im_reward']
        
        # buffer pointer
        self.buffer_head = (self.buffer_head + B) % self.buffer_size
        self.n_stream_record += B
        self.current_buffer_size = min(self.n_stream_record, self.buffer_size)
        # training is available when sufficient sample bufferred
        if self.n_stream_record >= self.start_train_at_step:
            self.is_training_available = True
        