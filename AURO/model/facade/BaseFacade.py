import torch
import torch.nn.functional as F
import random
import numpy as np

import utils
from model.facade.reward_func import *

class BaseFacade():
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
        - buffer_size
        - start_train_at_step
        - noise_var
        - noise_clip
        - reward_func
        '''
        parser.add_argument('--buffer_size', type=int, default=10000,
                            help='replay buffer size')
        parser.add_argument('--start_train_at_step', type=int, default=1000,
                            help='start timestamp for buffer sampling')
        parser.add_argument('--noise_var', type=float, default=0.1, 
                            help='noise magnitude for action embedding sampling')
        parser.add_argument('--noise_clip', type=float, default=1.0, 
                            help='noise magnitude for action embedding sampling')
        parser.add_argument('--reward_func', type=str, default='get_retention_reward', 
                            help='reward function name')
        return parser
        
    def __init__(self, args, environment, actor, critics):
        self.buffer_size = args.buffer_size
        self.start_train_at_step = args.start_train_at_step
        self.noise_var = args.noise_var
        self.noise_clip = args.noise_clip
        self.reward_func = eval(args.reward_func)
        super().__init__()
        
        self.device = args.device
        self.env = environment
        self.actor = actor
        self.critics = critics # RL agent may have multiple critics
        
        self.noise_decay = args.noise_var / args.n_iter[-1]
        
        self.immediate_response_weight = torch.FloatTensor([-1 if resp == 'is_hate' else 1 for resp in self.actor.feedback_types]).to(self.device)
        self.leave_response_weight = 0
        
        self.reward_history = []
        self.max_return_day = self.env.retention_stats['max_return_day']
        
    def initialize_train(self):
        '''
        Procedures before training
        '''
        # replay buffer
        self.buffer = self.env.generate_buffer(self.buffer_size, 
                                               state_dim = self.actor.state_dim, 
                                               action_dim = self.actor.action_dim)
        self.buffer_head = 0
        self.current_buffer_size = 0
        self.n_stream_record = 0
        self.is_training_available = False
        
    def reset_env(self, initial_params = {"batch_size": 1}):
        '''
        Reset user response environment
        '''
        initial_observation = self.env.reset(initial_params)
        return initial_observation
    
    def env_step(self, policy_output):
        action_dict = {'action': policy_output['action']}
        new_observation, user_feedback, updated_observation = self.env.step(action_dict)
        user_feedback['immediate_response_weight'] = self.immediate_response_weight
        user_feedback['leave_weight'] = self.leave_response_weight
        user_feedback['reward'] = self.reward_func(user_feedback)
        return new_observation, user_feedback, updated_observation
    
    def stop_env(self):
        self.env.stop()
    
    def get_episode_report(self, n_recent = 10):
#         recent_rewards = self.reward_history[-n_recent*self.env.max_n_session*self.env.initial_temper:]
        # (n_recent, B)
        recent_retention = np.array(self.env.user_return_history)[-n_recent:]
        episode_report = {'average_return_day': np.mean(recent_retention), 
                          'min_retention': np.min(recent_retention), 
                          'max_retention': np.max(recent_retention)}
        episode_report.update({f'retention_day{d}': np.mean(recent_retention <= d) for d in range(1,self.max_return_day)})
        return episode_report
    
    def apply_critic(self, observation, policy_output, critic_model):
        feed_dict = {"state": policy_output["state"], 
                     "action": policy_output["action"]}
        critic_output = critic_model(feed_dict)
        return critic_output
    
    def apply_policy(self, observation, policy_model, epsilon = 0, do_explore = False):
        '''
        @input:
        - observation: {'user_profile': {'user_id': (B,), 
                                         'uf_{feature_name}': (B, feature_dim)}, 
                        'user_history': {'history': (B, max_H), 
                                         'history_if_{feature_name}': (B, max_H * feature_dim), 
                                         'history_{response}': (B, max_H), 
                                         'history_length': (B,)}}
        - policy_model: the actor
        - epsilon: scalar for greedy epsilon
        - do_explore: exploration flag
        @output:
        - out_dict: {'state': (B, state_dim), 'action': (B, action_dim)}
        '''
        batch = {}
        batch.update(observation['user_profile'])
        batch.update(observation['user_history'])
        out_dict = policy_model(utils.wrap_batch(batch, self.device))
        if do_explore:
            action = out_dict['action']
            # sampling noise of action embedding
            if np.random.rand() < epsilon:
                action = torch.clamp(torch.rand_like(action)*self.noise_var, -self.noise_clip, self.noise_clip)
            else:
                action = action + torch.clamp(torch.rand_like(action)*self.noise_var, 
                                                      -self.noise_clip, self.noise_clip)
            out_dict['action'] = action
        return out_dict
    
    def sample_buffer(self, batch_size):
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
                   'user_response': {'done': (L,), 'retention':, (L,)}}
        '''
        indices = np.random.randint(0, self.current_buffer_size, size = batch_size)
        profile = {k:v[indices] for k,v in self.buffer["observation"]["user_profile"].items()}
        history = {k:v[indices] for k,v in self.buffer["observation"]["user_history"].items()}
        observation = {"user_profile": profile, "user_history": history}
        profile = {k:v[indices] for k,v in self.buffer["next_observation"]["user_profile"].items()}
        history = {k:v[indices] for k,v in self.buffer["next_observation"]["user_history"].items()}
        next_observation = {"user_profile": profile, "user_history": history}
        policy_output = {"state": self.buffer["policy_output"]["state"][indices], 
                         "action": self.buffer["policy_output"]["action"][indices]}
        reward = self.buffer["reward"][indices]
        done_mask = self.buffer["user_response"]["done"][indices]
        return observation, policy_output, reward, done_mask, next_observation
    
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
        - user_response: {'done': (B,), 'retention':, (B,)}}
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
        
        # buffer pointer
        self.buffer_head = (self.buffer_head + B) % self.buffer_size
        self.n_stream_record += B
        self.current_buffer_size = min(self.n_stream_record, self.buffer_size)
        # training is available when sufficient sample bufferred
        if self.n_stream_record >= self.start_train_at_step:
            self.is_training_available = True
        