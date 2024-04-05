import time
import copy
import torch
import torch.nn.functional as F
import numpy as np

import utils
from model.agent.TD3 import TD3
from model.UserEncoder import UserEncoder
from model.facade.TwoRewardFacade import TwoRewardFacade
from sklearn.cluster import KMeans
import os
    
class PAUR(TD3):
    
    @staticmethod
    def parse_model_args(parser):
        parser = TD3.parse_model_args(parser)
        parser.add_argument('--anneal_a_lr', type=bool, default=False, help='whether to anneal actor learning rate')
        parser.add_argument('--anneal_step', type=int, default=3000, help='number of steps to anneal actor learning rate')
        parser.add_argument('--annealed_lr', type=float, default=1e-6, help='annealed actor learning rate')
        parser.add_argument('--aux_loss_weight', type=float, default=0., help='weight of auxiliary loss')
        parser.add_argument('--n_clusters', type=int, default=4, help='number of clusters')
        parser.add_argument('--alpha', type=float, default=0.1, help='radis of the distance function')
        parser.add_argument('--lambda_', type=float, default=0.1, help='weight of the first part of the state abstraction function')
        parser.add_argument('--aux_loss_type', type=str, default='kmeans', help='type of auxiliary loss')

        return parser
    def __init__(self, args, facade):
        super().__init__(args, facade)
        self.n_step = 0
        self.anneal_a_lr = args.anneal_a_lr
        self.anneal_step = args.anneal_step
        self.annealed_lr = args.annealed_lr
        self.aux_loss_weight = args.aux_loss_weight
        if not hasattr(self.actor, 'state_abstraction'):
            self.aux_loss_weight = 0
        self.aux_loss_type = args.aux_loss_type
        self.n_clusters = args.n_clusters
        self.alpha = args.alpha
        self.delta = 1e-6
        self.lambda_ = args.lambda_
        if self.aux_loss_weight > 0:
            self.abstraction_optimizer = torch.optim.Adam(self.actor.state_abstraction.parameters(), lr=0.0003)

    def run_episode_step(self, *episode_args):
        
        '''
        One step of interaction
        '''
        episode_iter, epsilon, observation, do_buffer_update = episode_args
        self.epsilon = epsilon
        # sample action
        policy_output = self.facade.apply_policy(observation, self.actor, epsilon, do_explore = True, do_OAC=True)
        # print(policy_output['action'].shape)
        # apply action on environment and update replay buffer
        new_observation, user_feedback, updated_observation = self.facade.env_step(policy_output)
        # update replay buffer
        if do_buffer_update:
            self.facade.update_buffer(observation, policy_output, user_feedback, updated_observation)
        observation = new_observation
        return updated_observation

    def action_before_train(self):
        super().action_before_train()
        self.training_history["abstraction_loss"] = []

    def step_train(self):
        observation, policy_output, reward, done_mask, next_observation = self.facade.sample_buffer(self.batch_size)
        if isinstance(reward, dict):
            reward = reward['reward']
        reward = reward.to(torch.float)
        done_mask = done_mask.to(torch.float)
        
        critic_loss, actor_loss, abstraction_loss = self.get_td3_loss(observation, policy_output, reward, done_mask, next_observation)
        if hasattr(self.facade.actor, 'step') and self.facade.actor.step % 1000 == 0:
            output_path = self.facade.actor.state_output_path
            torch.save(self.facade.critics[0].state_dict(), os.path.join(output_path, f'critic1_{self.facade.actor.step}.pth'))
            torch.save(self.facade.critics[1].state_dict(), os.path.join(output_path, f'critic2_{self.facade.actor.step}.pth'))

        self.training_history['actor_loss'].append(actor_loss.item())
        self.training_history['abstraction_loss'].append(abstraction_loss.item())
        self.training_history['critic1_loss'].append(critic_loss[0])
        self.training_history['critic1'].append(critic_loss[1])
        self.training_history['critic2_loss'].append(critic_loss[2])
        self.training_history['critic2'].append(critic_loss[3])
        self.training_history['reward'].append(torch.mean(reward).item())

        # Update the frozen target models
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {"step_loss": (self.training_history['actor_loss'][-1], 
                              self.training_history['abstraction_loss'][-1],
                              self.training_history['critic1_loss'][-1], 
                              self.training_history['critic2_loss'][-1], 
                              self.training_history['critic1'][-1], 
                              self.training_history['critic2'][-1], 
                              self.training_history['reward'][-1])}
    

    def get_td3_loss(self, observation, policy_output, reward, done_mask, next_observation, do_actor_update=True, do_critic_update=True):
        critic_loss, actor_loss = super().get_td3_loss(observation, policy_output, reward, done_mask, next_observation, do_actor_update, do_critic_update)
        self.n_step += 1
        if self.n_step == self.anneal_step and self.anneal_a_lr:
            for param_group in self.actor_optimizer.param_groups:
                param_group['lr'] = self.annealed_lr
                
        # update the state abstraction module with auxiliary loss
        if self.aux_loss_weight > 0 and self.n_step % 10 == 0:
            self.abstraction_optimizer.zero_grad()
            abstraction_loss = self.get_abstraction_loss(observation, policy_output)
            abstraction_loss.backward()
            self.abstraction_optimizer.step()
        else:
            abstraction_loss = torch.Tensor([0])
        return critic_loss, actor_loss, abstraction_loss

    def get_abstraction_loss(self, observation, policy_output):
        with torch.no_grad():
            state = self.facade.apply_policy(observation, self.actor, self.epsilon, do_explore = False, do_OAC=False)['state'][:,:3*self.actor.enc_dim]
        B = state.shape[0]
        abstraction_output = self.actor.state_abstraction(state)

        if self.aux_loss_type == 'kmeans':
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(state.cpu().numpy())
            labels = kmeans.labels_
            label_set = set(kmeans.labels_)
        elif self.aux_loss_type[0] == 'Q':
            with torch.no_grad():
                q_value = self.facade.apply_critic(observation, policy_output, self.critic1)['q']
                q2 = self.facade.apply_critic(observation, policy_output, self.critic2)['q']
                if self.aux_loss_type == 'QMean':
                    q_value = (q_value+q2)/2
                elif self.aux_loss_type == 'QMax':
                    q_value = torch.max(q_value, q2)
                elif self.aux_loss_type == 'QMin':
                    q_value = torch.min(q_value, q2)
                sort_index = torch.argsort(q_value).cpu().numpy()
                labels = np.zeros(B, dtype=int)
                label_set = set(range(self.n_clusters))
                for i in range(self.n_clusters):
                    labels[sort_index[i*B//self.n_clusters:(i+1)*B//self.n_clusters]] = i

        z_hat = torch.concat([torch.mean(abstraction_output[labels==label], axis=0) for label in label_set]).reshape(-1, self.actor.abstraction_dim)
        loss = 0
            
        # Compute the determinant of the distance matrix
        # (n_clusters, abstraction_dim)
        z_dis_matrix = torch.exp(-self.alpha*(torch.norm(z_hat.unsqueeze(1)-z_hat.unsqueeze(0), dim=2)))
        # (n_clusters, n_clusters)
        det_z = torch.det(z_dis_matrix+self.delta)
        
        # Compute the distance to center of each cluster
        for i in range(B):
            loss += self.lambda_ * torch.norm(abstraction_output[i]-z_hat[labels[i]])
        
        loss += -torch.log(det_z)
        loss *= self.aux_loss_weight
        
        return loss