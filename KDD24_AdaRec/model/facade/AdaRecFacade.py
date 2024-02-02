import torch
import torch.nn.functional as F
import random
import numpy as np
from model.facade.TwoRewardFacade import TwoRewardFacade
import math
from torch.autograd import Variable

class AdaRecFacade(TwoRewardFacade):

    @staticmethod
    def parse_model_args(parser):
        parser = TwoRewardFacade.parse_model_args(parser)
        parser.add_argument('--beta_UB', type=float, default=0.1, help='beta for computing Q upper bound')
        parser.add_argument('--delta', type=float, default=0.1, help='delta for OAC')
        # same: 1/-1 in all dims; level: different reward according to the frequency of each dim
        parser.add_argument('--im_rew_type', type=str, default='same', help='type of immediate reward')

        return parser

    def __init__(self, args, environment, actor, critics):
        super().__init__(args, environment, actor, critics)
        self.beta_UB = args.beta_UB
        self.delta = args.delta
        if args.im_rew_type == 'level':
            self.im_rew_type = [1, 1, 30, 50, 50, 50, -50]
        elif args.im_rew_type == 'neg':
            self.im_rew_type = [-1, 1, 30, 50, 50, 50, -50]

    def apply_policy(self, observation, policy_model, epsilon=0, do_explore=False, do_OAC=False):
        if do_explore and do_OAC:
            with torch.no_grad():
                out_dict = super().apply_policy(observation, policy_model, epsilon, False)
        else:
            out_dict = super().apply_policy(observation, policy_model, epsilon, False)
        if do_explore:
            action = out_dict['action']
            # sampling noise of action embedding
            if np.random.rand() < epsilon:
                action = torch.clamp(torch.rand_like(action)*self.noise_var, -self.noise_clip, self.noise_clip)
            else:
                if do_OAC:
                    state = out_dict['state']
                    action.requires_grad_()
                    Q1 = self.critics[0]({'state': state, 'action': action})['q']
                    Q2 = self.critics[1]({'state': state, 'action': action})['q']
                    mu_Q = (Q1 + Q2) / 2
                    sigma_Q = torch.abs(Q1 - Q2) / 2
                    Q_UB = mu_Q + self.beta_UB * sigma_Q
                    Q_UB.backward(torch.ones_like(Q_UB))
                    grad = action.grad
                    assert grad is not None
                    assert action.shape == grad.shape

                    denom = torch.sqrt(
                        torch.sum(
                            torch.mul(torch.pow(grad, 2), self.noise_var),
                        )
                    ) + 1e-5
                    mu_C = math.sqrt(2*self.delta) * torch.mul(self.noise_var, grad) / denom
                
                else:
                    mu_C = torch.zeros_like(action)

                action = action + mu_C + torch.clamp(torch.rand_like(action)*self.noise_var, 
                                                      -self.noise_clip, self.noise_clip)
            out_dict['action'] = action.detach()
        return out_dict