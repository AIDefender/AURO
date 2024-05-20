import torch

def get_immediate_reward(user_feedback):
    '''
    @input:
    - user_feedback: {'immediate_response': (B, slate_size, n_feedback), 
                      'immediate_response_weight': (n_feedback),
                      'leave': (B,), 
                      'leave_weight': scalar,
                      ... other feedbacks}
    @output:
    - reward: (B,)
    '''
    # (B, slate_size, n_feedback
    point_reward = user_feedback['immediate_response'] * user_feedback['immediate_response_weight'].view(1,1,-1)
    combined_reward = torch.mean(point_reward, dim = 2)
    # (B,)
    #leave_reward = user_feedback['leave'] * user_feedback['leave_weight']
    # (B,)
    #reward = point_reward.sum(dim = -1) + leave_reward
    reward = torch.mean(combined_reward, dim = 1)
    return reward
    
def get_retention_reward(user_feedback, reward_base = 0.7):
    '''
    @input:
    - user_feedback: {'retention': (B,)}
    @output:
    - reward: (B,)
    '''
    reward = - user_feedback['retention']/10.0
    return reward

def get_big_retention_reward(user_feedback, reward_base = 0.7):
    '''
    @input:
    - user_feedback: {'retention': (B,)}
    @output:
    - reward: (B,)
    '''
    reward = 1 - user_feedback['retention']
    return reward

def get_power_retention_reward(user_feedback, reward_base = 0.7):
    '''
    @input:
    - user_feedback: {'retention': (B,)}
    @output:
    - reward: (B,)
    '''
    reward = reward_base ** user_feedback['retention']
    return reward
    
def get_immediate_and_retention_reward(user_feedback, retention_reward_base = 0.7):
    '''
    @input:
    - user_feedback: {'immediate': {'is_click': (B, K}, ...}, 'leave': (B,), 'return_day': (B,)}
    '''
    im = get_immediate_reward(user_feedback, is_combined)
    rt = get_retention_reward(user_feedback, retention_reward_base)
    return im/90000.0 + rt