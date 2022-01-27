# Q - Learning 變數設定

import numpy as np
import pandas as pd
import time

N_STATES = 6   
ACTIONS = ['left', 'right']     
EPSILON = 0.9   
ALPHA = 0.1     
GAMMA = 0.9    
MAX_EPISODES = 13  
FRESH_TIME = 0.3  

# Q - Learning Q table 設定

def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),     
        columns=actions,   
    )
    return table

# q_table:
#"""
#   left  right
#0   0.0    0.0
#1   0.0    0.0
#2   0.0    0.0
#3   0.0    0.0
#4   0.0    0.0
#5   0.0    0.0
#"""

# Q - Learning 選擇 action
def choose_action(state, q_table):
    state_actions = q_table.iloc[state, :]  
    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):  
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.argmax()    
    return action_name
  
 # Q - Learning 環境反饋 state / reward
def get_env_feedback(S, A):
    # This is how agent will interact with the environment
    if A == 'right':    
        if S == N_STATES - 2:   
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:   
        R = 0
        if S == 0:
            S_ = S  
        else:
            S_ = S - 1
    return S_, R
  
# Q - Learning 環境更新

def update_env(S, episode, step_counter):
    # This is how environment be updated
    env_list = ['-']*(N_STATES-1) + ['T']   # '---------T' our environment
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)
        
 # Q - Learning 主循環

def rl():
    q_table = build_q_table(N_STATES, ACTIONS)  # 初始 q table
    for episode in range(MAX_EPISODES):     # 回合
        step_counter = 0
        S = 0   # 回合初始位置
        is_terminated = False   # 是否回合结束
        update_env(S, episode, step_counter)    
        while not is_terminated:

            A = choose_action(S, q_table)   
            S_, R = get_env_feedback(S, A)  
            q_predict = q_table.loc[S, A]    
            if S_ != 'terminal':
                q_target = R + GAMMA * q_table.iloc[S_, :].max()   
            else:
                q_target = R     
                is_terminated = True    

            q_table.loc[S, A] += ALPHA * (q_target - q_predict) 
            S = S_  

            update_env(S, episode, step_counter+1)  
            step_counter += 1
    return q_table
  
  
  # Q - Learning 啟動

if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)
