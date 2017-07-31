'''
this is compeleted
make structure more clear ; name more reseanable
reference http://edu.51cto.com/lecturer/8863971.html course
'''
from gridworld import GridworldEnv
import numpy as np

DISCOUNT_FACTOR=0.5 #折减率
THETA = 0.0001 #收敛阈值

env = GridworldEnv()
def value_iteration(env):
    V = np.zeros(env.nS)
    while True: 
        delta = 0
        # 更新状态
        for s in range(env.nS):
            # 预测一次下一步最好的动作
            A = one_step_lookahead(s, V)
            best_action_value = np.max(A)
            # 计算delta
            delta = max(delta, np.abs(best_action_value - V[s]))
            # 更新动作价值
            V[s] = best_action_value        
        # 如果小于阈值 就停止
        if delta < THETA:
            break
    
    #选择最好策略
    policy = np.zeros([env.nS, env.nA])
    for s in range(env.nS):
        A = one_step_lookahead(s, V)
        best_action = np.argmax(A)
        policy[s, best_action] = 1.0
    return policy, V

#返回价值
def one_step_lookahead(state, V):
    Q = np.zeros(env.nA)
    for a in range(env.nA):
        #四个方向各得分数
        for prob, next_state, reward, done in env.P[state][a]:
            Q[a] += prob * (reward + DISCOUNT_FACTOR * V[next_state])
    return Q

policy, v = value_iteration(env)

# print("Policy Probability Distribution:")
# print(policy)
# print("")

print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
print(np.reshape(np.argmax(policy, axis=1), env.shape))
print("")

