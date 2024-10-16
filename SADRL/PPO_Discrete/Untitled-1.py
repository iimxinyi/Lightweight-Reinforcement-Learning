import gym

# 创建CartPole环境并指定渲染模式
env = gym.make('CartPole-v1', render_mode="human")

# 初始化环境，返回初始状态
state = env.reset()

# 设置最大步数
max_steps = 1000

# 执行一个简单的回合
for step in range(max_steps):
    # 随机选择一个动作（0或1，分别表示向左或向右施加力）
    action = env.action_space.sample()

    # 执行动作，返回新状态、奖励、是否结束、和额外信息
    next_state, reward, done, truncated, info = env.step(action)
    
    # 如果游戏结束 (杆子倒了或者其他终止条件)
    if done or truncated:
        print(f"Episode finished after {step + 1} steps")
        break

# 关闭环境
env.close()
