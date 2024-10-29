import gym
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace

from PPO import PPOAgent, Normalization, RewardScaling


def seed_everywhere(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def train(para):
    # env = gym.make('CartPole-v1', render_mode='human')
    env = gym.make('CartPole-v1')
    state_dim, action_dim = env.observation_space.shape[0], env.action_space.n
    agent = PPOAgent(para, state_dim, action_dim)
    state_norm = Normalization(dimension=state_dim)
    reward_norm = Normalization(dimension=1) if para.use_reward_norm else None
    reward_scaling = RewardScaling(dimension=1, gamma=para.gamma) if para.use_reward_scaling else None
    
    update_step, total_step = 0, 0
    reward_episode = []

    for episode in range(para.EPISODE):
        reward_step = []
        state = env.reset(seed=para.SEED)
        state = state_norm(state[0]) if para.use_state_norm else state[0]
        reward_scaling.reset() if para.use_reward_scaling else None

        for step in range(para.STEP):
            state_value = agent.critic.forward(torch.tensor(state, dtype=torch.float32))
            store_state_value = state_value.detach().numpy()
            action, action_prob = agent.choose_action(state)
            next_state, reward, done, _, _ = env.step(action)
            reward_step.append(reward)
            next_state = state_norm(next_state) if para.use_state_norm else next_state
            reward = reward_norm(reward) if para.use_reward_norm else (reward_scaling(reward) if para.use_reward_scaling else reward)
            next_state_value = agent.critic.forward(torch.tensor(next_state, dtype=torch.float32))
            store_next_state_value = next_state_value.detach().numpy()
            truncated = (step == para.STEP - 1)

            agent.replay_buffer.store(state, action, action_prob, reward, next_state, truncated, done, store_state_value, store_next_state_value)

            state = next_state
            total_step = total_step + 1
            if done: break

        if total_step > para.mini_batch_size and episode % para.update_frequency == 0:
            agent.update(update_step)
            update_step = update_step + 1

        #===================Save and Print Reward per Episode===================
        reward_episode.append(np.sum(reward_step))
        print('Episode:{}, Reward:{}'.format(episode, np.sum(reward_step)))
        #===================Save and Print Reward per Episode===================

    np.save('reward_episode.npy', reward_episode)


if __name__ == '__main__':
    env_para = {
        'SEED': 1,
        'EPISODE': 2000,
        'STEP': 500,
    }

    agent_para = {
        # Reference: https://zhuanlan.zhihu.com/p/512327050
        'update_frequency': 2,  # update network every 'update_frequency' episodes
        'max_train_step': 2e6,  # Maximum number of training steps, used for lr decay.
        'batch_size': 2048,  # Batch size
        'mini_batch_size': 256,  # Minibatch size
        'hidden_layer': [32, 32, 32],  # The number of neurons in hidden layers of the neural network
        'lr_actor': 0.00001,  # Learning rate of actor
        'lr_critic': 0.00001,  # Learning rate of critic
        'gamma': 0.99,  # Discount factor
        'lamda': 0.99,  # GAE parameter
        'epsilon': 0.2,  # PPO clip parameter
        'K_epochs': 1,  # PPO parameter
        'use_adv_norm': True,  # Trick 1: advantage normalization
        'use_state_norm': True,  # Trick 2: state normalization
        'use_reward_norm': False,  # Trick 3: reward normalization
        'use_reward_scaling': True,  # Trick 4: reward scaling
        'entropy_coef': 0.01,  # Trick 5: policy entropy
        'use_lr_decay': True,  # Trick 6: learning rate Decay
        'use_grad_clip': True,  # Trick 7: gradient clip
        'use_orthogonal_init': True,  # Trick 8: orthogonal initialization
        'set_adam_eps': True,  # Trick 9: set Adam epsilon=1e-5
        'use_tanh': True,  # Trick 10: tanh activation function
    }
    para = SimpleNamespace(**env_para, **agent_para)
    seed_everywhere(para.SEED)
    train(para)
    plt.plot(np.load('reward_episode.npy'))
    plt.show()
    