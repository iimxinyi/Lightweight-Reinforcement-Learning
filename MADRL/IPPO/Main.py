import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
from pettingzoo.mpe import simple_spread_v3

from IPPO import Normalization, RewardScaling, IPPO


def seed_everywhere(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def train(para):
    # env = simple_spread_v3.parallel_env(N=para.num_agent, local_ratio=0.5, max_cycles=25, continuous_actions=False, render_mode='human')
    env = simple_spread_v3.parallel_env(N=para.num_agent, local_ratio=0.5, max_cycles=25, continuous_actions=False)
    all_observation_dim = [env.observation_space(f'agent_{i}').shape[0] for i in range(para.num_agent)]
    all_action_dim = [env.action_space(f'agent_{i}').n for i in range(para.num_agent)]
    observation_dim = all_observation_dim[0]
    action_dim = all_action_dim[0]

    ippo_agent = IPPO(para, para.STEP, para.num_agent, observation_dim, action_dim)

    observation_norm = Normalization(dimension=observation_dim)
    reward_norm = Normalization(dimension=para.num_agent)
    reward_scaling = RewardScaling(dimension=para.num_agent, gamma=para.gamma)

    update_steps, total_steps, total_episodes = 0, 0, 0
    reward_episode = []

    for episode in range(para.EPISODE):
        reward_step = []
        all_observation_dict, _ = env.reset(seed=para.SEED)
        # =====Convert to our standard form ⬇️=====
        all_observation = [observation for observation in all_observation_dict.values()]
        # =====Convert to our standard form ⬆️=====
        
        all_observation = [observation_norm(observation) for observation in all_observation] if para.use_observation_norm else all_observation
        reward_scaling.reset() if para.use_reward_scaling else None

        for step in range(para.STEP):
            all_action, all_action_logprob, all_observation_value = [], [], []
            for agent in range(para.num_agent):
                action, action_logprob = ippo_agent.choose_action(agent, all_observation[agent])
                all_action.append(action)
                all_action_logprob.append(action_logprob)
                observation_value = ippo_agent.get_observation_value(agent, all_observation[agent])
                all_observation_value.append(observation_value)
            # =====Convert to our standard form ⬇️=====
            all_action_dist = {f'agent_{i}': all_action[i].copy() for i in range(para.num_agent)}
            all_next_observation_dict, all_reward_dict, all_done_dict, all_truncated_dict, _ = env.step(all_action_dist)
            all_next_observation = [observation for observation in all_next_observation_dict.values()]
            all_reward = [reward for reward in all_reward_dict.values()]
            all_done = [done for done in all_done_dict.values()]
            all_truncated = [True] * len(all_truncated_dict) if step == para.STEP else [truncated for truncated in all_truncated_dict.values()]
            # =====Convert to our standard form ⬆️=====

            reward_step.append(np.mean(all_reward))

            all_observation = [observation_norm(observation) for observation in all_observation] if para.use_observation_norm else all_observation
            all_reward = reward_norm(all_reward) if para.use_reward_norm else (reward_scaling(all_reward) if para.use_reward_scaling else all_reward)

            for agent in range(para.num_agent):
                ippo_agent.replay_buffer.store(step, agent, all_observation[agent], all_observation_value[agent], all_action[agent], all_action_logprob[agent], all_reward[agent], all_done[agent], all_truncated[agent])

            all_observation = all_next_observation
            total_steps += 1
            
            if all(all_done) or all(all_truncated): break

        for agent in range(para.num_agent):
            observation_value = ippo_agent.get_observation_value(agent, all_observation[agent])
            ippo_agent.replay_buffer.store_last_state_value(observation_value)

        total_episodes += 1

        if total_episodes > para.mini_batch_size and episode % para.update_frequency == 0:
            ippo_agent.update(update_steps)
            update_steps += 1

        #===================Save and Print Reward per Episode===================
        reward_episode.append(np.sum(reward_step))
        print('Episode:{}, Reward:{}'.format(episode, np.sum(reward_step)))
        #===================Save and Print Reward per Episode===================

    np.save('reward_episode.npy', reward_episode)


if __name__ == '__main__':
    env_para = {
        'SEED': 1,
        'EPISODE': 1000,
        'STEP': 30,
        'num_agent': 3,
    }

    agent_para = {
        # URL: https://zhuanlan.zhihu.com/p/512327050
        'update_frequency': 1,  # Update network every 'update_frequency' episodes
        'max_train_steps': 3e6,  # Maximum number of training steps, used for lr decay
        'batch_size': 32,  # Batch size (the number of episodes)
        'mini_batch_size': 8,  # Minibatch size (the number of episodes)
        'hidden_layer': [64, 64, 64],  # The number of neurons in hidden layers of the neural network
        'learning_rate': 5e-4,  # Learning rate of the actor and critic
        'gamma': 0.99,  # Discount factor
        'lamda': 0.95,  # GAE parameter
        'epsilon': 0.2,  # MAPPO clip parameter
        'K_epochs': 1,  # MAPPO parameter
        'use_adv_norm': True,  # Trick 1: Advantage normalization
        'use_observation_norm': False,  # Trick 2: Observation normalization
        'use_reward_norm': False,  # Trick 3: Reward normalization
        'use_reward_scaling': True,  # Trick 4: Reward scaling
        'entropy_coef': 0.01,  # Trick 5: Policy Entropy
        'use_lr_decay': True,  # Trick 6: Learning rate decay
        'use_grad_clip': True,  # Trick 7: Gradient clip
        'use_orthogonal_init': True,  # Trick 8: Orthogonal initialization
        'set_adam_eps': True,  # Trick 9: Set Adam epsilon=1e-5
        'use_tanh': True,  # Trick 10: Tanh activation function
        'use_value_clip': True,  # Whether to use value clip
    }

    para = SimpleNamespace(**env_para, **agent_para)
    seed_everywhere(para.SEED)
    train(para)
    plt.plot(np.load('reward_episode.npy'))
    plt.show()