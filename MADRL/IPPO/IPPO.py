import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import *
from torch.distributions import Categorical


class ReplayBuffer:
    def __init__(self, para, max_step, num_agent, observation_dim):
        self.batch_size = para.batch_size
        self.max_step = max_step
        self.num_agent = num_agent
        self.observation_dim = observation_dim
        self.buffer = None
        self.reset_buffer()

    def reset_buffer(self):
        self.position = 0
        self.buffer = {'observation': np.zeros([self.batch_size, self.max_step, self.observation_dim + self.num_agent]),
                       'observation_value': np.zeros([self.batch_size, self.max_step + 1]),
                       'action': np.zeros([self.batch_size, self.max_step]),
                       'action_logprob': np.zeros([self.batch_size, self.max_step]),
                       'reward': np.zeros([self.batch_size, self.max_step]),
                       'done': np.zeros([self.batch_size, self.max_step]),
                       'truncated': np.zeros([self.batch_size, self.max_step])
                        }
        
    def store(self, step, agent_id, observation, observation_value, action, action_logprob, reward, done, truncated):
        index = self.position % self.batch_size
        one_hot_agent_id = np.zeros(self.num_agent)
        one_hot_agent_id[agent_id] = 1
        new_observation = np.hstack((observation, one_hot_agent_id))
        self.buffer['observation'][index][step] = new_observation
        self.buffer['observation_value'][index][step] = observation_value
        self.buffer['action'][index][step] = action
        self.buffer['action_logprob'][index][step] = action_logprob
        self.buffer['reward'][index][step] = reward
        self.buffer['done'][index][step] = done
        self.buffer['truncated'][index][step] = truncated

    def store_last_state_value(self, observation_value):
        index = self.position % self.batch_size
        self.buffer['observation_value'][index][self.max_step] = observation_value
        self.position +=  1

    def numpy_to_tensor(self):
        batch = {}
        batch['observation'] = torch.tensor(self.buffer['observation'], dtype=torch.float)
        batch['observation_value'] = torch.tensor(self.buffer['observation_value'], dtype = torch.float)
        batch['action'] = torch.tensor(self.buffer['action'], dtype = torch.long)
        batch['action_logprob'] = torch.tensor(self.buffer['action_logprob'], dtype = torch.float)
        batch['reward'] = torch.tensor(self.buffer['reward'], dtype = torch.float)
        batch['done'] = torch.tensor(self.buffer['done'], dtype = torch.float)
        batch['truncated'] = torch.tensor(self.buffer['truncated'], dtype = torch.float)
        return batch


class RunningMeanStd:
    # URL: https://zhuanlan.zhihu.com/p/512327050
    def __init__(self, dimension):
        self.count = 0
        self.mean = np.zeros(dimension)
        self.variance_sum = np.zeros(dimension)
        self.standard_deviation = np.sqrt(self.variance_sum)

    def update(self, new_data):
        new_data = np.array(new_data)
        self.count += 1
        if self.count == 1:
            self.mean = new_data
            self.standard_deviation = new_data
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (new_data - old_mean) / self.count
            self.variance_sum = self.variance_sum + (new_data - old_mean) * (new_data - self.mean)
            self.standard_deviation = np.sqrt(self.variance_sum / self.count)


class Normalization:
    def __init__(self, dimension):
        self.running_mean_std = RunningMeanStd(dimension)

    def __call__(self, data, update=True):
        if update:
            self.running_mean_std.update(data)
        new_data = (data - self.running_mean_std.mean) / (self.running_mean_std.standard_deviation + 1e-8)
        return new_data
    

class RewardScaling:
    def __init__(self, dimension, gamma):
        self.dimension = dimension
        self.gamma = gamma
        self.running_mean_std = RunningMeanStd(dimension)
        self.cumulative_reward = np.zeros(self.dimension)

    def __call__(self, reward):
        self.cumulative_reward = self.gamma * self.cumulative_reward + reward
        self.running_mean_std.update(self.cumulative_reward)
        new_reward = reward / (self.running_mean_std.standard_deviation + 1e-8)
        return new_reward
    
    def reset(self):
        self.cumulative_reward = np.zeros(self.dimension)
    

def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class Actor(nn.Module):
    def __init__(self, para, actor_input_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(actor_input_dim, para.hidden_layer[0])
        self.fc2 = nn.Linear(para.hidden_layer[0], para.hidden_layer[1])
        self.fc3 = nn.Linear(para.hidden_layer[1], para.hidden_layer[2])
        self.fc4 = nn.Linear(para.hidden_layer[2], action_dim)
        self.activate_func = [nn.ReLU(), nn.Tanh()][para.use_tanh]

        if para.use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)
            orthogonal_init(self.fc4, gain=0.01)

    def forward(self, actor_input_dim):
        x = self.activate_func(self.fc1(actor_input_dim))
        x = self.activate_func(self.fc2(x))
        x = self.activate_func(self.fc3(x))
        action_prob = torch.softmax(self.fc4(x), dim=-1)
        return action_prob
    

class Critic(nn.Module):
    def __init__(self, para, critic_input_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(critic_input_dim, para.hidden_layer[0])
        self.fc2 = nn.Linear(para.hidden_layer[0], para.hidden_layer[1])
        self.fc3 = nn.Linear(para.hidden_layer[1], para.hidden_layer[2])
        self.fc4 = nn.Linear(para.hidden_layer[2], 1)
        self.activate_func = [nn.ReLU(), nn.Tanh()][para.use_tanh]

        if para.use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)
            orthogonal_init(self.fc4)

    def forward(self, critic_input_dim):
        x = self.activate_func(self.fc1(critic_input_dim))
        x = self.activate_func(self.fc2(x))
        x = self.activate_func(self.fc3(x))
        state_value = self.fc4(x)
        return state_value
    

class IPPO:
    def __init__(self, para, max_step, num_agent, observation_dim, action_dim):
        self.max_step = max_step
        self.num_agent = num_agent
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.batch_size = para.batch_size
        self.mini_batch_size = para.mini_batch_size
        self.max_train_steps = para.max_train_steps
        self.learning_rate = para.learning_rate
        self.gamma = para.gamma
        self.lamda = para.lamda
        self.epsilon = para.epsilon
        self.K_epochs = para.K_epochs
        self.entropy_coef = para.entropy_coef
        self.set_adam_eps = para.set_adam_eps
        self.use_grad_clip = para.use_grad_clip
        self.use_lr_decay = para.use_lr_decay
        self.use_adv_norm = para.use_adv_norm
        self.use_value_clip = para.use_value_clip

        self.actor_input_dim = self.observation_dim + self.num_agent
        self.critic_input_dim = self.observation_dim + self.num_agent
        
        self.actor = Actor(para, self.actor_input_dim, self.action_dim)
        self.critic = Critic(para, self.critic_input_dim)
        self.ac_parameters = list(self.actor.parameters()) + list(self.critic.parameters())

        if self.set_adam_eps:
            self.ac_optimizer = torch.optim.Adam(self.ac_parameters, lr=self.learning_rate, eps=1e-5)
        else:
            self.ac_optimizer = torch.optim.Adam(self.ac_parameters, lr=self.learning_rate)

        self.replay_buffer = ReplayBuffer(para, self.max_step, self.num_agent, self.observation_dim)

    def choose_action(self, agent_id, observation):
        with torch.no_grad():
            observation = torch.tensor(observation, dtype=torch.float)
            one_hot_agent_id = torch.zeros(self.num_agent)
            one_hot_agent_id[agent_id] = 1
            new_observation = torch.hstack((observation, one_hot_agent_id))
            prob = self.actor(new_observation)
            dist = Categorical(probs=prob)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
            return action.numpy(), action_logprob.numpy()
        
    def get_observation_value(self, agent_id, observation):
        with torch.no_grad():
            observation = torch.tensor(observation, dtype=torch.float)
            one_hot_agent_id = torch.zeros(self.num_agent)
            one_hot_agent_id[agent_id] = 1
            new_observation = torch.hstack((observation, one_hot_agent_id))
            observation_value = self.critic(new_observation)
            return observation_value.numpy()
        
    def get_ac_inputs(self, batch):
        actor_inputs, critic_inputs = [], []
        actor_inputs.append(batch['observation'])
        critic_inputs.append(batch['observation'])
        actor_inputs = torch.cat([x for x in actor_inputs], dim=-1)
        critic_inputs = torch.cat([x for x in critic_inputs], dim=-1)
        return actor_inputs, critic_inputs

    def update(self, update_steps):
        batch = self.replay_buffer.numpy_to_tensor()

        adv = []
        gae = 0
        with torch.no_grad():
            deltas = batch['reward'] + self.gamma * (1 - batch['done']) * (1 - batch['truncated']) * batch['observation_value'][:, 1:] - batch['observation_value'][:, :-1]  # (batch_size, max_step)
            for step in reversed(range(self.max_step)):
                gae = deltas[:,step] + self.gamma * self.lamda * gae
                adv.insert(0, gae)
            adv = torch.stack(adv, dim=1)  # (batch_size, max_step)
            values_target = adv + batch['observation_value'][:, :-1]
            if self.use_adv_norm:
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        actor_inputs, critic_inputs = self.get_ac_inputs(batch)

        for _ in range(self.K_epochs):
            for index in BatchSampler(SequentialSampler(range(self.batch_size)), self.mini_batch_size, False):
                probs_now = self.actor(actor_inputs[index])
                values_now = self.critic(critic_inputs[index]).squeeze(-1)

                dist_now = Categorical(probs_now)
                dist_entropy = dist_now.entropy()
                all_action_logprob_now = dist_now.log_prob(batch['action'][index])
                ratios = torch.exp(all_action_logprob_now - batch['action_logprob'][index].detach())
                surr1 = ratios * adv[index]
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy

                if self.use_value_clip:
                    values_old = batch['observation_value'][index, :-1].detach()
                    values_error_clip = torch.clamp(values_now - values_old, -self.epsilon, self.epsilon) + values_old - values_target[index]
                    values_error_original = values_now - values_target[index]
                    critic_loss = torch.max(values_error_clip ** 2, values_error_original ** 2)
                else:
                    critic_loss = (values_now - values_target[index]) ** 2

                self.ac_optimizer.zero_grad()
                ac_loss = actor_loss.mean() + critic_loss.mean()
                ac_loss.backward()
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.ac_parameters, 10.0)
                self.ac_optimizer.step()

        if self.use_lr_decay:
            self.lr_decay(update_steps)

    def lr_decay(self, update_steps):
        learning_rate_now = self.learning_rate * (1 - update_steps / self.max_train_steps)
        for p in self.ac_optimizer.param_groups:
            p['lr'] = learning_rate_now