import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import *
from torch.distributions import Categorical


class ReplayBuffer:
    def __init__(self, para, max_step, num_agent, observation_dim, state_dim):
        self.batch_size = para.batch_size
        self.max_step = max_step
        self.num_agent = num_agent
        self.observation_dim = observation_dim
        self.state_dim = state_dim
        self.buffer = None
        self.reset_buffer()

    def reset_buffer(self):
        self.position = 0
        self.buffer = {'all_observation': np.zeros([self.batch_size, self.max_step, self.num_agent, self.observation_dim]),
                       'state': np.zeros([self.batch_size, self.max_step, self.state_dim]),
                       'all_state_value': np.zeros([self.batch_size, self.max_step + 1, self.num_agent]),
                       'all_action': np.zeros([self.batch_size, self.max_step, self.num_agent]),
                       'all_action_logprob': np.zeros([self.batch_size, self.max_step, self.num_agent]),
                       'all_reward': np.zeros([self.batch_size, self.max_step, self.num_agent]),
                       'all_done': np.zeros([self.batch_size, self.max_step, self.num_agent]),
                       'all_truncated': np.zeros([self.batch_size, self.max_step, self.num_agent])
                        }
        
    def store(self, step, all_observation, state, all_state_value, all_action, all_action_logprob, all_reward, all_done, all_truncated):
        index = self.position % self.batch_size
        self.buffer['all_observation'][index][step] = all_observation
        self.buffer['state'][index][step] = state
        self.buffer['all_state_value'][index][step] = all_state_value
        self.buffer['all_action'][index][step] = all_action
        self.buffer['all_action_logprob'][index][step] = all_action_logprob
        self.buffer['all_reward'][index][step] = all_reward
        self.buffer['all_done'][index][step] = all_done
        self.buffer['all_truncated'][index][step] = all_truncated

    def store_last_state_value(self, all_state_value):
        index = self.position % self.batch_size
        self.buffer['all_state_value'][index][self.max_step] = all_state_value
        self.position +=  1

    def numpy_to_tensor(self):
        batch = {}
        batch['all_observation'] = torch.tensor(self.buffer['all_observation'], dtype=torch.float)
        batch['state'] = torch.tensor(self.buffer['state'], dtype=torch.float)
        batch['all_state_value'] = torch.tensor(self.buffer['all_state_value'], dtype = torch.float)
        batch['all_action'] = torch.tensor(self.buffer['all_action'], dtype = torch.long)
        batch['all_action_logprob'] = torch.tensor(self.buffer['all_action_logprob'], dtype = torch.float)
        batch['all_reward'] = torch.tensor(self.buffer['all_reward'], dtype = torch.float)
        batch['all_done'] = torch.tensor(self.buffer['all_done'], dtype = torch.float)
        batch['all_truncated'] = torch.tensor(self.buffer['all_truncated'], dtype = torch.float)
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
    

class MAPPO:
    def __init__(self, para, max_step, num_agent, observation_dim, state_dim, action_dim):
        self.max_step = max_step
        self.num_agent = num_agent
        self.observation_dim = observation_dim
        self.state_dim = state_dim
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
        self.add_agent_id = para.add_agent_id
        self.use_value_clip = para.use_value_clip

        self.actor_input_dim = self.observation_dim
        self.critic_input_dim = self.state_dim
        if self.add_agent_id:
            self.actor_input_dim += self.num_agent
            self.critic_input_dim += self.num_agent
        
        self.actors = nn.ModuleList([Actor(para, self.actor_input_dim, self.action_dim) for _ in range(self.num_agent)])
        self.critic = Critic(para, self.critic_input_dim)

        self.ac_parameters = list(self.critic.parameters()) + [param for actor in self.actors for param in actor.parameters()]

        if self.set_adam_eps:
            self.ac_optimizer = torch.optim.Adam(self.ac_parameters, lr=self.learning_rate, eps=1e-5)
        else:
            self.ac_optimizer = torch.optim.Adam(self.ac_parameters, lr=self.learning_rate)

        self.replay_buffer = ReplayBuffer(para, self.max_step, self.num_agent, self.observation_dim, self.state_dim)

    def choose_action(self, all_observation):
        with torch.no_grad():
            all_action = []
            all_action_logprob = []
            for agent_id in range(self.num_agent):
                actor_input = torch.tensor(all_observation[agent_id], dtype=torch.float).unsqueeze(0)  # (1, observation_dim)
                if self.add_agent_id:
                    agent_id_one_hot = torch.eye(self.num_agent)[agent_id].unsqueeze(0)  # (1, num_agent)
                    actor_input = torch.cat([actor_input, agent_id_one_hot], dim=-1)
                prob = self.actors[agent_id](actor_input)  # (1, action_dim)
                dist = Categorical(probs=prob)
                action = dist.sample()
                action_logprob = dist.log_prob(action)
                all_action.append(action.item())
                all_action_logprob.append(action_logprob.item())
            all_action = np.array(all_action)
            all_action_logprob = np.array(all_action_logprob)
            return all_action, all_action_logprob
    
    def get_state_value(self, state):
        with torch.no_grad():
            critic_input = torch.tensor(state, dtype=torch.float).unsqueeze(0).repeat(self.num_agent, 1)  # (num_agent, state_dim)
            if self.add_agent_id:
                agent_id_one_hot = torch.eye(self.num_agent)
                critic_input = torch.cat([critic_input, agent_id_one_hot], dim=-1)  # (num_agent, state_dim + num_agent)
            state_value = self.critic(critic_input)
            return state_value.squeeze(-1).numpy()
        
    def get_ac_inputs(self, batch):
        actor_inputs, critic_inputs = [], []
        actor_inputs.append(batch['all_observation'])
        critic_inputs.append(batch['state'].unsqueeze(2).repeat(1, 1, self.num_agent, 1))
        if self.add_agent_id:
            agent_id_one_hot = torch.eye(self.num_agent).unsqueeze(0).unsqueeze(0).repeat(self.batch_size, self.max_step, 1, 1)
            actor_inputs.append(agent_id_one_hot)
            critic_inputs.append(agent_id_one_hot)

        actor_inputs = torch.cat([x for x in actor_inputs], dim=-1)  # (batch_size, max_step, num_agent, actor_input_dim)
        critic_inputs = torch.cat([x for x in critic_inputs], dim=-1)  # (batch_size, max_step, num_agent, critic_input_dim)
        return actor_inputs, critic_inputs

    def update(self, update_steps):
        batch = self.replay_buffer.numpy_to_tensor()

        adv = []
        gae = 0
        with torch.no_grad():
            deltas = batch['all_reward'] + self.gamma * (1 - batch['all_done']) * (1 - batch['all_truncated']) * batch['all_state_value'][:, 1:] - batch['all_state_value'][:, :-1]  # (batch_size, max_step, num_agent)
            for step in reversed(range(self.max_step)):
                gae = deltas[:,step] + self.gamma * self.lamda * gae
                adv.insert(0, gae)
            adv = torch.stack(adv, dim=1)  # (batch_size, max_step, num_agent)
            values_target = adv + batch['all_state_value'][:, :-1]
            if self.use_adv_norm:
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        actor_inputs, critic_inputs = self.get_ac_inputs(batch)

        for _ in range(self.K_epochs):
            for index in BatchSampler(SequentialSampler(range(self.batch_size)), self.mini_batch_size, False):
                values_now = self.critic(critic_inputs[index]).squeeze(-1)

                actor_losses = []
                critic_losses = []
                for agent_id in range(self.num_agent):
                    probs_now = self.actors[agent_id](actor_inputs[index][:, agent_id])  # Get individual actor's probabilities
                    dist_now = Categorical(probs_now)
                    dist_entropy = dist_now.entropy()
                    all_action_logprob_now = dist_now.log_prob(batch['all_action'][index][:, agent_id])
                    ratios = torch.exp(all_action_logprob_now - batch['all_action_logprob'][index][:, agent_id].detach())
                    surr1 = ratios * adv[index][:, agent_id]
                    surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index][:, agent_id]
                    actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy
                    actor_losses.append(actor_loss)

                critic_loss = (values_now - values_target[index]) ** 2
                critic_losses.append(critic_loss)

                self.ac_optimizer.zero_grad()
                ac_loss = torch.stack(actor_losses).mean() + torch.stack(critic_losses).mean()
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