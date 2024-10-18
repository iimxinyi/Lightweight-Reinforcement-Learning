import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import Categorical


class ReplayBuffer:
    def __init__(self, para, state_dim):
        self.batch_size = para.batch_size
        self.state = np.zeros((self.batch_size, state_dim))
        self.action = np.zeros((self.batch_size, 1))
        self.action_logprob = np.zeros((self.batch_size, 1))
        self.reward = np.zeros((self.batch_size, 1))
        self.next_state = np.zeros((self.batch_size, state_dim))
        self.truncated = np.zeros((self.batch_size, 1))
        self.done = np.zeros((self.batch_size, 1))
        self.state_value = np.zeros((self.batch_size, 1))
        self.next_state_value = np.zeros((self.batch_size, 1))
        self.position = 0

    def store(self, state, action, action_logprob, reward, next_state, truncated, done, state_value, next_state_value):
        index = self.position % self.batch_size
        self.state[index] = state
        self.action[index] = action
        self.action_logprob[index] = action_logprob
        self.reward[index] = reward
        self.next_state[index] = next_state
        self.truncated[index] = truncated
        self.done[index] = done
        self.state_value[index] = state_value
        self.next_state_value[index] = next_state_value
        self.position = self.position + 1

    def numpy_to_tensor(self):
        state = torch.tensor(self.state, dtype=torch.float)
        action = torch.tensor(self.action, dtype=torch.long)
        action_logprob = torch.tensor(self.action_logprob, dtype=torch.float)
        reward = torch.tensor(self.reward, dtype=torch.float)
        next_state = torch.tensor(self.next_state, dtype=torch.float)
        truncated = torch.tensor(self.truncated, dtype=torch.float)
        done = torch.tensor(self.done, dtype=torch.float)
        state_value = torch.tensor(self.state_value, dtype=torch.float)
        next_state_value = torch.tensor(self.next_state_value, dtype=torch.float)

        return state, action, action_logprob, reward, next_state, truncated, done, state_value, next_state_value


class RunningMeanStd:
    # URL: https://github.com/Lizhi-sjtu/DRL-code-pytorch/blob/main/4.PPO-discrete/normalization.py
    def __init__(self, dimension):
        self.data_number = 0
        self.mean = np.zeros(dimension)
        self.S = np.zeros(dimension)
        self.std = np.sqrt(self.S)

    def update(self, data):
        data = np.array(data)
        self.data_number += 1
        if self.data_number == 1:
            self.mean = data
            self.std = data
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (data - old_mean) / self.data_number
            self.S = self.S + (data - old_mean) * (data - self.mean)
            self.std = np.sqrt(self.S / self.data_number)


class Normalization:
    def __init__(self, dimension):
        self.running_ms = RunningMeanStd(dimension)

    def __call__(self, data, update=True):
        if update:
            self.running_ms.update(data)
        new_data = (data - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return new_data
    

class RewardScaling:
    def __init__(self, dimension, gamma):
        self.dimension = dimension
        self.gamma = gamma
        self.running_ms = RunningMeanStd(dimension)
        self.R = np.zeros(self.dimension)

    def __call__(self, reward):
        self.R = self.gamma * self.R + reward
        self.running_ms.update(self.R)
        new_reward = reward / (self.running_ms.std + 1e-8)
        return new_reward
    
    def reset(self):
        self.R = np.zeros(self.dimension)
    

def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class Actor(nn.Module):
    def __init__(self, para, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, para.hidden_layer[0])
        self.fc2 = nn.Linear(para.hidden_layer[0], para.hidden_layer[1])
        self.fc3 = nn.Linear(para.hidden_layer[1], para.hidden_layer[2])
        self.fc4 = nn.Linear(para.hidden_layer[2], action_dim)
        self.activate_func = [nn.ReLU(), nn.Tanh()][para.use_tanh]

        if para.use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)
            orthogonal_init(self.fc4, gain=0.01)

    def forward(self, state):
        x = self.activate_func(self.fc1(state))
        x = self.activate_func(self.fc2(x))
        x = self.activate_func(self.fc3(x))
        action_prob = torch.softmax(self.fc4(x), dim=-1)
        
        return action_prob
    

class Critic(nn.Module):
    def __init__(self, para, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, para.hidden_layer[0])
        self.fc2 = nn.Linear(para.hidden_layer[0], para.hidden_layer[1])
        self.fc3 = nn.Linear(para.hidden_layer[1], para.hidden_layer[2])
        self.fc4 = nn.Linear(para.hidden_layer[2], 1)
        self.activate_func = [nn.ReLU(), nn.Tanh()][para.use_tanh]

        if para.use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)
            orthogonal_init(self.fc4)

    def forward(self, state):
        x = self.activate_func(self.fc1(state))
        x = self.activate_func(self.fc2(x))
        x = self.activate_func(self.fc3(x))
        state_value = self.fc4(x)
        return state_value


class PPOAgent:
    def __init__(self, para, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = para.batch_size
        self.mini_batch_size = para.mini_batch_size
        self.max_train_step = para.max_train_step
        self.lr_actor = para.lr_actor
        self.lr_critic = para.lr_critic
        self.gamma = para.gamma
        self.lamda = para.lamda
        self.epsilon = para.epsilon
        self.K_epochs = para.K_epochs
        self.entropy_coef = para.entropy_coef
        self.set_adam_eps = para.set_adam_eps
        self.use_grad_clip = para.use_grad_clip
        self.use_lr_decay = para.use_lr_decay
        self.use_adv_norm = para.use_adv_norm

        self.replay_buffer = ReplayBuffer(para, self.state_dim)
        self.actor = Actor(para, self.state_dim, self.action_dim)
        self.critic = Critic(para, self.state_dim)
        if self.set_adam_eps:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor, eps=1e-5)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic, eps=1e-5)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic)

    def choose_action(self, state):
        state = torch.unsqueeze(torch.tensor(state, dtype=torch.float), 0)
        with torch.no_grad():
            action_prob = self.actor(state)
            # print(action_prob)
            dist = Categorical(probs=action_prob)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
            return action.numpy()[0], action_logprob.numpy()[0]
        
    def update(self, total_steps):
        state, action, action_logprob, reward, next_state, truncated, done, state_value, next_state_value = self.replay_buffer.numpy_to_tensor()
        adv = []
        gae = 0
        with torch.no_grad():
            deltas = reward + self.gamma * (1.0 - done) * (1.0 - truncated) * next_state_value - state_value
            for delta, d, t in zip(reversed(deltas.flatten().numpy()), reversed(done.flatten().numpy()), reversed(truncated.flatten().numpy())):
                gae = delta + self.gamma * self.lamda * gae * (1.0 - d) * (1.0 - t)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1)
            target_value = adv + state_value
            if self.use_adv_norm:
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))
            
        for _ in range(self.K_epochs):
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False):
                dist_now = Categorical(probs=self.actor(state[index]))
                dist_entropy = dist_now.entropy().view(-1, 1)
                action_logprob_now = dist_now.log_prob(action[index].squeeze()).view(-1, 1)
                ratios = torch.exp(action_logprob_now - action_logprob[index])

                surr1 = ratios * adv[index]
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy  # mini_batch_size x 1
                # print(actor_loss.mean())

                self.optimizer_actor.zero_grad()
                actor_loss.mean().backward()
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()

                state_value_mini_batch = torch.tensor(state_value[index], dtype=torch.float32, requires_grad=True)
                critic_loss = F.mse_loss(target_value[index], state_value_mini_batch)
                # print(critic_loss.mean())

                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()

        if self.use_lr_decay:
            self.lr_decay(total_steps)

    def lr_decay(self, total_steps):
        lr_actor_now = self.lr_actor * (1 - total_steps / self.max_train_step)
        lr_critic_now = self.lr_critic * (1 - total_steps / self.max_train_step)
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_actor_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_critic_now
