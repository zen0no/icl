# code inspired by https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/A3C/pytorch/a3c.py
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.multiprocessing as mp

import numpy as np

import os

import wandb

# This lines avoids non-determenistic behaviour of LSTM on CUDA
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
    if isinstance(m, nn.RNNCellBase):
        torch.nn.init.xavier_uniform_(m.weight_hh)
        torch.nn.init.xavier_uniform_(m.weight_ih)



class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
            weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps,
                weight_decay=weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


class NetWithMemory(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, device):
        super(NetWithMemory, self).__init__()

        self.device = device

        self.hidden_dim = hidden_dim

        self.h = torch.zeros(1, self.hidden_dim, device=device)
        self.c = torch.zeros(1, self.hidden_dim, device=device)

        self.embed = nn.Linear(input_dim, hidden_dim).to(device)
        self.lstm = nn.LSTMCell(hidden_dim, hidden_dim).to(device)
        self.output = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
            ).to(device)

        self.init_weights()

    def init_weights(self):
        self.embed.apply(init_weights)
        self.lstm.apply(init_weights)
        self.output.apply(init_weights)

    def forward(self, x):
        x = self.embed(x)
        h, c = self.lstm(x, (self.h, self.c))

        x = self.h = h
        self.c = c

        x = F.gelu(x)
        return self.output(x)

    def reset_hidden(self):
        self.h = torch.zeros(1, self.hidden_dim, device=self.device)
        self.c = torch.zeros(1, self.hidden_dim, device=self.device)


class A3C(nn.Module):
    def __init__(self, state_dim, action_dim, gamma, hidden_dim=256, device=None):
        super().__init__()

        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")

        self.gamma = gamma

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.pi = NetWithMemory(state_dim, hidden_dim, action_dim, device=self.device)
        self.v = NetWithMemory(state_dim, hidden_dim, 1, device=self.device)

        self.states = []
        self.actions = []
        self.rewards = []

    def remember(self, s, a, r):
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)

    def clear_memory(self):
        # reset hidden state of memory nets
        self.pi.reset_hidden()
        self.v.reset_hidden()

        self.states = []
        self.actions = []
        self.rewards = []

    def return_trajectory(self):
        return {
            "states": self.states[:],
            "actions": self.actions[:],
            "rewards": self.rewards
        }

    def forward(self, state):
        log_prob = self.pi(state)
        value = self.v(state)

        return log_prob, value

    def loss(self, done):
        states = torch.tensor(self.states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(self.actions, dtype=torch.float32, device=self.device)

        seq_len = states.shape[0]

        values = torch.zeros((seq_len, ), device=self.device)
        pi = torch.zeros((seq_len, self.action_dim), device=self.device)

        self.pi.reset_hidden()
        self.v.reset_hidden()

        for i in range(seq_len):
            pi_, values_ = self.forward(states[i].unsqueeze(0))
            pi[i] = torch.squeeze(pi_)
            values[i] = torch.squeeze(values_)

        # calculate returns and critic loss
        R = values[-1]*(1-int(done))
        returns = []
        for reward in self.rewards[::-1]:
            R = reward + self.gamma*R
            returns.append(R)
        returns.reverse()
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        values.squeeze()
        critic_loss = (returns - values) ** 2

        # calculate actor loss
        probs = torch.softmax(pi)
        d = Categorical(probs)
        log_prob = d.log_prob(actions)
        actor_loss = -log_prob*(returns - values)

        total_loss = (critic_loss + actor_loss).mean()

        return total_loss

    @torch.no_grad()
    def act(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        pi, v = self.forward(state)
        probs = torch.softmax(pi, dim=1)
        d = Categorical(probs)
        action = d.sample().detach().cpu().numpy()[0]

        return action
    

class Worker(mp.Process):
    def __init__(self, global_ac, optimizer, env,
                gamma, name, global_ep, t_max, max_episode, hidden_dim, data_queue, device=None):
        super(Worker, self).__init__()
        
        self.env = env
        self.optimizer = optimizer
        self.gamma = gamma

        state_dim = env.state_dim
        action_dim = env.action_dim

        self.local_ac = A3C(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim, gamma=gamma, device=device)
        self.global_actor_critic = global_ac
        self.name = 'w%02i' % name
        self.episode_idx = global_ep
        self.data_queue = data_queue

        self.max_episode = max_episode
        self.t_max = t_max

    def run(self):
        while self.episode_idx.value < self.max_episode:
            done = False
            obs, _ = self.env.reset()
            score = 0
            t_step = 1

            self.local_ac.clear_memory()
            while t_step % self.t_max != 0 and not done:
                print(t_step)
                print(t_step % self.t_max == 0)
                action = self.local_ac.act(obs)
                obs_next, reward, done, _, _ = self.env.step(action)
                score += reward
                self.local_ac.remember(obs, action, reward)
                t_step += 1
                obs = obs_next

            loss = self.local_ac.loss(done)
            self.optimizer.zero_grad()
            loss.backward()

            for local_param, global_param in zip(
                    self.local_ac.parameters(),
                    self.global_actor_critic.parameters()):
                global_param._grad = local_param.grad
            self.optimizer.step()
            self.local_ac.load_state_dict(
                self.global_actor_critic.state_dict())
            self.local_ac.clear_memory()

            with self.episode_idx.get_lock():
                print(f'{self.episode_idx.value} loss: {loss.item()}')
                wandb.log({"loss": loss.item()}, step=self.episode_idx.value * self.t_max)
                states, actions, rewards = self.local_ac.return_memorized()
                self.episode_idx.value += 1
                self.data_queue.put(
                    {
                        "states": states,
                        "actions": actions,
                        "rewards": rewards
                    }
                )

            torch.cuda.empty_cache()
            gc.collect()