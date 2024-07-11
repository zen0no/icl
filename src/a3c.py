# code inspired by https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/A3C/pytorch/a3c.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.multiprocessing as mp

import os

# This lines avoids non-determenistic behaviour of LSTM on CUDA
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'


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
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(self, nn.Module).__init__()

        self.h = torch.zeros(1, self.hidden_dim, device=self.device)
        self.c = torch.zeros(1, self.hidden_dim, device=self.device)

        self.encode = nn.Sequential(
                                    nn.Linear(input_dim, hidden_dim),
                                    nn.GELU(hidden_dim, hidden_dim),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.GELU()
                                    )
        self.lstm = nn.LSTMCell(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.encode(x)
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

        self.pi = NetWithMemory(state_dim, hidden_dim, action_dim).to(self.device)
        self.v  = NetWithMemory(state_dim, hidden_dim, 1).to(self.device)

        self.states = []
        self.actions = []
        self.rewards = []

    def remember(self, s, a, r):
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)

    def clear_memory(self):
        #reset hidden state of memory nets
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
        states = torch.tensor(self.states, dtype=torch.float32)
        actions = torch.tensor(self.actions, dtype=torch.float32)

        pi, values = self.forward(states)

        # calculate returns and critic loss
        R = values[-1]*(1-int(done))
        returns = []
        for reward in self.rewards[::-1]:
            R = reward + self.gamma*R
            returns.append(R)
        returns.reverse()
        returns = torch.tensor(returns, dtype=torch.float32)
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
        state = torch.tensor([state], dtype=torch.float32)
        pi, v = self.forward(state)
        probs = torch.softmax(pi, dim=1)
        d = Categorical(probs)
        action = d.sample().numpy()[0]

        return action
    

class Worker(mp.Process):
    def __init__(self, global_ac, optimizer, env,
                gamma, name, global_episode_idx, t_max, max_episode, data_queue):
        super(Worker, self).__init__()
        
        self.env = env
        self.optimizer = optimizer
        
        self.env = env
        state_dim = env.state_dim 
        action_dim = env.action_dim

        self.local_ac = A3C(state_dim, action_dim, gamma)
        self.global_actor_critic = global_ac
        self.name = 'w%02i' % name
        self.episode_idx = global_episode_idx
        self.data_queue = data_queue

        self.max_episode = max_episode
        self.t_max = t_max

    def run(self):
        t_step = 1

        while self.episode_idx.value < self.max_episodes:
            done = False
            obs = self.env.reset()
            score = 0
            self.local_ac.clear_memory()

            while not done:
                action = self.local_ac.act(obs)
                obs_next, reward, done, _ = self.env.step(action)
                score += reward
                self.local_ac.remember(obs, action, reward)

                if t_step % self.t_max == 0 or done:
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
                t_step += 1
                obs = obs_next
            with self.episode_idx.get_lock():

                states, actions, rewards = self.local_ac.return_memorized()

                self.episode_idx.value += 1
                self.data_queue.put(
                    {
                        "states": states,
                        "actions": actions,
                        "rewards": rewards
                    }
                )
            