import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


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

        states = self.states[:]
        actions = self.actions[:]
        rewards = self.rewards[:]

        self.states = []
        self.actions = []
        self.rewards = []

        return states, actions, rewards

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
    

class Agent(mp.Process):
    def __init__(self, global_actor_critic, optimizer, input_dims, n_actions, 
                gamma, lr, name, global_ep_idx, env_id):
        super(Agent, self).__init__()
        self.local_actor_critic = A3(input_dims, n_actions, gamma)
        self.global_actor_critic = global_actor_critic
        self.name = 'w%02i' % name
        self.episode_idx = global_ep_idx
        self.env = gym.make(env_id)
        self.optimizer = optimizer

    def run(self):
        t_step = 1
        while self.episode_idx.value < N_GAMES:
            done = False
            observation = self.env.reset()
            score = 0
            self.local_actor_critic.clear_memory()
            while not done:
                action = self.local_actor_critic.choose_action(observation)
                observation_, reward, done, info = self.env.step(action)
                score += reward
                self.local_actor_critic.remember(observation, action, reward)
                if t_step % T_MAX == 0 or done:
                    loss = self.local_actor_critic.calc_loss(done)
                    self.optimizer.zero_grad()
                    loss.backward()
                    for local_param, global_param in zip(
                            self.local_actor_critic.parameters(),
                            self.global_actor_critic.parameters()):
                        global_param._grad = local_param.grad
                    self.optimizer.step()
                    self.local_actor_critic.load_state_dict(
                            self.global_actor_critic.state_dict())
                    self.local_actor_critic.clear_memory()
                t_step += 1
                observation = observation_
            with self.episode_idx.get_lock():
                self.episode_idx.value += 1
            print(self.name, 'episode ', self.episode_idx.value, 'reward %.1f' % score)