from model.agent import Agent
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

"""
The Actor calss uses a GRU cell that produces the state before it is fed into the fully-connected layers.
There are 2 fully connected layers defined by fc_policy_1 and fc_policy_2 as well as 2 different output layers 
that are also linear layers defined by fc_policy_out and fc_cash_out. 

To read how input data is fed into the network, we look at the forward function.
State is produced as an ouput from the gru cell.
This state is then passed through 2 fully-connected layers (fc_policy_1 and fc_policy2) with a ReLU activation function.
Subsequently, state is passed to both fc_cash_out and fc_policy_out with sigmoid activations to produce
two different outputs; cash and action respectively.

After taking the cash average, the action is produced from the last two lines of the forward function.
This resulting action is what is being called in both trade and train. During trading, 
no weight update is being donce as torch.no_grad() prevents the network weights from being updated, thus, only during 
training are the weights being updated.
"""

class Actor(nn.Module):
    def __init__(self, s_dim, b_dim, rnn_layers=1, dp=0.2):
        super(Actor, self).__init__()
        self.s_dim = s_dim
        self.b_dim = b_dim
        self.rnn_layers = rnn_layers
        self.gru = nn.GRU(self.s_dim, 128, self.rnn_layers, batch_first=True)
        self.fc_policy_1 = nn.Linear(128, 128)
        self.fc_policy_2 = nn.Linear(128, 64)
        self.fc_policy_out = nn.Linear(64, 1)
        self.fc_cash_out = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=dp)
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.initial_hidden = torch.zeros(self.rnn_layers, self.b_dim, 128, dtype=torch.float32)

    def forward(self, state, hidden=None, train=False):
        state, h = self.gru(state, hidden)
        if train:
            state = self.dropout(state)
        state = self.relu(self.fc_policy_1(state))
        state = self.relu(self.fc_policy_2(state))
        cash = self.sigmoid(self.fc_cash_out(state))
        action = self.sigmoid(self.fc_policy_out(state)).squeeze(-1).t()
        cash = cash.mean(dim=0)
        action = torch.cat(((1 - cash) * action, cash), dim=-1)
        action = action / (action.sum(dim=-1, keepdim=True) + 1e-10)
        return action, h.data


class DRLAgent(Agent):
    def __init__(self, s_dim, b_dim, batch_length=64, learning_rate=1e-3, rnn_layers=1):
        super().__init__()
        self.s_dim = s_dim
        self.b_dim = b_dim
        self.batch_length = batch_length
        self.pointer = 0
        self.s_buffer = []
        self.d_buffer = []

        self.train_hidden = None
        self.trade_hidden = None
        self.actor = Actor(s_dim=self.s_dim, b_dim=self.b_dim, rnn_layers=rnn_layers)
        self.optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)

    def _trade(self, state, train=False):
        with torch.no_grad():
            a, self.trade_hidden = self.actor(state[:, None, :], self.trade_hidden, train=False)
        return a

    def trade(self, state, train=False):
        state_ = torch.tensor(state, dtype=torch.float32)
        action = self._trade(state_, train=train)
        return action.numpy().flatten()

    def train(self):
        self.optimizer.zero_grad()
#         print(type(self.s_buffer))
#         print(self.s_buffer[1].shape)
#         print(len(self.s_buffer))
#         print(self.s_buffer)
        s = torch.stack(self.s_buffer).permute(1,0,2)
#         print(s.size())
        d = torch.stack(self.d_buffer)
#         print("----------------------------------------------")
#         print(self.d_buffer)
#         print(d.size())
        a_hat, self.train_hidden = self.actor(s, self.train_hidden, train=True)
        reward = -(a_hat[:, :-1] * d).mean()
        reward.backward()
        for param in self.actor.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def reset_model(self):
        self.s_buffer = []
        self.d_buffer = []
        self.trade_hidden = None
        self.train_hidden = None
        self.pointer = 0

    def save_transition(self, state, diff):
        if self.pointer < self.batch_length:
            self.s_buffer.append(torch.tensor(state, dtype=torch.float32))
            self.d_buffer.append(torch.tensor(diff, dtype=torch.float32))
            self.pointer += 1
        else:
            self.s_buffer.pop(0)
            self.d_buffer.pop(0)
            self.s_buffer.append(torch.tensor(state, dtype=torch.float32))
            self.d_buffer.append(torch.tensor(diff, dtype=torch.float32))

    def load_model(self, model_path='./DRL_Torch'):
        self.actor = torch.load(model_path + '/model.pkl')

    def save_model(self, model_path='./DRL_Torch'):
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        torch.save(self.actor, model_path + '/model.pkl')

