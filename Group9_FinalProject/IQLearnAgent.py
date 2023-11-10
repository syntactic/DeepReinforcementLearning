import numpy as np
import random
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from Agent import Agent
from utils import Buffer, iq_loss, get_max_Q

class IQLearnAgent(Agent):
    def __init__(self, action_space:np.ndarray, model, name:str = "IQLearn", gamma=0.9, epsilon=1.0, epsilon_decay=0.97, epsilon_floor = 0.1, training=True, training_freq=4, batch_size=8):
        super().__init__(action_space)
        self.name = name
        self.model = model
        self.loss_fn = iq_loss
        self.learning_rate = 1e-3
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_floor = epsilon_floor
        self.training = training
        self.training_freq = training_freq
        self.buffer = Buffer() # policy buffer
        self.expert_buffer = Buffer()
        self.batch_size = batch_size

    def set_expert_buffer(self, expert_buffer):
        self.expert_buffer = expert_buffer

    def get_action(self, state):

        # record the current state
        self.state = state

        # choose the next action (random or on policy based on epsilon)
            # currently, the agent can choose impossible actions, which
            # will result in no change to the state
        if (random.random() < self.epsilon):
            action = np.random.choice(self.action_space, size=1)[0]
        else:
            # get qval from the model
            qval_tensor = self.model.get_Q([self.state]).squeeze()
            qval = qval_tensor.data.numpy() 
            dist = F.softmax(qval_tensor/0.01)
            dist = Categorical(dist)
            action = dist.sample()
            #action = self.action_space[np.argmax(qval)]

        # store the action (assuming it is taken)
        self.action = action
        return action

    def inform_result(self, next_state, reward, game_over):
        super().inform_result(next_state, reward, game_over)

        # TODO collect information in a buffer to allow for the possibility of 
        # training after x number of steps instead of after every step
        # TODO think about parameters to control whether the agent is training or testing, etc

        #if self.training:
            #self.buffer.add((np.copy(self.state), self.action, self.reward, np.copy(self.next_state), self.game_over))
            
        # epsilon decay
        if self.epsilon > 0.1:
            self.epsilon = self.epsilon * self.epsilon_decay
        else:
            self.epsilon = self.epsilon_floor

    def train(self):
        if self.expert_buffer.size() >= self.batch_size:
            state, next_state, action, _, done = self.expert_buffer.get_samples(self.batch_size)
            action = torch.as_tensor(action, dtype=torch.int64, device=self.model.device)
            if action.ndim == 1:
                action = action.unsqueeze(1)
            done = torch.as_tensor(done, dtype=torch.float, device=self.model.device)
            Q_vals = self.model.get_Q(state)
            current_Q = Q_vals.gather(2, action.unsqueeze(1)).squeeze()
            current_V = get_max_Q(Q_vals)
            next_V = get_max_Q(self.model.get_Q(next_state))

            #  calculate 1st term for IQ loss
            #  -E_(ρ_expert)[Q(s, a) - γV(s')]
            y = (1 - done) * self.gamma * next_V
            #reward = (current_Q - y)#[is_expert]


            #
            loss = iq_loss(current_Q, current_V, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # track loss from training
            self.model.append_to_loss_bucket(loss.item())

    def reset(self):
        self.state = float("nan")
        self.next_state = float("nan")
        self.reward = float("nan")
        self.game_over = float("nan")
        self.action = float("nan")

