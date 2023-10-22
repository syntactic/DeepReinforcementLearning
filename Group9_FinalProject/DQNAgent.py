import numpy as np
import matplotlib.pyplot as plt
import random
import torch
from Agent import Agent
from utils import Buffer

class DQNAgent(Agent):
    def __init__(self, model, action_space:np.ndarray, name:str = "dqn", gamma=0.9, epsilon=1.0, epsilon_decay=0.97, epsilon_floor = 0.1, training=True, training_freq=10, batch_size=4):
        super().__init__(action_space)
        self.model = model
        self.loss_fn = torch.nn.MSELoss()
        self.learning_rate = 1e-3
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_floor = epsilon_floor
        self.loss_bucket = [] # temporarily holds losses for averaging
        self.losses = [] # holds average loss for each game trained on
        self.training = training
        self.training_freq = training_freq
        self.buffer = Buffer()
        self.batch_size = batch_size

        #print(self.model)
    
    def get_action(self, state):

        # get state in the right shape to pass to the model
        s = self.format_state(state)

        # record the current state
        self.state = s

        # get qval from the model
        qval_tensor = self.model(self.state)
        qval = qval_tensor.data.numpy() 

        # choose the next action (random or on policy based on epsilon)
            # currently, the agent can choose impossible actions, which
            # will result in no change to the state
        if (random.random() < self.epsilon):
            action = np.random.choice(self.action_space, size=1)[0]
        else:
            action = self.action_space[np.argmax(qval)]

        # store the action (assuming it is taken)
        self.action = action
        return action
    
    def inform_result(self, next_state, reward, game_over):

        super().inform_result(next_state, reward, game_over)

        # TODO collect information in a buffer to allow for the possibility of 
        # training after x number of steps instead of after every step
        # TODO think about parameters to control whether the agent is training or testing, etc
        if self.training:
            self.buffer.add((np.copy(self.state), self.action, self.reward, np.copy(self.next_state), self.game_over))

    def train(self):
        # TODO test if it is more efficient to save qval_tensor instead of running the model again
        qval_tensor = self.model(self.state)
        
        with torch.no_grad():
            newQ = self.model(self.next_state)
        maxQ = torch.max(newQ) 

        # calculate reward with decay 
        if not self.game_over:                           
            Y = self.reward + (self.gamma * maxQ)
        else:
            Y = self.reward
        
        output = torch.Tensor([Y]).detach()[0]
        input = qval_tensor.squeeze()[self.action]

        loss = self.loss_fn(input, output)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # track loss from training
        self.loss_bucket.append(loss.item())

        if self.game_over:
            self.losses.append(np.mean(self.loss_bucket))

            # epsilon decay
            if self.epsilon > 0.1:
                self.epsilon = self.epsilon * self.epsilon_decay
            else:
                self.epsilon = self.epsilon_floor
            
    def reset(self):
        self.state = float("nan")
        self.next_state = float("nan")
        self.reward = float("nan")
        self.game_over = float("nan")
        self.action = float("nan")
        self.loss_bucket = []

    def plot_losses(self):
        # set up the figure and axes
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 8))

        axes[0].plot(range(len(self.losses)), self.losses)
        axes[0].set_xlabel('games')
        axes[0].set_ylabel('loss')

    def print_results(self, save=False):
        self.plot_losses()
        
        if(save):
            self.save_results()

        plt.show()

    def save_results(self):
        """ saves the trajectory of the agent and the model"""
        pass

