import numpy as np
import matplotlib.pyplot as plt
from Agent import Agent
import pygame

class HumanAgent(Agent):
    def __init__(self, name:str, action_space:np.ndarray):
        super().__init__(action_space, name)
    
    def get_action(self, state):
        # waiting for a response
        while(True):
            action = None
            for event in pygame.event.get(): 
                # if the event is a keypress
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        action = self.action_space[0]
                    elif event.key == pygame.K_RIGHT:
                        action = self.action_space[1]
                    elif event.key == pygame.K_DOWN:
                        action = self.action_space[2]
                    elif event.key == pygame.K_LEFT:
                        action = self.action_space[3]
                self.action = action
                if self.action is not None:
                    return self.action
    
    def reset(self):
        pass
