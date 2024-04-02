from torch import nn as nn
import torch
import torch.nn.functional as F
from featureExtractors import *
from learningAgents import ReinforcementAgent
import game
from pacman import GameState
import random,util,math,sys
import numpy as np
from featureExtractors import *
from torch.distributions import Categorical




"""
Tasks:
    X Create a policy
    X Create a function to convert state to pass it to neural network 
    - Generate an episode
    - Update the weights of neural network
"""

"""
~Setup get action part
~Done policy
~Done direction and int map
Next:
    during a state action record it for an episode
    in observation step
"""

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(4, 5)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        action_scores = self.fc1(x)
        return F.softmax(action_scores, dim=1)




class ReinforceAgent(game.Agent):
    """
    REINFORCE ALGORITHIM:

    Policy: a differential policy pi(a|s, theta)
    While True:
        Generate an episode by taking action guided by the policy
        For each time step in an episode:
            G <- returns from step t
            polic_loss = gamma^t * G * log_action_probs 
            # Optimization step
            theta <- theta + learing_rate * gradient_policy_loss
    """

    _direction_to_int:map={
        Directions.NORTH:0,
        Directions.EAST:1,
        Directions.WEST:2,
        Directions.SOUTH:3,
        Directions.STOP:4
    }
    _int_to_direction:map={
        0:Directions.NORTH,
        1:Directions.EAST,
        2:Directions.WEST,
        3:Directions.SOUTH,
        4:Directions.STOP,
    }

    def __init__(self, actionFn = None, numTraining=100, epsilon=0.5, alpha=0.5, gamma=1,width=0,height=0):
        print("REINFORCE Agent initialized")

        self.policy = Policy()
        self.episode = 0
        self.last_score = 0

    def final(self, state):
        print(f"Episode {self.episode} Ends\n")
        print("Now calling train")
        self.episode+=1
    
        self.train()


    def getLegalAction(self, state:GameState)->list[int]:
        """
        Given state it maps the legal action to a binary list
        indicating which action is missing for the current state
        order = North, East, West, South, Stop
        [Stop, north] -> [1,0,0,0,1] 
        """
        ret = [0.0,0.0,0.0,0.0,0.0]
        legal_actions = state.getLegalPacmanActions()
        for action in legal_actions:
            ret[self._direction_to_int[action]]=1.0
        return ret


    def observationFunction(self, state:GameState)->GameState:
        """
            This is where we ended up after our last action.
            The simulation somehow enusers to call this step
          * This function should return state:gameState
        """

        print("Got an Observation")
        self.observationStep(state)

        return state
    
    def observationStep(self, state:GameState):
        print(f"Previous step reward: {state.getScore()-self.last_score}")
        self.last_score = state.getScore()
        # self.train()
        # x = input()
        

    def train(self):
        print("Train function called")



    def getAction(self, state:GameState):
        legal_action = self.getLegalAction(state)
        
        print("Extracting features")
        exctractFeature = SimpleExtractor()
        features_dict = exctractFeature.getFeatures(state)
        features = [features_dict[key] for key in features_dict]
        features = torch.tensor([features], dtype=torch.float32)
        actions_probs = self.policy(features)
        print("Policy suggested")
        print(actions_probs)

        indices_to_zero_tensor = torch.tensor(legal_action, dtype=torch.float32)
        actions_probs*=indices_to_zero_tensor

        m = Categorical(actions_probs)
        action = m.sample()

        print(self._int_to_direction[action.item()])
        print()

        return self._int_to_direction[action.item()]
        

    
