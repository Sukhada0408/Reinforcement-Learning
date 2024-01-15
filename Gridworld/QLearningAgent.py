import numpy as np

import util
from agent import Agent


# TASK 3

class QLearningAgent(Agent):

    def __init__(self, actionFunction, discount=0.9, learningRate=0.1, epsilon=0.3):
        """ A Q-Learning agent gets nothing about the mdp on construction other than a function mapping states to
        actions. The other parameters govern its exploration strategy and learning rate. """
        self.setLearningRate(learningRate)
        self.setEpsilon(epsilon)
        self.setDiscount(discount)
        self.actionFunction = actionFunction

        self.qInitValue = 0  # initial value for states
        self.Q = {}

    def setLearningRate(self, learningRate):
        self.learningRate = learningRate

    def setEpsilon(self, epsilon):
        self.epsilon = epsilon

    def setDiscount(self, discount):
        self.discount = discount

    def getValue(self, state):
        """ Look up the current value of the state. """
        # *********
        # TODO 3.1.
        if state in self.Q.keys():
            return max(self.Q[state].values())
        else:
            return 0
        # *********

    def getQValue(self, state, action):
        """ Look up the current q-value of the state action pair. """
        # *********
        # TODO 3.2.
        if state in self.Q.keys() and action in self.Q[state].keys():
            return self.Q[state][action]
        else:
            return 0
        # *********

    def getPolicy(self, state):
        """ Look up the current recommendation for the state. """
        # *********
        # TODO 3.3.
        all_actions = self.actionFunction(state)
        if state in self.Q.keys():
            if len(all_actions) > 0:
                action = all_actions[0]
                if state in self.Q.keys() and action in self.Q[state].keys():
                    max_val = self.Q[state][action]
                    for act, val in self.Q[state].items():
                        if val > max_val:
                            action = act
                    return action
            else:
                return "exit"
        else:
            return self.getRandomAction(state)
        # *********

    def getRandomAction(self, state):
        all_actions = self.actionFunction(state)
        if len(all_actions) > 0:
            # *********
            return np.random.choice(all_actions)
            # *********
        else:
            return "exit"

    def getAction(self, state):
        """ Choose an action: this will require that your agent balance exploration and exploitation as appropriate. """
        # *********
        # TODO 3.4.
        if np.random.rand() < self.epsilon:
            return self.getRandomAction(state)
        else:
            return self.getPolicy(state)
        # *********

    def update(self, state, action, nextState, reward):
        """ Update parameters in response to the observed transition. """
        # *********
        # TODO 3.5.
        all_actions = self.actionFunction(state)

        if state in self.Q.keys():
            if nextState == (-1, -1):
                self.Q[state][action] += (self.learningRate * (reward - self.Q[state][action]))

            elif nextState in self.Q.keys():
                self.Q[state][action] += self.learningRate * (
                        reward + self.discount * self.getValue(nextState) - self.Q[state][action])

            else:
                self.Q[nextState] = {}
                for a in self.actionFunction(nextState):
                    self.Q[nextState][a] = 0.0
                self.Q[state][action] += (self.learningRate * (reward - self.Q[state][action]))

        else:
            self.Q[state] = {}
            for a in all_actions:
                self.Q[state][a] = 0.0
            if nextState not in self.Q.keys():
                self.Q[nextState] = {}
                for a in self.actionFunction(nextState):
                    self.Q[nextState][a] = 0.0
                self.Q[state][action] += (self.learningRate * (reward - self.Q[state][action]))
        return self.Q
        # *********
