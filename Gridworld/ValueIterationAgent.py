from agent import Agent
import numpy as np


# TASK 2
class ValueIterationAgent(Agent):

    def __init__(self, mdp, discount=0.90, iterations=100):
        """
        Your value iteration agent take an mdp on
        construction, run the indicated number of iterations
        and then act according to the resulting policy.
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations

        states = self.mdp.getStates()
        number_states = len(states)
        start_state = states[0]
        # *************
        #  TODO 2.1 a)
        # self.V = ...
        self.V = {s: 0 for s in states}
        theta = 0.001

        # ************
        for i in range(iterations):
            newV = {}
            delta = 0.0
            for s in states:
                actions = self.mdp.getPossibleActions(s)
                q = {a: 0 for a in actions}
                # **************
                # TODO 2.1. b)
                # if ...
                if len(actions) < 1:
                    newV[s] = 0
                # else: ...
                else:
                    for a in actions:
                        s_dash = self.mdp.getTransitionStatesAndProbs(s, a)
                        for next_state, prob in s_dash:
                            r = self.mdp.getReward(s, a, next_state)
                            q[a] = q[a] + (prob * (r + self.discount * self.V[next_state]))
                    opt_act = max(q, key=q.get)

                    newV[s] = q[opt_act]
                #print(abs(newV[s] - self.V[s]))
                delta = max(delta, abs((newV[s] - self.V[s])))
                self.V.update(newV)

            #print("delta", delta)
            if delta < theta:
                print("Converged after ", i)
                break
                # ***************

    def getValue(self, state):
        """
        Look up the value of the state (after the indicated
        number of value iteration passes).
        """
        # **********
        # TODO 2.2
        return self.V[state]

        # **********

    def getQValue(self, state, action):
        """
        Look up the q-value of the state action pair
        (after the indicated number of value iteration
        passes).  Note that value iteration does not
        necessarily create this quantity and you may have
        to derive it on the fly.
        """
        # ***********
        # TODO 2.3.
        s_dash = self.mdp.getTransitionStatesAndProbs(state, action)
        q = 0
        for next_state, prob in s_dash:
            r = self.mdp.getReward(state, action, next_state)
            q = q + (prob * (r + self.discount * self.V[next_state]))
        return q

        # **********

    def getPolicy(self, state):
        """
        Look up the policy's recommendation for the state
        (after the indicated number of value iteration passes).
        """

        actions = self.mdp.getPossibleActions(state)
        if len(actions) < 1:
            return None

        else:
            q = {a: 0 for a in actions}
            for a in actions:
                s_dash = self.mdp.getTransitionStatesAndProbs(state, a)
                for next_state, prob in s_dash:
                    r = self.mdp.getReward(state, a, next_state)
                    q[a] = q[a] + (prob * (r + self.discount * self.V[next_state]))

            policy = max(q, key=q.get)
            return policy

        # **********
        # TODO 2.4

        # ***********

    def getAction(self, state):
        """
        Return the action recommended by the policy.
        """
        return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        """
        Not used for value iteration agents!
        """

        pass
