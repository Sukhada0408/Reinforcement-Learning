import numpy as np
from agent import Agent


# TASK 1

class PolicyIterationAgent(Agent):

    def __init__(self, mdp, discount=0.9, iterations=100):  # default mdp not given
        """
        Your policy iteration agent take an mdp on
        construction, run the indicated number of iterations
        and then act according to the resulting policy.
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations

        states = self.mdp.getStates()
        number_states = len(states)
        print(states[0])
        # Policy initialization
        # ******************
        # TODO 1.1.a)
        # self.V = ...
        self.V = {s: 0.0 for s in states}

        # *******************

        self.pi = {s: self.mdp.getPossibleActions(s)[-1] if self.mdp.getPossibleActions(s) else None for s in states}

        counter = 0
        theta = 0.001

        while True:
            # Policy evaluation
            for i in range(iterations):
                newV = {}
                delta = 0.0
                for s in states:
                    a = self.pi[s]
                    # ***************** 
                    # TODO 1.1.b)
                    # if...
                    if a is None:
                        pass
                    #
                    # else:...
                    else:
                        # actions = self.mdp.getPossibleActions(s)
                        s_dash = self.mdp.getTransitionStatesAndProbs(s, a)
                        r = self.mdp.getReward(s, a, None)
                        for next_state, prob in s_dash:
                            # print ("Reward", r)
                            if s in newV:
                                newV[s] = newV[s] + (
                                        prob * (r + self.discount * self.V[next_state]))  ###Check this part
                            else:
                                newV[s] = (prob * (r + self.discount * self.V[next_state]))
                    if s != (-1, -1):
                        delta = max(delta, abs((newV[s] - self.V[s])))

                # update value estimate
                # self.V=...
                self.V.update(newV)
                if delta < theta:
                    break

                # ******************
            policy_stable = True
            for s in states:
                actions = self.mdp.getPossibleActions(s)
                # q = {a: 0 for a in actions}
                if len(actions) < 1:
                    self.pi[s] = None
                else:
                    old_action = self.pi[s]
                    # ************
                    # TODO 1.1.c)
                    # self.pi[s] = ...
                    q = {a: 0 for a in actions}
                    for a in actions:
                        s_dash = self.mdp.getTransitionStatesAndProbs(s, a)
                        r = self.mdp.getReward(s, a, None)
                        for next_state, prob in s_dash:
                            q[a] = q[a] + (prob * (r + self.discount * self.V[next_state]))
                    self.pi[s] = max(q, key=q.get)
                    # policy_stable = 
                    policy_stable = (old_action == self.pi[s])

            counter += 1
            if policy_stable: break
        print("Policy converged after %i iterations of policy iteration" % counter)

    def getValue(self, state):
        """
        Look up the value of the state (after the policy converged).
        """
        # *******
        # TODO 1.2.
        return self.V[state]
        # ********

    def getQValue(self, state, action):
        """
        Look up the q-value of the state action pair
        (after the indicated number of value iteration
        passes).  Note that policy iteration does not
        necessarily create this quantity and you may have
        to derive it on the fly.
        """
        # *********
        # TODO 1.3.
        s_dash = self.mdp.getTransitionStatesAndProbs(state, action)
        q = 0
        r = self.mdp.getReward(state, action, None)
        for next_state, prob in s_dash:
            q = q + (prob * (r + self.discount * self.V[next_state]))

        return q
        # **********

    def getPolicy(self, state):
        """
        Look up the policy's recommendation for the state
        (after the indicated number of value iteration passes).
        """
        # **********
        # TODO 1.4.
        #         actions = self.mdp.getPossibleActions(state)
        #         if (len(actions)<1):
        #             return(None)
        #         else:
        #             for a in actions:
        #                 q = {a: 0 for a in actions}
        #                 s_dash = self.mdp.getTransitionStatesAndProbs(state, a)
        #                 r = self.mdp.getReward(state, a, None)
        #                 for next_state, prob in s_dash:
        #                     q[a] = q[a]+((prob)*(r + self.discount*self.V[next_state]))
        #             policy = max(q, key=q.get)

        return self.pi[state]

        # **********

    def getAction(self, state):
        """
        Return the action recommended by the policy.
        """
        return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        """
        Not used for policy iteration agents!
        """

        pass
