# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        """
        Runs the batch version of value iteration for self.iterations.
        V_k+1(s) <- max_a sum_{s'} T(s,a,s')[R(s,a,s') + gamma * V_k(s')]
        """
        states = self.mdp.getStates()
        gamma = self.discount

        for i in range(self.iterations):
            # Use the "batch" version: compute new_values using current self.values
            new_values = self.values.copy() 

            for state in states:
                if self.mdp.isTerminal(state):
                    # Terminal states have a value of 0, but since they never change, 
                    # we can simply skip them or ensure their value remains 0.
                    new_values[state] = 0
                    continue

                possible_actions = self.mdp.getPossibleActions(state)
                
                # If there are no legal actions (but it's not a terminal state, which 
                # shouldn't typically happen in a well-defined MDP but is good to guard against), 
                # the max Q-value would be -infinity. For a practical MDP, this loop should run.
                if not possible_actions:
                    new_values[state] = 0 # Or the default value from self.values, which is 0.
                    continue
                
                max_q_value = -float('inf')

                for action in possible_actions:
                    q_value = self.computeQValueFromValues(state, action)
                    max_q_value = max(max_q_value, q_value)

                # Update the value for this state using the max Q-value found
                new_values[state] = max_q_value

            # Update the stored values after all states have been processed for this iteration
            self.values = new_values


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        q_value = 0.0
        gamma = self.discount
        
        # Get (nextState, prob) pairs
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)

        for next_state, prob in transitions:
            if prob > 0:
                reward = self.mdp.getReward(state, action, next_state)
                # V_k(s') is fetched from self.values
                next_state_value = self.getValue(next_state) 
                
                # Q-value component: prob * [Reward + gamma * V_k(s')]
                q_value += prob * (reward + gamma * next_state_value)

        return q_value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        if self.mdp.isTerminal(state):
            return None

        possible_actions = self.mdp.getPossibleActions(state)

        if not possible_actions:
            # Handle the case where there are no legal actions.
            return None

        # To find the best action, we calculate the Q-value for each possible action.
        q_values = util.Counter() # A Counter is used to easily store and find the max
        
        for action in possible_actions:
            q_values[action] = self.computeQValueFromValues(state, action)
            
        # util.argMax returns the key (action) that corresponds to the maximum value.
        best_action = q_values.argMax()
        
        return best_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class PrioritizedSweepingValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        # Helper function to compute the highest Q-value for a state
        def max_q_value(state):
            if self.mdp.isTerminal(state):
                return 0.0
            possible_actions = self.mdp.getPossibleActions(state)
            if not possible_actions:
                return 0.0
            
            # Using computeQValueFromValues, which is inherited/available
            q_values = [self.computeQValueFromValues(state, action) 
                        for action in possible_actions]
            return max(q_values)


        # 1. Compute predecessors of all states.
        # Predecessors[s] = set of states p that can transition to s.
        predecessors = collections.defaultdict(set)
        for state in self.mdp.getStates():
            for action in self.mdp.getPossibleActions(state):
                for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                    if prob > 0:
                        predecessors[next_state].add(state)
        
        # 2. Initialize an empty priority queue.
        pq = util.PriorityQueue()
        
        # 3. For each non-terminal state s, initialize the priority queue.
        for state in self.mdp.getStates(): # Iterate in order for autograder
            if self.mdp.isTerminal(state):
                continue
            
            # Find the absolute difference (diff = |V(s) - max_a Q(s,a)|)
            highest_q = max_q_value(state)
            diff = abs(self.values[state] - highest_q)
            
            # Push s into the priority queue with priority -diff (prioritizes largest errors)
            pq.push(state, -diff)

        # 4. Main Iteration Loop
        for i in range(self.iterations):
            # If the priority queue is empty, terminate.
            if pq.isEmpty():
                break
                
            # Pop a state s off the priority queue.
            state = pq.pop()
            
            # Update the value of s in self.values.
            # (Guaranteed non-terminal by initialization and update logic)
            self.values[state] = max_q_value(state)
            
            # For each predecessor p of s, do:
            for p in predecessors[state]:
                if self.mdp.isTerminal(p):
                    continue
                    
                # Find the absolute difference (diff = |V(p) - max_a Q(p,a)|)
                highest_q_p = max_q_value(p)
                diff = abs(self.values[p] - highest_q_p)
                
                # If diff > theta, push p into the priority queue with priority -diff.
                if diff > self.theta:
                    # util.PriorityQueue's update method handles inserting or updating priority
                    pq.update(p, -diff)
        

