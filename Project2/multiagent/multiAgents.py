# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        print(successorGameState)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # Heuristic: reward getting closer to food, avoid nearby non-scared ghosts,
        # small penalty for stopping. We add the heuristic to the successor state's score
        # so that the final returned value (successorGameState.getScore()) reflects it.
        foodList = newFood.asList()
        if len(foodList) > 0:
            minFoodDist = min([manhattanDistance(newPos, f) for f in foodList])
        else:
            minFoodDist = 0

        # Ghost distances (ignore ghosts with unknown position)
        ghostPositions = [g.getPosition() for g in newGhostStates if g.getPosition() is not None]
        if len(ghostPositions) > 0:
            ghostDistances = [manhattanDistance(newPos, gp) for gp in ghostPositions]
            minGhostDist = min(ghostDistances)
        else:
            minGhostDist = float('inf')

        heuristic = 0.0
        # closer food is better
        if minFoodDist > 0:
            heuristic += 10.0 / (minFoodDist)
        # avoid stopping
        if action == Directions.STOP:
            heuristic -= 2.0
        # strongly avoid close non-scared ghosts
        for g in newGhostStates:
            pos = g.getPosition()
            if pos is None:
                continue
            dist = manhattanDistance(newPos, pos)
            if g.scaredTimer == 0 and dist <= 1:
                heuristic -= 200.0
        # small bonus when ghosts are scared and nearby (can eat)
        if any([t > 0 for t in newScaredTimes]):
            if minGhostDist < 3:
                heuristic += 50.0 / (minGhostDist + 1)

        # apply heuristic to successor state's score so return value changes accordingly
        try:
            successorGameState.data.score += heuristic
        except Exception:
            # fall back: do nothing if structure unexpected
            pass

        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        # Multi-agent Minimax
        numAgents = gameState.getNumAgents()

        def minimax(agentIndex, depth, state):
            # terminal or depth cutoff
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            # Pacman (maximizer)
            if agentIndex == 0:
                bestVal = float('-inf')
                for action in state.getLegalActions(0):
                    if action is None:
                        continue
                    succ = state.generateSuccessor(0, action)
                    val = minimax(1, depth, succ)
                    if val > bestVal:
                        bestVal = val
                return bestVal
            else:
                # Ghosts (minimizers)
                bestVal = float('inf')
                nextAgent = agentIndex + 1
                nextDepth = depth
                if agentIndex == numAgents - 1:
                    nextAgent = 0
                    nextDepth = depth + 1

                for action in state.getLegalActions(agentIndex):
                    if action is None:
                        continue
                    succ = state.generateSuccessor(agentIndex, action)
                    val = minimax(nextAgent, nextDepth, succ)
                    if val < bestVal:
                        bestVal = val
                return bestVal

        # Choose best action from root (Pacman)
        bestScore = float('-inf')
        bestAction = None
        for action in gameState.getLegalActions(0):
            succ = gameState.generateSuccessor(0, action)
            score = minimax(1, 0, succ)
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        numAgents = gameState.getNumAgents()

        def alphabeta(agentIndex, depth, state, alpha, beta):
            # Terminal test
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            # Pacman (maximizer)
            if agentIndex == 0:
                value = float('-inf')
                for action in state.getLegalActions(0):
                    succ = state.generateSuccessor(0, action)
                    value = max(value, alphabeta(1, depth, succ, alpha, beta))
                    if value > beta:
                        return value
                    alpha = max(alpha, value)
                return value
            else:
                # Ghosts (minimizers)
                value = float('inf')
                nextAgent = agentIndex + 1
                nextDepth = depth
                if agentIndex == numAgents - 1:
                    nextAgent = 0
                    nextDepth = depth + 1

                for action in state.getLegalActions(agentIndex):
                    succ = state.generateSuccessor(agentIndex, action)
                    value = min(value, alphabeta(nextAgent, nextDepth, succ, alpha, beta))
                    if value < alpha:
                        return value
                    beta = min(beta, value)
                return value

        # Root decision
        bestAction = None
        alpha = float('-inf')
        beta = float('inf')
        bestScore = float('-inf')
        for action in gameState.getLegalActions(0):
            succ = gameState.generateSuccessor(0, action)
            score = alphabeta(1, 0, succ, alpha, beta)
            if score > bestScore:
                bestScore = score
                bestAction = action
            if bestScore > beta:
                break
            alpha = max(alpha, bestScore)
        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        numAgents = gameState.getNumAgents()

        def expectimax(agentIndex, depth, state):
            # Terminal test
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            # Pacman (maximizer)
            if agentIndex == 0:
                best = float('-inf')
                for action in state.getLegalActions(0):
                    succ = state.generateSuccessor(0, action)
                    val = expectimax(1, depth, succ)
                    if val > best:
                        best = val
                return best
            else:
                # Ghosts: compute expected value assuming uniform random actions
                nextAgent = agentIndex + 1
                nextDepth = depth
                if agentIndex == numAgents - 1:
                    nextAgent = 0
                    nextDepth = depth + 1

                actions = state.getLegalActions(agentIndex)
                if len(actions) == 0:
                    return self.evaluationFunction(state)
                total = 0.0
                for action in actions:
                    succ = state.generateSuccessor(agentIndex, action)
                    total += expectimax(nextAgent, nextDepth, succ)
                return total / float(len(actions))

        # Root: choose best action for Pacman
        bestAction = None
        bestScore = float('-inf')
        for action in gameState.getLegalActions(0):
            succ = gameState.generateSuccessor(0, action)
            score = expectimax(1, 0, succ)
            if score > bestScore:
                bestScore = score
                bestAction = action
        return bestAction

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    # Features to consider: current score, distance to nearest food, number of food left,
    # distance to ghosts (penalize proximity to active ghosts, reward proximity to scared ghosts),
    # capsules left.
    pacPos = currentGameState.getPacmanPosition()
    foodGrid = currentGameState.getFood()
    foodList = foodGrid.asList()
    ghostStates = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()
    score = currentGameState.getScore()

    # If winning state, return very large value
    if currentGameState.isWin():
        return float('inf')
    if currentGameState.isLose():
        return -float('inf')

    # Distance to nearest food
    if len(foodList) > 0:
        minFoodDist = min([manhattanDistance(pacPos, f) for f in foodList])
    else:
        minFoodDist = 0

    # Food count and capsule count
    foodCount = len(foodList)
    capsuleCount = len(capsules)

    # Ghost related features
    ghostDistances = []
    scaredTimes = []
    for g in ghostStates:
        gp = g.getPosition()
        if gp is None:
            # ignore ghosts with no position
            continue
        ghostDistances.append(manhattanDistance(pacPos, gp))
        scaredTimes.append(g.scaredTimer)

    # Start with the game score
    value = float(score)

    # Prefer states with fewer food remaining
    value += -4.0 * foodCount

    # Prefer closer food (inverse distance)
    if minFoodDist > 0:
        value += 10.0 / float(minFoodDist)
    else:
        # if on food, give a modest bonus
        value += 5.0

    # Capsules: prefer fewer capsules remaining (encourage consuming them)
    value += -20.0 * capsuleCount

    # Ghost interactions
    for idx, dist in enumerate(ghostDistances):
        scared = scaredTimes[idx] if idx < len(scaredTimes) else 0
        if scared > 0:
            # if ghost is scared, encourage approaching to eat it
            value += 50.0 / float(dist + 1)
        else:
            # if active ghost is very close, heavy penalty; otherwise mild penalty inverse to distance
            if dist <= 1:
                value -= 200.0
            else:
                value -= 10.0 / float(dist)

    return value

# Abbreviation
better = betterEvaluationFunction
