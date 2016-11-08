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

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
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

    def evaluationFunction(self, currentGameState, action):
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
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        return util.undefined()
def scoreEvaluationFunction(currentGameState):
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
    def getAction(self, gameState):
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
        """
        "*** YOUR CODE HERE ***"

        def max_value(state, currentDepth):
            """ Max value for the state at the current depth """

            currentDepth = currentDepth + 1
            if state.isWin() or state.isLose() or currentDepth == self.depth:
                return self.evaluationFunction(state)
            v = float('-Inf')
            moves = state.getLegalActions(0)
            return reduce(lambda a, d: max(a, min_value(state.generateSuccessor(0, d), currentDepth, 1)), moves, v)

        def min_value(state, currentDepth, ghostNum):
            """ Min value for the state at the current depth """
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            v = float('Inf')
            moves = state.getLegalActions(ghostNum)
            def set_min_value(current_min_value, move):
                if ghostNum == gameState.getNumAgents() - 1:
                    return min(current_min_value, max_value(state.generateSuccessor(ghostNum, move), currentDepth))
                else:
                    return min(current_min_value, min_value(state.generateSuccessor(ghostNum, move), currentDepth, ghostNum + 1))

            return reduce(lambda a, d: set_min_value(a, d), moves, v)

        pacmanActions = gameState.getLegalActions(0)
        maximum = float('-Inf')
        maxAction = None
        for action in pacmanActions:
            currentDepth = 0
            currentMax = min_value(gameState.generateSuccessor(0, action), currentDepth, 1)
            if currentMax > maximum:
                maximum = currentMax
                maxAction = action
        return maxAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def max_prune(self, gameState, depth, agentIndex, alpha, beta):
        maxEval= float("-inf")
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        moves = gameState.getLegalActions(0)
        for action in moves:
            successor = gameState.generateSuccessor(0, action)
            tempEval = self.min_prune(successor, depth, 1, alpha, beta)

            if tempEval > beta:
                return tempEval
            if tempEval > maxEval:
                maxEval = tempEval
                maxAction = action

            alpha = max(alpha, maxEval)

        if depth == 1:
            return maxAction
        else:
            return maxEval

    def min_prune(self, gameState, depth, agentIndex, alpha, beta):
        minEval= float("inf")
        numAgents = gameState.getNumAgents()
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        moves = gameState.getLegalActions(agentIndex)
        for action in moves:
            successor = gameState.generateSuccessor(agentIndex, action)
            if agentIndex == numAgents - 1:
              if depth == self.depth: tempEval = self.evaluationFunction(successor)
              else: tempEval = self.max_prune(successor, depth+1, 0, alpha, beta)
            else: tempEval = self.min_prune(successor, depth, agentIndex+1, alpha, beta)
            if tempEval < alpha: return tempEval
            if tempEval < minEval:
              minEval = tempEval
              minAction = action
            beta = min(beta, minEval)
        return minEval

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        maxAction = self.max_prune(gameState, 1, 0, float("-inf"), float("inf"))
        return maxAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        def max_value(state, currentDepth):
            currentDepth = currentDepth + 1
            if state.isWin() or state.isLose() or currentDepth == self.depth: return self.evaluationFunction(state)
            return reduce(lambda a, d: max(a, exp_value(state.generateSuccessor(0, d), currentDepth, 1)), state.getLegalActions(0), float('-Inf'))

        def exp_value(state, currentDepth, ghostNum):
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            def set_exp_value(current_value, move, ghostNum):
                if ghostNum == gameState.getNumAgents() - 1:
                    return current_value + (max_value(state.generateSuccessor(ghostNum, move), currentDepth))/len(state.getLegalActions(ghostNum))
                else:
                    return current_value + (exp_value(state.generateSuccessor(ghostNum, move), currentDepth, ghostNum + 1))/len(state.getLegalActions(ghostNum))
            return reduce(lambda a,d: set_exp_value(a, d, ghostNum), state.getLegalActions(ghostNum), 0)

        pacmanActions = gameState.getLegalActions(0)
        maximum = float('-Inf')
        maxAction = ''
        for action in pacmanActions:
            currentDepth = 0
            currentMax = exp_value(gameState.generateSuccessor(0, action), currentDepth, 1)
            if currentMax > maximum or (currentMax == maximum and random.random() > .3):
                maximum = currentMax
                maxAction = action
        return maxAction

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isWin():  return float("inf")
    if currentGameState.isLose(): return float("-inf")

    # Fetch several data we require to analyze thecurrent state of the pacman's environment
    pacmanPos = currentGameState.getPacmanPosition()
    ghostStates = currentGameState.getGhostStates()
    foodPos = currentGameState.getFood()
    capsules = currentGameState.getCapsules()

    def manhattan(xy1, xy2):
        """
        Our lil' and old Manhattan taxi drive distance function
        """
        return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

    manhattans_food = [ ( manhattan(pacmanPos, food) ) for food in foodPos.asList() ]

    min_manhattans_food = min(manhattans_food)

    manhattans_ghost = [ manhattan(pacmanPos, ghostState.getPosition()) for ghostState in ghostStates if ghostState.scaredTimer == 0 ]
    min_manhattans_ghost = -3
    if ( len(manhattans_ghost) > 0 ):
        min_manhattans_ghost = min(manhattans_ghost)

    manhattans_ghost_scared = [ manhattan(pacmanPos, ghostState.getPosition()) for ghostState in ghostStates if ghostState.scaredTimer > 0 ]
    min_manhattans_ghost_scared = 0;
    if ( len(manhattans_ghost_scared) > 0 ):
        min_manhattans_ghost_scared = min(manhattans_ghost_scared)
    score = scoreEvaluationFunction(currentGameState)

    score += -1.5 * min_manhattans_food
    score += -2 * ( 1.0 / min_manhattans_ghost )
    score += -2 * min_manhattans_ghost_scared
    score += -20 * len(capsules)
    score += -4 * len(foodPos.asList())

    return score

# Abbreviation
better = betterEvaluationFunction
