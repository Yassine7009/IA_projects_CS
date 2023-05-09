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


from cmath import inf
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
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

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
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # score computed with three objectives : minimize the distance to the closest food, maximize the distance with the ghost, minimize the number of foods

        # minimize the distance to closest food
        food_list = newFood.asList()
        food_list = sorted(food_list, key=lambda x: manhattanDistance(x, newPos))

        if len(food_list) > 0:
            prefered_food_pos = food_list[0]
        else:
            prefered_food_pos = newPos
        score_dist = -manhattanDistance(prefered_food_pos, newPos) * 2

        # max the distance to active ghosts :
        pos_ghost = [
            ghost.getPosition() for ghost in newGhostStates if ghost.scaredTimer == 0
        ]
        pos_ghost = sorted(pos_ghost, key=lambda x: manhattanDistance(x, newPos))
        if len(pos_ghost) > 0:
            worst_pos_ghost = pos_ghost[0]
        else:
            worst_pos_ghost = newPos
        score_ghost = manhattanDistance(worst_pos_ghost, newPos)

        # we can scare the ghosts:
        scare_score = sorted(newScaredTimes)[0] * 40

        # basic case : if the ghost is near, escape...
        if score_ghost < 5:
            score_ghost = score_ghost * 30

        # promote the food eating
        score_food = (len(currentGameState.getFood().asList()) - len(food_list)) * 40

        # calculating the final score
        score = score_dist + score_ghost + score_food + scare_score
        return score


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

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
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
        "*** YOUR CODE HERE ***"
        totalindex = gameState.getNumAgents()

        def main_value(state, thedepth, index):
            if state.isWin() or state.isLose() or thedepth == self.depth:
                return self.evaluationFunction(state)
            elif index == 0:
                value = max_fct_value(state, thedepth)
            else:
                value = min_fct_value(state, index, thedepth)
            return value

        def max_fct_value(state, thedepth):

            maxEval = float("-inf")
            legalMoves = state.getLegalActions(0)
            for action in legalMoves:
                successorGameState = state.generateSuccessor(0, action)
                eval = main_value(successorGameState, thedepth, 1)
                if eval > maxEval:
                    maxEval = eval

            return maxEval

        def min_fct_value(state, index, thedepth):

            minEval = float("inf")
            legalMoves = state.getLegalActions(index)

            if index == totalindex - 1:
                nextdepth = thedepth + 1
                for action in legalMoves:
                    successorGameState = state.generateSuccessor(index, action)
                    eval = main_value(successorGameState, nextdepth, 0)
                    minEval = min(eval, minEval)

                return minEval

            else:

                nextdepth = thedepth
                for action in legalMoves:
                    successorGameState = state.generateSuccessor(index, action)
                    eval = main_value(successorGameState, nextdepth, index + 1)
                    minEval = min(eval, minEval)

                return minEval

        ##NEW VERSION
        maxEval = float(-inf)
        best_action_pac = Directions.STOP

        for action in gameState.getLegalActions(0):
            successorGameState = gameState.generateSuccessor(0, action)
            eval = main_value(successorGameState, 0, 1)
            if eval > maxEval:
                maxEval = eval
                best_action_pac = action
        return best_action_pac


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        totalindex = gameState.getNumAgents()

        def main_value(state, thedepth, index, alpha, beta):
            if state.isWin() or state.isLose() or thedepth == self.depth:
                return self.evaluationFunction(state)
            elif index == 0:
                value = max_fct_value(state, thedepth, alpha, beta)
            else:
                value = min_fct_value(state, index, thedepth, alpha, beta)
            return value

        def max_fct_value(state, thedepth, alpha, beta):

            maxEval = float("-inf")
            legalMoves = state.getLegalActions(0)
            for action in legalMoves:
                successorGameState = state.generateSuccessor(0, action)
                eval = main_value(successorGameState, thedepth, 1, alpha, beta)
                maxEval = max(maxEval, eval)
                if maxEval > beta:
                    return maxEval
                alpha = max(alpha, maxEval)

            return maxEval

        def min_fct_value(state, index, thedepth, alpha, beta):

            minEval = float("inf")
            legalMoves = state.getLegalActions(index)
            if index == totalindex - 1:
                nextdepth = thedepth + 1
                for action in legalMoves:
                    successorGameState = state.generateSuccessor(index, action)
                    eval = main_value(successorGameState, nextdepth, 0, alpha, beta)
                    minEval = min(eval, minEval)
                    if minEval < alpha:
                        break
                    beta = min(minEval, beta)

                return minEval

            else:

                nextdepth = thedepth
                for action in legalMoves:
                    successorGameState = state.generateSuccessor(index, action)
                    eval = main_value(
                        successorGameState, nextdepth, index + 1, alpha, beta
                    )
                    minEval = min(eval, minEval)
                    if minEval < alpha:
                        break
                    beta = min(minEval, beta)

                return minEval

        ##NEW VERSION
        maxEval = float("-inf")
        best_action_pac = Directions.STOP
        alpha = float("-inf")
        beta = float("inf")
        for action in gameState.getLegalActions(0):
            successorGameState = gameState.generateSuccessor(0, action)
            eval = main_value(successorGameState, 0, 1, alpha, beta)
            if eval > maxEval:
                maxEval = eval
                best_action_pac = action
            if maxEval > beta:
                break
            alpha = max(alpha, maxEval)

        return best_action_pac


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
        "*** YOUR CODE HERE ***"
        totalindex = gameState.getNumAgents()

        def main_value(state, thedepth, index):
            if state.isWin() or state.isLose() or thedepth == self.depth:
                return self.evaluationFunction(state)
            elif index == 0:
                value = max_fct_value(state, thedepth)
            else:
                value = min_fct_value(state, index, thedepth)
            return value

        def max_fct_value(state, thedepth):

            maxEval = float("-inf")
            legalMoves = state.getLegalActions(0)
            for action in legalMoves:
                successorGameState = state.generateSuccessor(0, action)
                eval = main_value(successorGameState, thedepth, 1)
                if eval > maxEval:
                    maxEval = eval

            return maxEval

        def min_fct_value(state, index, thedepth):

            legalMoves = state.getLegalActions(index)
            eval = 0
            if index == totalindex - 1:
                nextdepth = thedepth + 1

                for action in legalMoves:
                    successorGameState = state.generateSuccessor(index, action)
                    eval = eval + main_value(successorGameState, nextdepth, 0)

            else:
                nextdepth = thedepth
                for action in legalMoves:
                    successorGameState = state.generateSuccessor(index, action)
                    eval = eval + main_value(successorGameState, nextdepth, index + 1)

            return eval / len(legalMoves)

        ##NEW VERSION
        maxEval = float(-inf)
        best_action_pac = Directions.STOP

        for action in gameState.getLegalActions(0):
            successorGameState = gameState.generateSuccessor(0, action)
            eval = main_value(successorGameState, 0, 1)
            if eval > maxEval:
                maxEval = eval
                best_action_pac = action
        return best_action_pac


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    Pos = currentGameState.getPacmanPosition()
    Food = currentGameState.getFood()
    GhostStates = currentGameState.getGhostStates()
    ScaredTimes = [ghost.scaredTimer for ghost in GhostStates]

    # food :
    food_list = [manhattanDistance(x, Pos) for x in Food]
    if len(food_list) > 0:
        food_score = -min(food_list)  # there is a minus to minimize it
    else:
        food_score = 0

    food_num = -len(Food.asList())

    ghost_list = [
        manhattanDistance(Pos, ghostState.getPosition()) for ghostState in GhostStates
    ]
    if len(ghost_list) > 0:
        ghost_score = min(ghost_list)

    scare_score = sum(ScaredTimes)

    return (
        currentGameState.getScore()
        + scare_score
        + float(ghost_score) / float(food_score)
        + food_num
    )


# Abbreviation
better = betterEvaluationFunction
