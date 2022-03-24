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
        newGhostStates = successorGameState.getGhostStates()
        "*** YOUR CODE HERE ***"
        import math
        if action == 'Stop':
            return -math.inf
        newFood = successorGameState.getFood().asList()     # list that contains all food positions
        score = 0   # the value that the function will return
        if len(successorGameState.getFood().asList()) < len(currentGameState.getFood().asList()):   # if pacman ate a food by doing this move
            score += 2000       # increase the score by 2000 because eating a food is a good move
        for food in newFood:    # for every remaining food
            foodDist = manhattanDistance(newPos, food)  # find disttance between the new position and the food
            if foodDist == 0:    # never happens. Just for safety reasons, so as not to divide by zero
                continue
            score += 1000/foodDist  # the closer we are from the food, the bigger the number is
            # we want the new move to lead us closer to foods, so the total score is increased depending on how close we are to every food

        for ghostState in newGhostStates:   # for every ghost
            ghostPos = ghostState.getPosition()  # get the position of the goast
            if newPos == ghostPos:      # if pacman is at the same position with a ghost
                if ghostState.scaredTimer == 0:  # we are in the same position with the goast and cannot eat it, so we die
                    return -math.inf        # return -infinity
                else:
                    score += 2000   # ate a goast, so we get extra points
        return score

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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        import math

        def maxValue(gameState, PacmanId, depth):
            legalMoves = gameState.getLegalActions(PacmanId)    # get all legal moves for max player (pacman)
            v = -(math.inf)     # initialize value
            bestMove = None     # initialize best move
            for move in legalMoves:     # for every legal move
                newState = gameState.generateSuccessor(PacmanId, move)  # generate the new game state after this move
                moveValue = minimax(newState, PacmanId+1, depth)        # call minimax to continue the algorithm with the next agent with the new game state and same depth
                if moveValue[0] > v:        # after minimax return a value, keep only the bigger value and the best move
                    v = moveValue[0]
                    bestMove = move
            return (v, bestMove)    # after minimax has retured for every legal move return the best value and the best move

        def minValue(gameState, id, depth):
            legalMoves = gameState.getLegalActions(id)  # get all legal moves for min player (ghosts)
            v = math.inf    # initialize value
            bestMove = None  # initialize best move
            for move in legalMoves:  # for every legal move
                newState = gameState.generateSuccessor(id, move)    # generate the new game state after this move
                moveValue = minimax(newState, id+1, depth)      # call minimax to continue with the next min agent or the max (if we are at the last min agent).
                # Depth is increased inside minimax, when needed
                if moveValue[0] < v:    # after minimax return a value, keep only the lower value and the best move
                    v = moveValue[0]
                    bestMove = move
            return (v, bestMove)    # after minimax has retured for every legal move return the best value and the best move

        def minimax(gameState, agentId, depth):
            if agentId >= gameState.getNumAgents():  # if minimax was called from the last ghost agent, increase the depth and assign pacman as the next agent to "play"
                depth += 1
                agentId = 0   # pacman id

            if(gameState.isWin() or gameState.isLose() or (depth == self.depth)):   # check if minimax should stop
                return (self.evaluationFunction(gameState), None)
            if agentId == 0:    # if pacman, call maxValue (MAX player)
                return maxValue(gameState, agentId, depth)
            else:   # if ghost agent, call minValue (MIN player)
                return minValue(gameState, agentId, depth)

        return minimax(gameState, 0, 0)[1]  # getAction function calls minimax with starting values for depth 0 and pacman to play first (id=0), and returns the move that minimax returned



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        import math
        # same as minimax algorithm in getAction function with minor differences
        # only the changes will be commented here, all the other are the same as above

        def maxValue(gameState, PacmanId, depth, a, b):  # a, b have been added to the arguments so as to acheive the pruning
            legalMoves = gameState.getLegalActions(PacmanId)
            v = -(math.inf)
            bestMove = None
            for move in legalMoves:
                newState = gameState.generateSuccessor(PacmanId, move)
                moveValue = minimax(newState, PacmanId+1, depth, a, b)  # call minimax with a,b as extra arguments
                if moveValue[0] > v:
                    v = moveValue[0]
                    bestMove = move
                if v > b:       # if the value that minimax returned is bigger than b, we stop and return the value and the move
                    return (v, bestMove)
                else:       # else we update a
                    a = max(a, v)
            return (v, bestMove)

        def minValue(gameState, id, depth, a, b):   # a, b have been added to the arguments so as to acheive the pruning
            legalMoves = gameState.getLegalActions(id)
            v = math.inf
            bestMove = None
            for move in legalMoves:
                newState = gameState.generateSuccessor(id, move)
                moveValue = minimax(newState, id+1, depth, a, b)    # call minimax with a,b as extra arguments
                if moveValue[0] < v:
                    v = moveValue[0]
                    bestMove = move
                if v < a:       # if the value that minimax returned is smaller than a, we stop and return the value and the move
                    return (v, bestMove)
                else:        # else we update b
                    b = min(b, v)

            return (v, bestMove)

        def minimax(gameState, agentId, depth, a, b):   # same, with extra arguments a,b so as to call maxValue and minValue. a and b are not used here
            if agentId >= gameState.getNumAgents():
                depth += 1
                agentId = 0

            if(gameState.isWin() or gameState.isLose() or (depth == self.depth)):
                return (self.evaluationFunction(gameState), None)
            if agentId == 0:
                return maxValue(gameState, agentId, depth, a, b)
            else:
                return minValue(gameState, agentId, depth, a, b)

        return minimax(gameState, 0, 0, -math.inf, math.inf)[1]  # minimax is called with the same starting values as before and a=-inf anf b=inf
        # get getAction returns the move that minimax returned


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
        import math

        def maxValue(gameState, PacmanId, depth):   # same as above
            legalMoves = gameState.getLegalActions(PacmanId)
            v = -(math.inf)
            bestMove = None
            for move in legalMoves:
                newState = gameState.generateSuccessor(PacmanId, move)
                moveValue = minimax(newState, PacmanId+1, depth)
                if moveValue[0] > v:
                    v = moveValue[0]
                    bestMove = move

            return (v, bestMove)

        def minValue(gameState, id, depth):
            legalMoves = gameState.getLegalActions(id)
            propability = 1.0/len(legalMoves)   # each move has same propability to happen
            v = 0
            for move in legalMoves:
                newState = gameState.generateSuccessor(id, move)
                moveValue = minimax(newState, id+1, depth)
                v += moveValue[0]*propability   # we sum the values that minimax returned for every move multiplied with the propability

            return (v, None)    # we return the total value, and no move because all have the same propability so we don't know which will actually happen

        def minimax(gameState, agentId, depth):     # same as above
            if agentId >= gameState.getNumAgents():
                depth += 1
                agentId = 0

            if(gameState.isWin() or gameState.isLose() or (depth == self.depth)):
                return (self.evaluationFunction(gameState), None)
            if agentId == 0:
                return maxValue(gameState, agentId, depth)
            else:
                return minValue(gameState, agentId, depth)

        return minimax(gameState, 0, 0)[1]


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    # *** Better expelantion in the documentation!!

    import math
    if currentGameState.isWin():    # check for winning state
        return currentGameState.getScore()*10000
    if currentGameState.isLose():   # check for losing state
        return -math.inf

    currentPosition = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    capsulesList = currentGameState.getCapsules()
    ghostStates = currentGameState.getGhostStates()

    foodDist = 0
    for food in foodList:   # sum up the distances between pacman and every food
        distance = manhattanDistance(food, currentPosition)
        foodDist += distance

    capsuleDist = 0
    for capsule in capsulesList:     # sum up the distances between pacman and every capsule
        distance = manhattanDistance(food, currentPosition)
        capsuleDist += distance

    ghostDanger = 0
    for ghost in ghostStates:   # for every ghost
        ghostDist = manhattanDistance(ghost.getPosition(), currentPosition)
        if ghost.scaredTimer > 0:   # if pacman can eat it, add to the score
            ghostDanger += 10/ghostDist
        else:       # if pacman cannot eat it, substract from the score
            ghostDanger -= 100/ghostDist

    return currentGameState.getScore()-1*foodDist-2*capsuleDist+ghostDanger  # also get the current score of the game, and use all the above to compose the total score


# Abbreviation
better = betterEvaluationFunction
