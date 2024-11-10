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
 
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        score = successorGameState.getScore()
        
        # We want to reward packman for moving towards uneaten food!
        if newFood:
            minFoodDist = min([util.manhattanDistance(newPos, food) for food in newFood])
            score += 10.0 / minFoodDist 

        # We also need to reward the ghost for eating food
        foodLeft = len(newFood)
        score -= 50 * foodLeft  
        
        # Ghosts are a little more complicated. If they are scared we move towards them, if they aren't we move away.
        for ghostState, scaredTime in zip(newGhostStates, newScaredTimes):
            ghostPos = ghostState.getPosition()
            ghostDist = util.manhattanDistance(newPos, ghostPos)
            
            if scaredTime == 0:
                if ghostDist > 0:
                    score -= 15.0 / ghostDist 
            # Approach scared ghosts
            else:
                score += 5.0 / ghostDist
        
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

        def maxAgent(depth, gameState):
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            
            legalMoves = gameState.getLegalActions(0)
            max_val = float('-inf')
            
            for action in legalMoves:
                successorState = gameState.generateSuccessor(0, action)
                score = minAgent(depth, successorState, 1)
                max_val = max(max_val, score)

            return max_val

        def minAgent(depth, gameState, ghostIndex):
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            legalMoves = gameState.getLegalActions(ghostIndex)
            min_val = float('inf')
            
            for action in legalMoves:
                successorState = gameState.generateSuccessor(ghostIndex, action)
                
                if ghostIndex < gameState.getNumAgents() - 1:
                    score = minAgent(depth, successorState, ghostIndex + 1)
                else:
                    score = maxAgent(depth + 1, successorState)
                
                min_val = min(min_val, score)

            return min_val

        best_score = float('-inf')
        best_action = None
        legalMoves = gameState.getLegalActions(0)

        for action in legalMoves:
            score = minAgent(0, gameState.generateSuccessor(0, action), 1)
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
       
        def maxAgent(depth, gameState, alpha, beta):
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            
            legalMoves = gameState.getLegalActions(0)
            max_val = float('-inf')
            
            for action in legalMoves:
                successorState = gameState.generateSuccessor(0, action)
                score = minAgent(depth, successorState, 1, alpha, beta)
                max_val = max(max_val, score)
                if max_val > beta:
                    return max_val

                alpha = max(max_val, alpha)

            return max_val

        def minAgent(depth, gameState, ghostIndex, alpha, beta):
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            legalMoves = gameState.getLegalActions(ghostIndex)
            min_val = float('inf')
            
            for action in legalMoves:
                successorState = gameState.generateSuccessor(ghostIndex, action)
                
                if ghostIndex < gameState.getNumAgents() - 1:
                    score = minAgent(depth, successorState, ghostIndex + 1, alpha, beta)
                else:
                    score = maxAgent(depth + 1, successorState, alpha, beta)
                
                min_val = min(min_val, score)

                if min_val < alpha:
                    return min_val
                
                beta = min(min_val, beta)

            return min_val

        best_score = float('-inf')
        best_action = None
        legalMoves = gameState.getLegalActions(0)


        alpha = float('-inf')
        beta = float('inf')
        for action in legalMoves:
            score = minAgent(0, gameState.generateSuccessor(0, action), 1, alpha, beta)
            if score > best_score:
                best_score = score
                best_action = action

            if best_score > beta:
                    return best_action

            alpha = max(best_score, alpha)

            

        return best_action

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
        
        def maxAgent(depth, gameState):
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            
            legalMoves = gameState.getLegalActions(0)
            max_val = float('-inf')
            
            for action in legalMoves:
                successorState = gameState.generateSuccessor(0, action)
                score = randomAgent(depth, successorState, 1)
                max_val = max(max_val, score)

            return max_val

        def randomAgent(depth, gameState, ghostIndex):
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            legalMoves = gameState.getLegalActions(ghostIndex)
            num_moves = len(legalMoves)
            random_val = 0
            
            for action in legalMoves:
                successorState = gameState.generateSuccessor(ghostIndex, action)
                
                if ghostIndex < gameState.getNumAgents() - 1:
                    score = randomAgent(depth, successorState, ghostIndex + 1)
                else:
                    score = maxAgent(depth + 1, successorState)
                
                random_val += (1.0 / num_moves) * score

            return random_val

        best_score = float('-inf')
        best_action = None
        legalMoves = gameState.getLegalActions(0)

        for action in legalMoves:
            score = randomAgent(0, gameState.generateSuccessor(0, action), 1)
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    Pos = currentGameState.getPacmanPosition()
    Food = currentGameState.getFood().asList()
    GhostStates = currentGameState.getGhostStates()
    ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]

    score = currentGameState.getScore()
    
    # We want to reward packman for moving towards uneaten food!
    if Food:
        minFoodDist = min([util.manhattanDistance(Pos, food) for food in Food])
        score += 10.0 / minFoodDist 

    # We also need to reward the ghost for eating food
    foodLeft = len(Food)
    score -= 50 * foodLeft  
    
    # Ghosts are a little more complicated. If they are scared we move towards them, if they aren't we move away.
    for ghostState, scaredTime in zip(GhostStates, ScaredTimes):
        ghostPos = ghostState.getPosition()
        ghostDist = util.manhattanDistance(Pos, ghostPos)
        
        if scaredTime == 0:
            if ghostDist > 0:
                score -= 15.0 / ghostDist 
        # Approach scared ghosts
        else:
            score += 5.0 / ghostDist
    
    return score

# Abbreviation
better = betterEvaluationFunction
