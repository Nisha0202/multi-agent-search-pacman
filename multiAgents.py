# multiAgents.py

from util import manhattanDistance
from game import Directions
import random, util
from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.
    """

    def getAction(self, gameState):
        """
        Chooses among the best options according to the evaluation function.
        """
        legalMoves = gameState.getLegalActions()

        # Evaluate all legal moves
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Evaluates a state based on food distance, ghost proximity, etc.
        """
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()

        # Calculate distance to the closest food
        foodDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        foodScore = -min(foodDistances) if foodDistances else 0

        # Calculate ghost proximity
        ghostDistances = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        ghostScore = sum([-10 / (dist + 1) for dist in ghostDistances if dist < 3])

        return successorGameState.getScore() + foodScore + ghostScore


def scoreEvaluationFunction(currentGameState):
    """
    Returns the score of the state as displayed in the Pacman GUI.
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    Common elements for all multi-agent searchers.
    """
    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Implements the Minimax algorithm for adversarial search.
    """
    def getAction(self, gameState):
        def minimax(state, depth, agentIndex):
            # Check for terminal state
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            # Pacman's turn (maximizing agent)
            if agentIndex == 0:
                return max(
                    minimax(state.generateSuccessor(agentIndex, action), depth, 1)
                    for action in state.getLegalActions(agentIndex)
                )

            # Ghosts' turn (minimizing agent)
            nextAgent = (agentIndex + 1) % state.getNumAgents()
            nextDepth = depth + 1 if nextAgent == 0 else depth
            return min(
                minimax(state.generateSuccessor(agentIndex, action), nextDepth, nextAgent)
                for action in state.getLegalActions(agentIndex)
            )

        # Return the action with the best minimax score
        legalMoves = gameState.getLegalActions()
        scores = [minimax(gameState.generateSuccessor(0, action), 0, 1) for action in legalMoves]
        bestScore = max(scores)
        bestActions = [action for action, score in zip(legalMoves, scores) if score == bestScore]
        return random.choice(bestActions)

# my code 
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Implementing Minimax with Alpha-Beta Pruning.
    """

    def getAction(self, gameState):

   
        def alphaBeta(state, depth, agentIndex, alpha, beta):
            """
            Implementing the Alpha-Beta pruning algorithm.
          
            """
            # Checkong if the game is over (win/lose) or if the depth limit is reached.
            # In such cases, return the evaluation score of the current state.
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            if agentIndex == 0:  # Maximizing agent (Pacman's turn)
                value = float('-inf')  # Start with the smallest possible value.
                for action in state.getLegalActions(agentIndex):  
                    # Calculating the value of the successor state using Alpha-Beta recursively.
                    value = max(value, alphaBeta(state.generateSuccessor(agentIndex, action), depth, 1, alpha, beta))
                    alpha = max(alpha, value)  # update alpha with the maximum value found so far.
                    if alpha >= beta: 
                        break
                return value

            else:  # Minimizing agent (Ghosts' turn)
                value = float('inf')  # Start with the largest possible value.
                # Determining the next agent and increment depth only if the next agent is Pacman.
                nextAgent = (agentIndex + 1) % state.getNumAgents()
                nextDepth = depth + 1 if nextAgent == 0 else depth
                for action in state.getLegalActions(agentIndex): 
                    # calculating the value of the successor state using Alpha-Beta recursively.
                    value = min(value, alphaBeta(state.generateSuccessor(agentIndex, action), nextDepth, nextAgent, alpha, beta))
                    beta = min(beta, value)  # update beta with the minimum value found so far.
                    if beta <= alpha: 
                        break
                return value

        # Get all legal actions for Pacman at the root of the game tree.
        legalMoves = gameState.getLegalActions()
        bestAction = None  # To store the best action for Pacman.
        alpha, beta = float('-inf'), float('inf')  # Initializing alpha and beta with extreme values.

        for action in legalMoves: 
          
            value = alphaBeta(gameState.generateSuccessor(0, action), 0, 1, alpha, beta)
            if value > alpha:  # If the action has a better value, update alpha and store the action.
                alpha = value
                bestAction = action

        return bestAction  # Return the action with the highest evaluated score.


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Implements the Expectimax algorithm.
    """
    def getAction(self, gameState):
        def expectimax(state, depth, agentIndex):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            if agentIndex == 0:  # Maximizing agent
                return max(
                    expectimax(state.generateSuccessor(agentIndex, action), depth, 1)
                    for action in state.getLegalActions(agentIndex)
                )

            else:  # Chance agent
                nextAgent = (agentIndex + 1) % state.getNumAgents()
                nextDepth = depth + 1 if nextAgent == 0 else depth
                actions = state.getLegalActions(agentIndex)
                probabilities = 1 / len(actions)
                return sum(
                    probabilities * expectimax(state.generateSuccessor(agentIndex, action), nextDepth, nextAgent)
                    for action in actions
                )

        legalMoves = gameState.getLegalActions()
        scores = [expectimax(gameState.generateSuccessor(0, action), 0, 1) for action in legalMoves]
        bestScore = max(scores)
        bestActions = [action for action, score in zip(legalMoves, scores) if score == bestScore]
        return random.choice(bestActions)

# my code
def betterEvaluationFunction(currentGameState):
  
    # If Pacman has no legal actions, it means the game ends (loss), return 0
    if not currentGameState.getLegalActions():
        return 0

    # Extract important state information
    position = currentGameState.getPacmanPosition()  # Current position of Pacman
    food = currentGameState.getFood().asList()      # List of all food positions
    ghostStates = currentGameState.getGhostStates()  # List of ghost states
    capsules = currentGameState.getCapsules()       # List of remaining capsules

    # Food 
    foodDistances = [manhattanDistance(position, f) for f in food]
    closestFoodDistance = min(foodDistances) if food else 0
    # Increased weight to strongly encourage eating the closest food so that it not just keep looking here and there
    foodScore = -2 * closestFoodDistance if food else 0 

    # Ghost 
    ghostScore = 0
    for ghost in ghostStates:
        dist = manhattanDistance(position, ghost.getPosition())  # Distance to the ghost
        if ghost.scaredTimer > 0: 
            ghostScore += 300 / (dist + 1)  # Encourages eating scared ghosts
        elif dist < 4:  
            ghostScore -= 75 / (dist + 1)  # reduced penalty for close ghosts

    # Capsule Penalty
    capsuleScore = -100 * len(capsules)  # Encourages eating capsules but reduce penalty weight

    # Combining scores
    return currentGameState.getScore() + foodScore + ghostScore + capsuleScore


# Abbreviation
better = betterEvaluationFunction

# python pacman.py -p AlphaBetaAgent -l minimaxClassic -a depth=4
# python pacman.py -p AlphaBetaAgent -a depth=3 -l smallClassic 