# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

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
    if successorGameState.isWin():
      return float("inf")

    score = successorGameState.getScore()

    ghostDistances = [util.manhattanDistance(newPos, g.getPosition()) for g in newGhostStates]
    maxGhostDistance = max(ghostDistances)
    score += maxGhostDistance

    foodDistances = [util.manhattanDistance(newPos, food) for food in newFood.asList()]
    minFoodDistance = min(foodDistances)
    score -= 3 * minFoodDistance
    if successorGameState.getNumFood() < currentGameState.getNumFood():
      score *= 2
    if action == Directions.STOP:
      score -= 10

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

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"
    def maxNode(gameState, depth, numGhosts):
      if gameState.isWin() or gameState.isLose() or depth == 0:
        return self.evaluationFunction(gameState)
      actions = gameState.getLegalActions(0)
      actions.remove(Directions.STOP)
      minNodes = [minNode(gameState.generateSuccessor(0, action), 1, depth - 1, numGhosts) for action in actions]
      return max(minNodes)

    def minNode(gameState, agent, depth, numGhosts):
      if gameState.isWin() or gameState.isLose() or depth == 0:
        return self.evaluationFunction(gameState)
      score = None
      actions = gameState.getLegalActions(agent)
      if agent == numGhosts:
        maxNodes = [maxNode(gameState.generateSuccessor(agent, action), depth - 1, numGhosts) for action in actions]
        score = min(maxNodes)
      else:
        minNodes = [minNode(gameState.generateSuccessor(agent, action), agent + 1, depth, numGhosts) for action in actions]
        score = min(minNodes)
      return score

    numGhosts = gameState.getNumAgents() - 1
    actions = gameState.getLegalActions(0)
    actions.remove(Directions.STOP)
    resAction = Directions.STOP
    score = -float("inf")
    for action in actions:
      newScore = minNode(gameState.generateSuccessor(0, action), 1, self.depth, numGhosts)
      if newScore > score:
        score = newScore
        resAction = action
    return resAction





class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    def maxNode(gameState, depth, numGhosts, alpha, beta):
      if gameState.isWin() or gameState.isLose() or depth == 0:
        return self.evaluationFunction(gameState)
      actions = gameState.getLegalActions(0)
      maxScore = -float("inf")
      for action in actions:
        node = minNode(gameState.generateSuccessor(0, action), 1, depth - 1, numGhosts, alpha, beta)
        maxScore = max(maxScore, node)
        if maxScore >= beta:
          return maxScore
        alpha = max(alpha, maxScore)
      return alpha

    def minNode(gameState, agent, depth, numGhosts, alpha, beta):
      if gameState.isWin() or gameState.isLose() or depth == 0:
        return self.evaluationFunction(gameState)
      minScore = float("inf")
      actions = gameState.getLegalActions(agent)
      if agent == numGhosts:
        for action in actions:
          node = maxNode(gameState.generateSuccessor(agent, action), depth - 1, numGhosts, alpha, beta)
          minScore = min(minScore, node)
          if minScore <= alpha:
            return minScore
          beta = min(beta, minScore)
      else:
        for action in actions:
          node = minNode(gameState.generateSuccessor(agent, action), agent + 1, depth - 1, numGhosts, alpha, beta)
          minScore = min(minScore, node)
          if minScore <= alpha:
            return minScore
          beta = min(beta, minScore)
      return minScore

    numGhosts = gameState.getNumAgents() - 1
    actions = gameState.getLegalActions(0)
    resAction = Directions.STOP
    score = -float("inf")
    alpha = -float("inf")
    beta = float("inf")
    for action in actions:
      newScore = minNode(gameState.generateSuccessor(0, action), 1, self.depth, numGhosts, alpha, beta)
      if newScore > score:
        score = newScore
        resAction = action
      if score >= beta:
        return resAction
      alpha = max(alpha, score)
    return resAction

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
    def maxNode(gameState, depth, numGhosts):
      if gameState.isWin() or gameState.isLose() or depth == 0:
        return self.evaluationFunction(gameState)
      actions = gameState.getLegalActions(0)
      expValues = [expValue(gameState.generateSuccessor(0, action), 1, depth) for action in actions]
      return max(expValues)

    def expValue(gameState, agent, depth):
      if gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState)
      numGhosts = gameState.getNumAgents() - 1
      actions = gameState.getLegalActions(agent)
      prob = float(1) / len(actions)
      if agent == numGhosts:
        values = [maxNode(gameState.generateSuccessor(agent, action), depth - 1, numGhosts) for action in actions]
      else:
        values = [expValue(gameState.generateSuccessor(agent, action), agent + 1, depth) for action in actions]
      value = sum(values)
      return prob * value

    actions = gameState.getLegalActions(0)
    resAction = Directions.STOP
    score = -float("inf")
    for action in actions:
      newScore = expValue(gameState.generateSuccessor(0, action), 1, self.depth)
      if newScore > score:
        score = newScore
        resAction = action
    return resAction

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    The first thing I did was to include two special cases for win and loss to ensure
    they were always the highest and lowest scores, respectively.

    Next, I gathered the food positions, ghost states (whether they were scared), and
    the capsule locations. I initialized the score to the value of the scoreEvaluationFunction
    so that better game scores had a higher chance of being picked.

    the first calculation was distance to a ghost. If the ghost was scared, I wanted pacman
    to be as near as possible. It was less important to be far away if the ghost was not scared.

    Next, I decided that it was fairly important for Pacman to always be near a food item, and updated
    the score based on the minimum food distance appropriately.

    Lastly, I added to the score if Pacman did not eat a capsule while at least one scared
    ghost remained, and subtracted if capsules remained with no scared ghosts.

    The weights were all decided through trial and error, although their general relation to one
    another (higher/lower) did not really change throughout the process.
  """
  "*** YOUR CODE HERE ***"
  # Special cases to encourage winning/discourage losing
  if currentGameState.isWin():
    return float("inf")
  if currentGameState.isLose():
    return -float("inf")

  # Needed state values
  pos = currentGameState.getPacmanPosition()
  food = currentGameState.getFood()
  foodPos = food.asList()
  numGhosts = currentGameState.getNumAgents() - 1
  ghostStates = currentGameState.getGhostStates()
  scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
  capsules = currentGameState.getCapsules()

  score = scoreEvaluationFunction(currentGameState)

  for i in range(1, numGhosts + 1):
    ghostPos = currentGameState.getGhostPosition(i)
    dist = util.manhattanDistance(pos, ghostPos)
    if scaredTimes[i - 1] > 0:
      # Ghost is scared, want to be close
      score -= 3 * dist
    else:
      score += dist

  foodDistances = [util.manhattanDistance(pos, foodPosition) for foodPosition in foodPos]
  minFoodDist = min(foodDistances)
  score -= 2 * minFoodDist

  if max(scaredTimes) > 0:
    score += 2.5 * len(capsules)
  else:
    score -= 2.5 * len(capsules)

  return score



# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()
