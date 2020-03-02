"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def uniVersalSearch(problem, state, frontier, explored, visited, heuristic=nullHeuristic):
    while not frontier.isEmpty() and not problem.isGoalState(state[0]):
        while not frontier.isEmpty():
            state = frontier.pop()
            if not state[0] in visited:  # Visited is a dictionary. potential place for danger cause of unhashable objects,
                visited[state[0]] = 1
                explored.push(state)
                if not problem.isGoalState(state[0]):  # Dont expand it
                    getChildren(state, problem, frontier, heuristic)
                break




def getChildren(state, problem, frontier, heuristic=nullHeuristic):
    temp = problem.getSuccessors(state[0])
    for i in temp:
        frontier.push([i[0], i[1], state[0], (state[3] + i[2]), heuristic(i[0], problem)])
        #i[0] - state info for child,  i[1] - Direction , state[0] - state infor for parent,  state[3] - totals cost so far , heuristic based on state info , calculated for child.
        # print(i[2])


def getPath(explored):
    previous = explored.pop()
    dir = [previous[1]]
    while not explored.isEmpty():
        current = explored.pop()
        if (current[0] == previous[2]):
            dir.insert(0, current[1])
            previous = current
    return dir


def initializeSearch(dataStructure, problem, heuristic=nullHeuristic):
    state = [problem.getStartState(), 0, 0, 0, 0]
    # Declare DataStructures
    explored = util.Stack()
    frontier = dataStructure
    visited = {state[0]: 1}
    # Initialize first elements
    ##pathMap[state] = [0, 0, 0] # This helps us to get the path later on ....
    # explored.push(state)

    getChildren(state, problem, frontier, heuristic)
    # Put the algorithm into work with stack
    uniVersalSearch(problem, state, frontier, explored, visited, heuristic)
    d = getPath(explored)
    return d


def depthFirstSearch(problem):
    frontier = util.Stack()
    actionPath = initializeSearch(frontier, problem)
    return actionPath
    # Declare DataStructures


def breadthFirstSearch(problem):
    frontier = util.Queue()
    actionPath = initializeSearch(frontier, problem)
    return actionPath


def uniformCostSearch(problem):
    frontier = util.PriorityQueueWithFunction(priorityUCS)
    actionPath = initializeSearch(frontier, problem)
    return actionPath


def aStarSearch(problem, heuristic=nullHeuristic):
    frontier = util.PriorityQueueWithFunction(
        priorityAStar)  # Defines the AsStar datastructure to use item[3] - overall distande and item[4] heuristic value
    actionPath = initializeSearch(frontier, problem, heuristic)
    return actionPath

    "Search the node that has the lowest combined cost and heuristic first."
    "*** YOUR CODE HERE ***"


def priorityUCS(item):
    return item[3]


def priorityAStar(item):
    return item[3] + item[4]


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch