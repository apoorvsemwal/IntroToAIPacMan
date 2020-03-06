# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
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
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    frontier = util.Stack()  # Initialize the stack.
    closed = set()  # Initialize a visited set that contain the visited nodes during traversal.
    start = problem.getStartState()  # Get the start state of agent.
    frontier.push((start, []))  # push the start state into the stack.
    return execute_common_search_logic(frontier, closed, problem)


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    frontier = util.Queue()  # Initialize the Queue.
    closed = set()  # Initialize a visited set that contain the visited nodes during traversal.
    start = problem.getStartState()  # Get the start state of agent.
    frontier.push((start, []))  # push the start state and empty list of actions into the queue.
    return execute_common_search_logic(frontier, closed, problem)


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    def priority_func(path):
        return problem.getCostOfActions(path[1])

    frontier = util.PriorityQueueWithFunction(priority_func)  # Priority Queue with priority function based on
    # cost of path.
    closed = set()  # Initialize a visited set that contain the visited nodes during traversal.
    start = problem.getStartState()  # Get the start state of agent.
    frontier.push((start, []))  # push the start state into the Priority Queue.
    return execute_common_search_logic(frontier, closed, problem)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    def priority_func(path):
        act_cost_gn = problem.getCostOfActions(path[1])
        est_cost_fn = heuristic(path[0], problem)
        return act_cost_gn + est_cost_fn

    frontier = util.PriorityQueueWithFunction(priority_func)  # Priority Queue with priority function based on
    # cost of path.
    closed = set()  # Initialize a visited set that contain the visited nodes during traversal.
    start = problem.getStartState()  # Get the start state of agent.
    frontier.push((start, []))  # push the start state into the Priority Queue.
    return execute_common_search_logic(frontier, closed, problem)


def execute_common_search_logic(frontier, closed, problem):
    while not frontier.isEmpty():
        current_node, current_action = frontier.pop()  # remove out the node(start) and actions taken by the
        # agent(n,e,w,s)
        if problem.isGoalState(current_node):
            return current_action  # Anchor condition
        if current_node not in closed:
            closed.add(current_node)  # If node not visited, add in set.
            node_successors = problem.getSuccessors(current_node)  # Get the successors of the current Node.
            for current_child in node_successors:
                # For each child node, get it's child nodes and action to get there.
                current_successor, action, _ = current_child
                # Pushing successor along with the action required to reach that successor
                frontier.push((current_successor, current_action + [action]))
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
