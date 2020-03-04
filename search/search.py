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
    return  [s, s, w, s, w, w, s, w]

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
    stack = util.Stack() #Initialize the stack.
    visited = set() #Initialize a visited set that contain the visited nodes during traversal.
    start = problem.getStartState() # Get the start state of agent.
    stack.push((start, [])) #push the start state into the stack.

    while not stack.isEmpty():
        currentNode, currentAction = stack.pop()  #pop out the node(start) and actions taken by the agent(n,e,w,s)
        if problem.isGoalState(currentNode):
            return currentAction   #base case
        if currentNode not in visited:
            visited.add(currentNode)  #If node not visited, add in set.
            NodeSuccessors = problem.getSuccessors(currentNode) #Get the successors of the current Node in the stack.
            for currentchild in NodeSuccessors:
                    #For each child node, get it's child nodes and action to get there and also the cost.
                    #This will continue until the stack is empty.
                currentSuccessor, action, currentCost = currentchild
                stack.push((currentSuccessor, currentAction + [action]))
    return []
    
    
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    
    queue = util.Queue() #Initialize the Queue.
    visited = set() #Initialize a visited set that contain the visited nodes during traversal.
    start = problem.getStartState() # Get the start state of agent.
    print("Start Node", start)
    queue.push((start, [])) #push the start state and empty list of actions into the queue.

    while not queue.isEmpty():
        currentNode, currentAction = queue.pop()  #pop out the node(start) and actions taken by the agent(n,e,w,s)
        if problem.isGoalState(currentNode):
            return currentAction   #base case
        if currentNode not in visited:
            print("current Node", currentNode)
            visited.add(currentNode)  #If node not visited, add in set.
            NodeSuccessors = problem.getSuccessors(currentNode) #Get the successors of the current Node in the queue.
            print("Node Successors ", NodeSuccessors)
            for currentchild in NodeSuccessors:
                    #For each child node, get it's child nodes and action to get there and also the cost.
                    #This will continue until the stack is empty.
                currentSuccessor, action, currentCost = currentchild
                queue.push((currentSuccessor, currentAction + [action]))
    return []
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    #Incomplete.
    open = util.PriorityQueue()
    close = []
    start = problem.getStartState()

    #Base Case.
    if problem.isGoalState(start):
        return []


    cost = 0 + manhattanHeuristic(start, problem)
    open.push((start, []), cost)

    while not open.isEmpty():
        currentNode, currentAction = open.pop()
        if problem.isGoalState(currentNode):
            return currentAction

            generatedSuccessors = problem.getSuccessors(currentNode)
            for eachSuccessor in generatedSuccessors:
                successor, action, cost = eachSuccessor
                open.push((successor, currentAction + [action]), problem.getCostOfActions(currentAction) +
                          cost + heuristic(successor, problem))
    return []
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
