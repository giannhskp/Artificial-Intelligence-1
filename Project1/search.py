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
    from util import Stack      # i use stack to store the frontier
    frontier = Stack()          # initialize frontier
    explored = set()           # keep the nodes that we have visited (that were popped from the stack). I use a set() in order to be able to search a node in O(1) time
    path = []               # a list that tracks the path for every node
    if problem.isGoalState(problem.getStartState()):        # check if first node is the goal
        return []                                           # return empty path as we are already in the goal node
    frontier.push((problem.getStartState(), path))           # initalize frontier stack with the start node
    # in the stack are pushed tuples that contain the node and it's path from the start node
    while(not frontier.isEmpty()):                      # loop until the stack is empty
        currentState, path = frontier.pop()             # pop the tuple from the stack and get the node and the path that it has followed
        if problem.isGoalState(currentState):           # check if this node is the goal
            return path                                 # if it is return the path
        explored.add(currentState)                   # add the node(state) to he explored list
        successors = problem.getSuccessors(currentState)    # get the successors of the node (expand the node)
        for currentSucc in successors:                      # for every successor node
            if currentSucc[0] not in explored:              # if we have not already explored this node
                newPath = path + [currentSucc[1]]          # add the node to the path of his parent node to create the new node's path
                frontier.push((currentSucc[0], newPath))    # add this successor node(with his new path) to the stack

    return []       # if the stack is empty. That means that there is no path from the start node to the goal node


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from util import Queue      # i use queue to store the frontier
    frontier = Queue()          # initialize frontier
    explored = set()           # keep the nodes that we have visited (that were popped from the stack). I use a set() in order to be able to search a node in O(1) time
    path = []               # a list that tracks the path for every node
    if problem.isGoalState(problem.getStartState()):        # check if first node is the goal
        return []                                           # return empty path as we are already in the goal node
    frontier.push((problem.getStartState(), path))           # initalize frontier queue with the start node
    # in the queue are pushed tuples that contain the node and it's path from the start node
    while(not frontier.isEmpty()):
        currentState, path = frontier.pop()             # pop the tuple from the queue and get the node and the path that it has followed
        if problem.isGoalState(currentState):           # check if this node is the goal
            return path                                 # if it is return the path
        explored.add(currentState)                      # add the node that we popped to the explored nodes
        successors = problem.getSuccessors(currentState)    # get the successors of the node (expand the node)
        for currentSucc in successors:                      # for every successor node (child node)
            if currentSucc[0] not in explored:              # check if we have already explored the node. If so, skip this node
                if currentSucc[0] not in (frntd[0] for frntd in frontier.list):  # check if the succresor node already exists in the queue. If so, skip the node (dont insert it again in the frontier)
                    newPath = path + [currentSucc[1]]          # add the node to the path of his parent node to create the new node's path
                    frontier.push((currentSucc[0], newPath))    # insert the new node to the frontier queue
    return []       # if the queue is empty. That means that there is no path from the start node to the goal node


def uniformCostSearch(problem):                 # read the documentation file for a better explanation of this function and why i chose to implement it in this way.
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue      # i use PriorityQueue to store the frontier
    frontier = PriorityQueue()          # initialize frontier
    explored = set()           # keep the nodes that we have visited (that were popped from the queue)
    path = []               # a list that tracks the path for every node
    keepPaths = {}           # dictionary that keeps the path and it's cost of every node that is currently in the frontier (explained more on the Documentation file)
    if problem.isGoalState(problem.getStartState()):        # check if first node is the goal
        return []                                           # return empty path as we are already in the goal node
    frontier.push(problem.getStartState(), 0)           # initalize frontier queue with the start node
    # in the PriorityQueue is pushed the node (without the path) and the priority
    keepPaths[problem.getStartState()] = (path, 0)  # add the path and the cost of the first node in the dictionary
    while(not frontier.isEmpty()):
        currentState = frontier.pop()             # pop the node with smaller priority from the stack
        (path, _) = keepPaths[currentState]       # find and store the path of the node we popped from the dictionary (in time O(1))
        keepPaths.pop(currentState)               # delete the node from the dictionary as it was popped from the PriorityQueue (frontier)
        if problem.isGoalState(currentState):           # check if this node is the goal
            return path
        explored.add(currentState)                      # add the node to the explored set
        successors = problem.getSuccessors(currentState)    # get the successors of the node (expand the node)
        for currentSucc in successors:                      # for every successor node
            nodeInFrontier = (currentSucc[0] in (frntd[2] for frntd in frontier.heap))      # frntd[2] because heap contains tuple in the form of (priority,count,item) and we only want item
            # nodeInFrontier variable keeps a boollean value. It is used so as not to do the same funcrion 2 times as we need it in both the following if statements.
            if (currentSucc[0] not in explored) and (not nodeInFrontier):       # if the node is not in explored or frontier
                newPath = path + [currentSucc[1]]          # add the node to the path of his parent node to create the new node's path
                newCost = problem.getCostOfActions(newPath)  # find the node's cost using the new path
                frontier.push(currentSucc[0], newCost)      # add the new node to the frontier PriorityQueue using the new cost as priority
                keepPaths[currentSucc[0]] = (newPath, newCost)  # add the new node with it's path and cost to the dictionary
            elif (currentSucc[0] not in explored) and nodeInFrontier:   # if the node exists only in the frontier
                (_, oldCost) = keepPaths[currentSucc[0]]    # find the old cost of this node using the dictionary
                newPath = path + [currentSucc[1]]          # add the node to the path of his parent node to create the new node's path
                newCost = problem.getCostOfActions(newPath)  # find the cost of this node using his new path
                if(oldCost > newCost):                       # if the new cost is better (smaller) than the old one
                    frontier.update(currentSucc[0], newCost)    # update the cost (priority) of this node in he frontier PriorityQueue
                    keepPaths[currentSucc[0]] = (newPath, newCost)  # also update the dictionary for this node with the new (and better) path and cost
    # this implementation with the dictionary was chosen beacause passing a tuple (item,path) as an item to the PriorityQueue makes the update function not work properly
    # when passing a tuple as item, the update function compares tuples to find if they already exist in the queue and change their priority.
    # In an expample that we want to update a nodes priority we will give as an argument to update function a tuple with the same node but a different (better) path.
    # In this case update compares also the paths (not only the nodes) that they are different and therefore never updates the priority of the node.
    # So we end up with a frontier with the same node multiple times (with different paths and cost/priorities) instead of updating the node (something that autograder does not detect)
    # My implemention solves this problem as a node exists only one time at the frontier (with the best path and cost) as it is updated every time it has to.
    return []

# def uniformCostSearch(problem):
#     from util import PriorityQueue      # i use PriorityQueue to store the frontier
#     frontier = PriorityQueue()          # initialize frontier
#     explored = set()           # keep the nodes that we have visited (that were popped from the queue)
#     path = []               # a list that tracks the path for every node
#     if problem.isGoalState(problem.getStartState()):        # check if first node is the goal
#         return []                                           # return empty path as we are already in the goal node
#     frontier.push((problem.getStartState(), path), 0)           # initalize frontier queue with the start node
#     while(not frontier.isEmpty()):
#         currentState, path = frontier.pop()             # pop the tuple from the queue and get the node and the path that it has followed
#         if problem.isGoalState(currentState):           # check if this node is the goal
#             return path                                 # if it is return the path
#         explored.add(currentState)                      # add the node to the explored set
#         successors = problem.getSuccessors(currentState)    # get the successors of the node (expand the node)
#         for currentSucc in successors:                       # for every successor node
#             nodeInFrontier = (currentSucc[0] in (frntd[2][0] for frntd in frontier.heap))         #same as above
#             newPath = path + [currentSucc[1]]          # add the node to the path of his parent node to create the new node's path
#             newCost = problem.getCostOfActions(newPath)   # find the node's cost using the new path
#             if (currentSucc[0] not in explored) and (not nodeInFrontier):     # if the node is not in explored or frontier
#                 frontier.push([currentSucc[0], newPath], newCost)             # just push it in to the frontier priority queue
#             elif nodeInFrontier:                  # if the node exists only in the frontier
#                 for x in frontier.heap:               # find the node in the frontier
#                     if x[2][0] == currentSucc[0]:
#                         oldCost = problem.getCostOfActions(x[2][1])       # get the old cost of this node
#                 if oldCost > newCost:         # compare it with the new cost. If the new cost is smaller update the path and the cost of this node. Else do nothing
#                     frontier.update((currentSucc[0], newPath), newCost)
#
#     return []         # there is no solution
    # this implementation has the downside explained in the previous function


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue     # i use PriorityQueue to store the frontier
    frontier = PriorityQueue()          # initialize frontier
    explored = set()           # keep the nodes that we have visited (that were popped from the queue)
    path = []               # a list that tracks the path for every node
    if problem.isGoalState(problem.getStartState()):        # check if first node is the goal
        return []                                           # return empty path as we are already in the goal node
    frontier.push((problem.getStartState(), path), problem.getCostOfActions(path) + heuristic(problem.getStartState(), problem))           # initalize frontier queue with the start node
    # the priority is equal to f(n) = g(n) + h(n)   where g(n) is the cost of the path from the start and h(n) is the result of the given heuristic function for the given node
    # the item pushed in the queue is a tuple that contains tha node and it's path from the starting node
    while(not frontier.isEmpty()):
        currentState, path = frontier.pop()             # pop the tuple from the queue and get the node and the path that it has followed
        if problem.isGoalState(currentState):           # check if this node is the goal
            return path                                 # if it is return the path
        if currentState in explored:     # check if we already explored this node. If we did, the current cost will be bigger so just skip him
            continue
        explored.add(currentState)          # add the node to the explored set
        successors = problem.getSuccessors(currentState)    # get the successors of the node (expand the node)
        for currentSucc in successors:
            if currentSucc[0] not in explored:
                newPath = path + [currentSucc[1]]          # add the node to the path of his parent node to create the new node's path
                frontier.push((currentSucc[0], newPath), problem.getCostOfActions(newPath) + heuristic(currentSucc[0], problem))  # push the node into the queue with priority f(n) (explained above)
    return []           # no solution path


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
