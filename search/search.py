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
    
    # Defining the fringe stack and visited nodes list and initial start state
    fringe_stack = util.Stack()
    visited_nodes = []
    start_state = (problem.getStartState(), [])

    # Pushing the initial start state into the fringe
    fringe_stack.push(start_state)
    

    while not fringe_stack.isEmpty():
        node, current_path = fringe_stack.pop()

        if node in visited_nodes:
            continue
    
        visited_nodes.append(node)

        if problem.isGoalState(node):
            return current_path
         
        successors = problem.getSuccessors(node)
        
        if not len(successors):
            continue

        for successor in successors:
            if successor[0] in visited_nodes:
                continue
            fringe_stack.push((successor[0], current_path + [successor[1]]))

    return []


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    fringe_queue = util.PriorityQueue()
    visited_nodes = []
    start_state = (problem.getStartState(), [], 0)
    
    fringe_queue.push(start_state, 0)

    while not fringe_queue.isEmpty():
        node = fringe_queue.pop()

        if node[0] in visited_nodes:
            continue

        visited_nodes.append(node[0])
        
        if problem.isGoalState(node[0]):
            return node[1]

        successors = problem.getSuccessors(node[0])
        
        if not len(successors):
            continue

        for successor in successors:
            if successor[0] in visited_nodes:
                continue

            state = (successor[0], node[1] + [successor[1]], successor[2])
            fringe_queue.push(state, len(visited_nodes) + 1)

    return []
    


def uniformCostSearch(problem):
    """Search the node of least total cost first."""

    fringe_queue = util.PriorityQueue()
    visited_nodes = []
    start_state = (problem.getStartState(), [], 0) 
    cost = 0
   
    fringe_queue.push(start_state, cost) # Push the root node in the priority queue with the priority as path cost
    
    while not fringe_queue.isEmpty():
        node = fringe_queue.pop()
        
        if node[0] in visited_nodes:
            continue

        visited_nodes.append(node[0])
    
        cost = node[2]

        if problem.isGoalState(node[0]):
            return node[1]

        successors = problem.getSuccessors(node[0])

        if not len(successors):
            continue
        
        for successor in successors:
            if successor[0] in visited_nodes:
                continue

            uptd_cost = successor[2] + cost
            state = (successor[0], node[1] + [successor[1]], uptd_cost)
            fringe_queue.push(state, successor[2] + uptd_cost)

    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    fringe_queue = util.PriorityQueue()
    visited_nodes = []
    start_state = problem.getStartState()
    cost = 0
    heuristic_node = heuristic(start_state, problem)
    
    fringe_queue.push((start_state, [], 0), cost + heuristic_node) # Push the root node in the priority queue with the priority as path cost and heuristic
    
    while not fringe_queue.isEmpty():
        node = fringe_queue.pop()
        
        if node[0] in visited_nodes:
            continue

        visited_nodes.append(node[0])
        cost = node[2]

        if problem.isGoalState(node[0]):
            return node[1]

        successors = problem.getSuccessors(node[0])
        
        for successor in successors:
            if successor[0] in visited_nodes:
                continue

            heuristic_node = heuristic(successor[0], problem)
            uptd_cost = successor[2] + cost
            state = (successor[0], node[1] + [successor[1]], uptd_cost)
            fringe_queue.push(state, uptd_cost + heuristic_node)
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
