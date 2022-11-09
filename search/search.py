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
import heapq
from game import Directions


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

    start_state = problem.getStartState()
    closed = []
    fringe = util.Stack()
    # The fringe contains a state and the path to the state
    # The fringe is a Stack
    fringe.push((start_state, []))

    while not fringe.isEmpty():
        if fringe.isEmpty():
            return []
        node = fringe.pop()
        state, path = node
        if problem.isGoalState(state):
            return path
        if state not in closed:
            closed.append(state)
            for successor in problem.getSuccessors(state):
                nextState, action, cost = successor
                fringe.push((nextState, path + [action]))

    return []


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    start_state = problem.getStartState()
    closed = []
    fringe = util.Queue()
    # The fringe contains a state and the path to the state
    # The fringe is a queue
    fringe.push((start_state, []))

    while not fringe.isEmpty():
        if fringe.isEmpty():
            return []
        node = fringe.pop()
        state, path = node
        if problem.isGoalState(state):
            return path
        if state not in closed:
            closed.append(state)
            for successor in problem.getSuccessors(state):
                nextState, action, cost = successor
                fringe.push((nextState, path + [action]))

    return []


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    start_state = problem.getStartState()
    closed = []
    fringe = util.PriorityQueue()
    # The fringe contains a state, the g() value, the path to the state
    # The fringe is a priority queue with g() as the priority
    fringe.push((start_state, [], 0), 0)  # g(start) = 0

    while not fringe.isEmpty():
        if fringe.isEmpty():
            return []
        node = fringe.pop()
        state, path, g_val = node
        if problem.isGoalState(state):
            return path
        if state not in closed:
            closed.append(state)
            for successor in problem.getSuccessors(state):
                nextState, action, cost = successor
                fringe.push((nextState, path + [action], g_val + cost), g_val + cost)

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

    start_state = problem.getStartState()
    closed = []
    fringe = util.PriorityQueue()
    # The fringe contains a state, the g() value, the path to the state
    # The fringe is a priority queue with f() as the priority
    g_start = 0
    h_start = heuristic(start_state, problem)
    f_start = g_start + h_start
    fringe.push((start_state, [], g_start), f_start)

    while not fringe.isEmpty():
        if fringe.isEmpty():
            return []
        node = fringe.pop()
        state, path, g_val = node
        if problem.isGoalState(state):
            return path
        if state not in closed:
            closed.append(state)
            for successor in problem.getSuccessors(state):
                nextState, action, cost = successor
                fringe.push((nextState, path + [action], g_val + cost), g_val + cost + heuristic(nextState, problem))

    return []


def manhattanHeuristic(node1, node2):
    x1, y1 = node1
    x2, y2 = node2
    return abs(x1 - x2) + abs(y1 - y2)


def mm(problem, heuristic=nullHeuristic):

    def getActionSequence(initial_state, goal_state, parent_info):

        action_sequence = []

        state = goal_state

        while state != initial_state:
            parent_state, action_parent_child = parent_info[state]
            action_sequence.append(action_parent_child)
            state = parent_state

        action_sequence.reverse()
        return action_sequence

    def getMinimumGAndFValue(open_list, initial_state, goal_state, parent_info):

        f_values = []
        g_values = []

        for node in open_list:
            _, _, state = node
            action_sequence_start_to_current = getActionSequence(initial_state, state, parent_info)

            g_value = problem.getCostOfActions(action_sequence_start_to_current, initial_state)
            h_value = heuristic(state, goal_state)

            g_values.append(g_value)
            f_values.append(g_value + h_value)

        return min(g_values), min(f_values)

    def getOppositeAction(current_action):
        if current_action == Directions.EAST:
            return Directions.WEST
        if current_action == Directions.WEST:
            return Directions.EAST
        if current_action == Directions.NORTH:
            return Directions.SOUTH
        if current_action == Directions.SOUTH:
            return Directions.NORTH
        return None

    initial_state_forward = problem.getStartState()
    initial_state_backward = problem.goal

    open_list_forward = util.PriorityQueue()
    open_list_backward = util.PriorityQueue()

    open_list_forward.push(initial_state_forward, heuristic(initial_state_forward, initial_state_backward))
    open_list_backward.push(initial_state_backward, heuristic(initial_state_backward, initial_state_forward))

    explored_nodes_forward = []
    explored_nodes_backward = []

    parent_info_forward = {}
    parent_info_backward = {}

    U = float('inf')
    e = 1.0

    while not (open_list_forward.isEmpty() and open_list_backward.isEmpty()):

        min_g_val_forward, min_f_val_forward = getMinimumGAndFValue(
            open_list=open_list_forward.heap,
            initial_state=initial_state_forward,
            goal_state=initial_state_backward,
            parent_info=parent_info_forward
        )

        min_g_val_backward, min_f_val_backward = getMinimumGAndFValue(
            open_list=open_list_backward.heap,
            initial_state=initial_state_backward,
            goal_state=initial_state_forward,
            parent_info=parent_info_backward
        )

        min_priority_forward = heapq.nsmallest(1, open_list_forward.heap)[0][0]
        min_priority_backward = heapq.nsmallest(1, open_list_backward.heap)[0][0]

        C = min(min_priority_forward, min_priority_backward)
        # middle_node = initial_state_forward

        if U <= max(C, min_f_val_forward, min_f_val_backward, min_g_val_forward + min_g_val_backward + e):

            action_sequence_forward = getActionSequence(initial_state_forward, middle_node, parent_info_forward)
            action_sequence_backward = getActionSequence(initial_state_backward, middle_node, parent_info_backward)

            inverted_action_sequence_backward = []

            for action in action_sequence_backward:
                reverse_action = getOppositeAction(action)
                inverted_action_sequence_backward.append(reverse_action)

            inverted_action_sequence_backward.reverse()

            return action_sequence_forward + inverted_action_sequence_backward

        if C == min_priority_forward:
            current_state = open_list_forward.pop()

            if current_state not in explored_nodes_forward:
                children_nodes = problem.getSuccessors(current_state)
                explored_nodes_forward.append(current_state)

                for child_node in children_nodes:
                    child_state, action, step_cost = child_node

                    if child_state not in explored_nodes_forward:

                        if child_state not in parent_info_forward.keys():
                            parent_info_forward[child_state] = (current_state, action)

                            action_sequence_to_child = getActionSequence(
                                initial_state=initial_state_forward,
                                goal_state=child_state,
                                parent_info=parent_info_forward
                            )

                            g_val = problem.getCostOfActions(action_sequence_to_child, initial_state_forward)
                            h_val = heuristic(child_state, initial_state_backward)

                            priority = max(g_val + h_val, 2 * g_val)

                            open_list_forward.push(child_state, priority)

                        else:
                            action_sequence_to_child = getActionSequence(
                                initial_state=initial_state_forward,
                                goal_state=child_state,
                                parent_info=parent_info_forward
                            )

                            action_sequence_to_current = getActionSequence(
                                initial_state=initial_state_forward,
                                goal_state=current_state,
                                parent_info=parent_info_forward
                            )

                            old_g_val = problem.getCostOfActions(action_sequence_to_child)
                            new_g_val = problem.getCostOfActions(action_sequence_to_current) + step_cost

                            h_val = heuristic(child_state, initial_state_backward)

                            if old_g_val > new_g_val:
                                g_val = new_g_val
                                open_list_forward.push(child_state, g_val + h_val)
                                parent_info_forward[child_state] = (current_state, action)

                                priority = max(g_val + h_val, 2 * g_val)
                                open_list_forward.push(child_state, priority)

                    if child_state in explored_nodes_backward:

                        action_sequence_start_to_child = getActionSequence(
                            initial_state=initial_state_forward,
                            goal_state=child_state,
                            parent_info=parent_info_forward
                        )

                        g_val_forward = problem.getCostOfActions(action_sequence_start_to_child)

                        action_sequence_goal_to_child = getActionSequence(
                            initial_state=initial_state_backward,
                            goal_state=child_state,
                            parent_info=parent_info_backward
                        )

                        g_val_backward = problem.getCostOfActions(action_sequence_goal_to_child, initial_state_backward)

                        U = min(U, g_val_forward + g_val_backward)
                        if g_val_forward + g_val_backward == U:
                            middle_node = child_state

        else:
            current_state = open_list_backward.pop()

            if current_state not in explored_nodes_backward:
                children_nodes = problem.getSuccessors(current_state)
                explored_nodes_backward.append(current_state)

                 for child_node in children_nodes:
                    child_state, action, step_cost = child_node

                    if child_state not in explored_nodes_backward:

                        if child_state not in parent_info_backward.keys():
                            parent_info_backward[child_state] = (current_state, action)

                            action_sequence_to_child = getActionSequence(
                                initial_state=initial_state_backward,
                                goal_state=child_state,
                                parent_info=parent_info_backward
                            )

                            g_val = problem.getCostOfActions(action_sequence_to_child, initial_state_backward)
                            h_val = heuristic(child_state, initial_state_forward)

                            priority = max(g_val + h_val, 2 * g_val)

                            open_list_backward.push(child_state, priority)
                        
                        else:
                            action_sequence_to_child = getActionSequence(
                                initial_state=initial_state_backward,
                                goal_state=child_state,
                                parent_info=parent_info_backward
                            )

                            action_sequence_to_current = getActionSequence(
                                initial_state=initial_state_backward,
                                goal_state=current_state,
                                parent_info=parent_info_backward
                            )

                            old_g_val = problem.getCostOfActions(action_sequence_to_child)
                            new_g_val = problem.getCostOfActions(action_sequence_to_current) + step_cost

                            h_val = heuristic(child_state, initial_state_forward)

                            if old_g_val > new_g_val:
                                g_val = new_g_val
                                open_list_backward.push(child_state, g_val + h_val)
                                parent_info_backward[child_state] = (current_state, action)

                                priority = max(g_val + h_val, 2 * g_val)
                                open_list_backward.push(child_state, priority)
                            
                    if child_state in explored_nodes_forward:

                        action_sequence_start_to_child = getActionSequence(
                            initial_state=initial_state_forward,
                            goal_state=child_state,
                            parent_info=parent_info_forward
                        )

                        g_val_forward = problem.getCostOfActions(action_sequence_start_to_child)

                        action_sequence_goal_to_child = getActionSequence(
                            initial_state=initial_state_backward,
                            goal_state=child_state,
                            parent_info=parent_info_backward
                        )

                        g_val_backward = problem.getCostOfActions(action_sequence_goal_to_child, initial_state_backward)

                        U = min(U, g_val_forward + g_val_backward)
                        if g_val_forward + g_val_backward == U:
                            middle_node = child_state
            # pass
    return None


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch