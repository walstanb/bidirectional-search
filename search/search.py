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
    """
    :return: The manhattan distance between two locations in the 2D space
    """
    x1, y1 = node1
    x2, y2 = node2
    return abs(x1 - x2) + abs(y1 - y2)


def meetInTheMiddle(problem, heuristic=nullHeuristic):
    """
    The "Meet in the Middle" heuristic-based Bi-direction search
    :param problem: The specific problem in the PACMAN domain, e.g., Position Search Problem
    :param heuristic: The heuristic, e.g., NUll heuristic, Manhattan Distance
    :return: The optimal action for the agent
    """
    def getActionSequence(initial_state, goal_state, parent_info):
        """
        Returns the path from the initial_state to the goal_state using the parent_info of each node
        :param initial_state: the state to start from
        :param goal_state: the state to end at
        :param parent_info: the parent information stored for every node
        :return:
        """
        action_sequence = []

        state = goal_state

        # Find the path from the goal state to the initial state using the parent information of each node
        while state != initial_state:
            parent_state, action_parent_child = parent_info[state]
            action_sequence.append(action_parent_child)
            state = parent_state

        # Reverse the path from goal to start to get the path from start to goal
        action_sequence.reverse()

        return action_sequence

    def getMinimumGAndFValue(open_list, initial_state, goal_state, parent_info):
        """
        Returns the minimum g-value and f-value from the open list, given the initial and the final state
        :param open_list: the fringe
        :param initial_state: the initial state of the agent
        :param goal_state: the final state of the agent
        :param parent_info: the parent information stored for every node from start to goal
        :return:
        """
        f_values = []
        g_values = []

        # For each node in the open list calculate the g_value and the f_value
        for node in open_list:
            _, _, state = node
            action_sequence_start_to_current = getActionSequence(initial_state, state, parent_info)

            # g_value is the cost of the path from the initial state to the current state
            g_value = problem.getCostOfActions(action_sequence_start_to_current, initial_state)

            # h_value is the estimate of the cheapest path from the current state to the goal state
            h_value = heuristic(state, goal_state)

            g_values.append(g_value)

            # f_value = g_value + h_value
            f_values.append(g_value + h_value)

        # Return the minimum of all g_values and f_values
        return min(g_values), min(f_values)

    def getOppositeAction(current_action):
        """
        Returns the opposite direction action given the current direction
        :param current_action: one of the 4 possible actions: East, West, North, South
        :return:
        """
        if current_action == Directions.EAST:
            return Directions.WEST
        if current_action == Directions.WEST:
            return Directions.EAST
        if current_action == Directions.NORTH:
            return Directions.SOUTH
        if current_action == Directions.SOUTH:
            return Directions.NORTH
        return None

    # The initial state in the forward direction is the start state of the problem
    initial_state_forward = problem.getStartState()
    # The initial state in the backward direction is the goal state of the problem
    initial_state_backward = problem.goal

    # Both the open lists are Priority Queues
    open_list_forward = util.PriorityQueue()
    open_list_backward = util.PriorityQueue()

    # Initially both the open lists contain the corresponding initial state
    open_list_forward.push(initial_state_forward, heuristic(initial_state_forward, initial_state_backward))
    open_list_backward.push(initial_state_backward, heuristic(initial_state_backward, initial_state_forward))

    # A list of explored nodes in both directions
    explored_nodes_forward = []
    explored_nodes_backward = []

    # A dictionary to keep the parent information of each state, in both directions
    # Key: the child state
    # Value: the parent state, and the action from the parent state to reach the child state
    parent_info_forward = {}
    parent_info_backward = {}

    # Initialization
    U = float('inf')
    e = 1.0

    # The Meet in the Middle Algorithm
    while not (open_list_forward.isEmpty() and open_list_backward.isEmpty()):

        # Get the min g_value, and f_value in forward direction
        min_g_val_forward, min_f_val_forward = getMinimumGAndFValue(
            open_list=open_list_forward.heap,
            initial_state=initial_state_forward,
            goal_state=initial_state_backward,
            parent_info=parent_info_forward
        )

        # Get the min g_value, and f_value in backward direction
        min_g_val_backward, min_f_val_backward = getMinimumGAndFValue(
            open_list=open_list_backward.heap,
            initial_state=initial_state_backward,
            goal_state=initial_state_forward,
            parent_info=parent_info_backward
        )

        # Get the minimum priority from the open lists, in both direction
        min_priority_forward = heapq.nsmallest(1, open_list_forward.heap)[0][0]
        min_priority_backward = heapq.nsmallest(1, open_list_backward.heap)[0][0]

        # The current cost is the minimum of both directional minimum priorities
        C = min(min_priority_forward, min_priority_backward)

        # The terminating condition: Meet in the middle!
        if U <= max(C, min_f_val_forward, min_f_val_backward, min_g_val_forward + min_g_val_backward + e):

            # Get the action sequence from the start to goal through the middle node
            # Path(start, goal) = Path(start, middle) + Opposite(Path(goal, middle))
            action_sequence_forward = getActionSequence(initial_state_forward, middle_node, parent_info_forward)
            action_sequence_backward = getActionSequence(initial_state_backward, middle_node, parent_info_backward)

            inverted_action_sequence_backward = []

            for action in action_sequence_backward:
                reverse_action = getOppositeAction(action)
                inverted_action_sequence_backward.append(reverse_action)

            inverted_action_sequence_backward.reverse()
           
            # For display purposes only
            problem.isGoalState(problem.goal)

            return action_sequence_forward + inverted_action_sequence_backward

        # The forward search
        if C == min_priority_forward:

            # Get the minimum priority state from the open list
            current_state = open_list_forward.pop()

            # Perform actions on the state only if it has not been explored already
            if current_state not in explored_nodes_forward:

                # Get the children states of the current states
                children_nodes = problem.getSuccessors(current_state)
                # Mark the current_state as explored to avoid infinite loops
                explored_nodes_forward.append(current_state)

                for child_node in children_nodes:
                    child_state, action, step_cost = child_node

                    # If the child state has not already been explored, perform the following actions
                    if child_state not in explored_nodes_forward:

                        # If the child node has already been seen through some of its other parent
                        # compare its already seen path, and the current path, and keep the cheaper one in the open list
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

                        # If the child node is seen for the first time, push it to the open list
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

                    # If the child state has already been explored in the backward algorithm, it is a potential
                    # candidate to be the middle node
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
         # The backward search
        else:
            # Get the minimum priority state from the open list
            current_state = open_list_backward.pop()

            # Perform actions on the state only if it has not been explored already
            if current_state not in explored_nodes_backward:
                children_nodes = problem.getSuccessors(current_state)
                explored_nodes_backward.append(current_state)

                for child_node in children_nodes:
                    child_state, action, step_cost = child_node

                    #Verify if the child node exists in the explored nodes
                    if child_state not in explored_nodes_backward:

                        if child_state not in parent_info_backward.keys():
                            parent_info_backward[child_state] = (current_state, action)
                            
                            #Get sequence of actions to go from initial state to child node 
                            action_sequence_to_child = getActionSequence(
                                initial_state=initial_state_backward,
                                goal_state=child_state,
                                parent_info=parent_info_backward
                            )

                            g_val = problem.getCostOfActions(action_sequence_to_child, initial_state_backward)
                            h_val = heuristic(child_state, initial_state_forward)

                            priority = max(g_val + h_val, 2 * g_val)

                            open_list_backward.push(child_state, priority)

                        # If the child node is seen for the first time, push it to the open list
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

                    # If the child state has already been explored in the forward algorithm, it is a potential
                    # candidate to be the middle node    
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

    return []

def meetInMiddleCornerSearch(problem, heuristic=nullHeuristic):
    
    '''Formulating Meet in the Middle for Corner Search Problem'''

    def corners_heuristic(state, problem, direction): # calculate corner heuristic

        max_val = -1 
        if(direction=='Forward'): # if direction is forward
            for iters in range(1,5):
                max_val = max(max_val, (util.manhattanDistance(state[0], state[iters][0]) * int(state[iters][1])) ) # calculating manhattan distance and understanding if this is a corner or not
        else: # if direction is backward
            for iters in range(1,5):
                max_val = max(max_val, (util.manhattanDistance(state[0], state[iters][0]) * (not int(state[iters][1]))) ) # calculating manhattan distance and understanding if this is a corner or not
        return max_val

    def get_action_sequence(parent_child_info, initial_state, goal):

        """
        Returns the path from the initial_state to the goal using the parent_child_info of each node 
        :param parent_child_info: the parent information stored for every node
        :param initial_state: the state to start from
        :param goal: the state to end at
        :return:
        """

        action_sequence = []                                        
        current_state = goal                               

        # Find the path from the goal state to the initial state using the parent information of each node
        
        while(initial_state != current_state):
            parent_state, action_parent_child = parent_child_info[current_state]
            action_sequence.append(action_parent_child)
            current_state = parent_state
        
        # Reverse the path from goal to start to get the path from start to goal
        action_sequence.reverse()
        return action_sequence

    def get_min_g_f(nodes, initial_state, parent_child_info, problem, search_direction):
        
        """
        Returns the minimum g-value and f-value from the open list, given the initial and the final state
        :param nodes: the fringe
        :param initial_state: the initial state of the agent
        :param parent_child_info: the parent information stored for every node from start to goal
        :param search_direction: to understand forward or backward search
        :return:
        """

        g_values = []
        f_values = []
        
        # For each node in the open list calculate the g_value and the f_value
        for node in nodes:
            action_sequence=get_action_sequence(parent_child_info, initial_state, node[2])
             # g_value is the cost of the path from the initial state to the current state
            if(search_direction == 'Forward'):
                g_value= problem.getCostOfActions(action_sequence)
            else:
                g_value = problem.getCostOfActions(action_sequence, initial_state[0])
            # h_value is the estimate of the cheapest path from the current state to the goal state
            h_value = corners_heuristic(node[2], problem, search_direction)
            g_values.append(g_value)
            f_values.append(g_value + h_value)
        
        # Return the minimum of all g_values and f_values
        return min(g_values), min(f_values)
    
    def getOppositeAction(current_action):
        
        """
        Returns the opposite direction action given the current direction
        :param current_action: one of the 4 possible actions: East, West, North, South
        :return:
        """

        if current_action == Directions.EAST:
            return Directions.WEST
        if current_action == Directions.WEST:
            return Directions.EAST
        if current_action == Directions.NORTH:
            return Directions.SOUTH
        if current_action == Directions.SOUTH:
            return Directions.NORTH
        return None
    
    # defining the constants
    U = float("inf")    
    threshold = 1         


    # fringes are lists which are Priority Queue
    open_list_forward=util.PriorityQueue()                                                          
    open_list_backward=util.PriorityQueue() 

    # to hold parent information of each state
    parent_link_forward={}
    parent_link_backward={}            
                                                            
    explored_forward_nodes = []                              
    explored_nodes_backward = []                             
    
    # The initial state in the forward direction is the start state of the problem
    initial_state_forward = problem.getStartState()
    # initial state of our agent
    initial_pacman_position= initial_state_forward[0] 
    
    all_corners=problem.corners                              
    distance_dict = {}                                  
    for corners in all_corners: 
        distance = util.manhattanDistance(corners, initial_pacman_position)
        distance_dict[distance]=corners

    
    start_state_backward = distance_dict[max(distance_dict.keys())]  # start state of backward search is the farthest from the initial state
    initial_state_backward = []
    initial_state_backward.append((tuple(start_state_backward)))

    for corners in all_corners:
        if(corners == start_state_backward):
            initial_state_backward.append((corners, True)) # True, if this corner is start state
        else:
            initial_state_backward.append((corners, False)) # False, otherwise
    
    initial_state_backward=tuple(initial_state_backward)  # The initial state of the backward search

    open_list_forward.push(initial_state_forward,corners_heuristic(initial_state_forward, problem, 'Forward'))     # appending initial state for forward search 
    open_list_backward.push(initial_state_backward,corners_heuristic(initial_state_backward, problem, 'Backward')) # appending initial state for backward search 
    
    while not (open_list_forward.isEmpty() and open_list_backward.isEmpty()) : # loop until the fringes are empty
   
        min_g_val_forward, min_f_val_forward = get_min_g_f(open_list_forward.heap, initial_state_forward, parent_link_forward, problem, 'Forward') # find minimum g and f value in the forward direction
        min_g_val_backward, min_f_val_backward = get_min_g_f(open_list_backward.heap, initial_state_backward, parent_link_backward, problem, 'Backward') # find minimum g and f value in the backward direction
        
        min_priority_forward = heapq.nsmallest(1,open_list_forward.heap)[0][0] # get minimum priority value forward
        min_priority_backward = heapq.nsmallest(1, open_list_backward.heap)[0][0] # get minimum priority value backward
        C = min(min_priority_forward, min_priority_backward)

        if(U <= max(C, min_f_val_forward, min_f_val_backward, min_g_val_forward + min_g_val_backward + threshold)):  # The terminating condition: Meet in the middle!

            # Get the action sequence from the start to goal through the middle node
            # Path(start, goal) = Path(start, middle) + Opposite(Path(goal, middle))

            action_sequence_forward=get_action_sequence(parent_link_forward, initial_state_forward, middle_node)
            action_sequence_backward=get_action_sequence(parent_link_backward, initial_state_backward, middle_node)
            inverted_action_sequence_backward = []
            for action in action_sequence_backward:
                reverse_action = getOppositeAction(action)
                inverted_action_sequence_backward.append(reverse_action)
            inverted_action_sequence_backward.reverse()

            # For display purposes only
            problem.isGoalState(problem.goal)

            return action_sequence_forward + inverted_action_sequence_backward

        # The forward search

        elif (C== min_priority_forward):
            
            # Get the minimum priority state from the open list
            current_state = open_list_forward.pop()  
            # Perform actions on the state only if it has not been explored already
            if current_state not in explored_forward_nodes:
                
                # Get the children states of the current states
                children_nodes = problem.getSuccessors(current_state)  
                # Mark the current_state as explored to avoid infinite loops
                explored_forward_nodes.append(current_state)
                for children in children_nodes:

                    child_state, action, step_cost = children
                    # If the child state has not already been explored, perform the following actions
                    if(child_state not in explored_forward_nodes):
                        
                        # If the child node has already been seen through some of its other parent
                        # compare its already seen path, and the current path, and keep the cheaper one in the open list

                        if child_state not in parent_link_forward.keys():
                            parent_link_forward[child_state] = (current_state, action)
                            action_sequence_to_child = get_action_sequence(parent_link_forward, initial_state_forward, child_state)
                            g_val = problem.getCostOfActions(action_sequence_to_child, initial_pacman_position)
                            h_val = corners_heuristic(child_state, problem, 'Forward')
                            priority = max(g_val + h_val, 2*g_val)
                            open_list_forward.push(child_state, priority)

                        else:
                             # If the child node is seen for the first time, push it to the open list
                            action_sequence_to_child = get_action_sequence(parent_link_forward, initial_state_forward, child_state)
                            action_sequence_to_current = get_action_sequence(parent_link_forward, initial_state_forward, current_state)
                            old_g_val = problem.getCostOfActions(action_sequence_to_child)                          
                            new_g_val = problem.getCostOfActions(action_sequence_to_current)+children[2]
                            h_val = corners_heuristic(child_state, problem, 'Forward')
                            if(old_g_val > new_g_val + h_val):
                                open_list_forward.push(child_state,new_g_val+h_val)
                                parent_link_forward[child_state]=(current_state, action)
                                priority= max(new_g_val+h_val, 2*new_g_val)    
                                open_list_forward.push(child_state, priority)
                        
                        # If the child state has already been explored in the backward algorithm, it is a potential
                        # candidate to be the middle node
                        if(child_state in explored_nodes_backward):
                            action_sequence_start_to_child = get_action_sequence(parent_link_forward, initial_state_forward, child_state)
                            action_sequence_goal_to_child = get_action_sequence(parent_link_backward, initial_state_backward, child_state)
                            g_val_forward = problem.getCostOfActions(action_sequence_start_to_child)
                            g_val_backward = problem.getCostOfActions(action_sequence_goal_to_child , initial_state_backward[0])
                            U = min(U, g_val_forward + g_val_backward)
                            if(g_val_forward + g_val_backward==U):
                                middle_node = child_state
        # The backward search
        else:
            # Get the minimum priority state from the open list
            current_state = open_list_backward.pop()  
            # Perform actions on the state only if it has not been explored already
            if current_state not in explored_nodes_backward :
                children_nodes=problem.getSuccessors(current_state, True) 
                explored_nodes_backward.append(current_state)

                for children in children_nodes:
                    child_state, action, step_cost = children
                    # Verify if the child node exists in the explored nodes
                    if child_state not in explored_nodes_backward :
                        if child_state not in parent_link_backward.keys():

                            parent_link_backward[child_state] = (current_state, action)
                            # Get sequence of actions to go from initial state to child node 
                            action_sequence_to_child = get_action_sequence(parent_link_backward, initial_state_backward, child_state)
                            g_val = problem.getCostOfActions(action_sequence_to_child,initial_state_backward[0])
                            h_val = corners_heuristic(child_state, problem, 'Backward')
                            priority= max(g_val + h_val, 2 * g_val)
                            open_list_backward.push(child_state, priority)

                        else:
                            # If the child node is seen for the first time, push it to the open list
                            action_sequence_start_to_child = get_action_sequence(parent_link_backward, initial_state_backward, child_state)
                            action_sequence_goal_to_child = get_action_sequence(parent_link_backward, initial_state_backward, current_state)
                            
                            old_g_val = problem.getCostOfActions(action_sequence_start_to_child, initial_state_backward[0])                          
                            new_g_val = problem.getCostOfActions(action_sequence_goal_to_child, initial_state_backward[0]) + children[2]
                            h_val = corners_heuristic(child_state, problem, 'Backward')
                            if(old_g_val > new_g_val + h_val):
                                open_list_forward.push(child_state,new_g_val+h_val)
                                parent_link_backward[child_state]=(current_state, action)
                                priority= max(new_g_val+h_val, 2*new_g_val)    
                                open_list_backward.push(child_state, priority)

                        # If the child state has already been explored in the forward algorithm, it is a potential
                        # candidate to be the middle node  
                        if(child_state in explored_forward_nodes):

                            action_sequence_start_to_child = get_action_sequence(parent_link_forward, initial_state_forward, child_state)
                            action_sequence_goal_to_child = get_action_sequence(parent_link_backward, initial_state_backward, child_state)
                            g_val_forward = problem.getCostOfActions(action_sequence_start_to_child)
                            g_val_backward = problem.getCostOfActions(action_sequence_goal_to_child , initial_state_backward[0]) 
                            U = min(U, g_val_forward + g_val_backward)   
                            if(g_val_forward + g_val_backward==U):
                                middle_node = child_state

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
mm = meetInTheMiddle
mm_corner = meetInMiddleCornerSearch