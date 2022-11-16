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

    return None

def meetInMiddleCornerSearch(problem, heuristic=nullHeuristic):
    """
        This function implements the 'Meet in the Middle' Bi-directional heuristic search algorithm as introduced in the paper,
        "Bidirectional Search That Is Guaranteed to Meet in the Middle" by Robert C. Holte,Ariel Felner, Guni Sharon and Nathan R. Sturtevant;
        In the Proceedings of the Thirtieth AAAI Conference on Artificial Intelligence (AAAI-16)
        This function implements the meet in the middle algorithm for the corner search problem in Berkley AI project
    """    

    print("Meet in the middle Algorithm for corner search problem Initiated")


    """ Importing the required libraries """
    import time
    import util    
    import heapq
    import random

    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    n=Directions.NORTH
    e=Directions.EAST

    actionDictionary={'North':n, 'South':s, 'East':e, 'West':w}


    def manhatten(position, corner):
            "The Manhattan distance heuristic for a Corner Search Problem"
            xy1 = position
            xy2 = corner
            return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])
            #return 0


    def cornersHeuristic(state, problem, direction):
        """
        A heuristic for the CornersProblem that you defined.
          state:   The current search state
                   (a data structure you chose in your search problem)
          problem: The CornersProblem instance for this layout.
        This function should always return a number that is a lower bound on the
        shortest path from the state to a goal of the problem; i.e.  it should be
        admissible (as well as consistent).
        """
        corners = problem.corners # These are the corner coordinates
        walls = problem.walls # These are the walls of the maze, as a Grid (game.py)

        

        def manhatten(position, corner):
            "The Manhattan distance heuristic for a Corner Search Problem"
            xy1 = position
            xy2 = corner
            return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])
            #return 0    

        """ 
        Manhattan Distance of individual corners when there is food available at that particular corner 
        or else it is zero 
        """ 

        if(direction=='Forward'):

            manhatten_CR1= manhatten(state[0], state[1][0]) * int(state[1][1]) 
            manhatten_CR2= manhatten(state[0], state[2][0]) * int(state[2][1])
            manhatten_CR3= manhatten(state[0], state[3][0]) * int(state[3][1]) 
            manhatten_CR4= manhatten(state[0], state[4][0]) * int(state[4][1])
        
        else:    
            manhatten_CR1= manhatten(state[0], state[1][0]) * (not(int(state[1][1]))) 
            manhatten_CR2= manhatten(state[0], state[2][0]) * (not(int(state[2][1])))
            manhatten_CR3= manhatten(state[0], state[3][0]) * (not(int(state[3][1]))) 
            manhatten_CR4= manhatten(state[0], state[4][0]) * (not(int(state[4][1])))
        
        return max(manhatten_CR1,manhatten_CR2,manhatten_CR3,manhatten_CR4)

    def backtrace(parentChild, initialState, goalNode):
        """ 
        Function to backtrace the actions from initialState to the GoalNode.
        It starts from the goalNode and traces back to the initialNode using the successive parentNode link
        """ 
        actions_=[]                                         #Initializing an empty list to store the actions

        currentState=goalNode                               #Initializing a currentState variable, which shall act as counter till the intialState is reached

        while(currentState!=initialState):
            """ Backtracing is done until we reach the initial state"""
            actions_.insert(0, parentChild[currentState][1])
            currentState=parentChild[currentState][0]       #Updating the currentState to the Parent State
        
        return actions_                                     #Returning the actions_ list

    def get_both(allNodes, initialState, parentChildLink, problem, searchDirection):
        f_values = []
        g_values = []

        if(searchDirection=='Forward'):
            for node in allNodes:
                seqAction=backtrace(parentChildLink, initialState, node[2])
                g_value= problem.getCostOfActions(seqAction)
                h_value = cornersHeuristic(node[2], problem, searchDirection)
                g_values.append(g_value)
                f_values.append(g_value + h_value)
        else:
            for node in allNodes:
                seqAction=backtrace(parentChildLink, initialState, node[2])
                #g_value = problem.getCostOfActionsBackward(seqAction, initialState[0])
                g_value = problem.getCostOfActions(seqAction, initialState[0])
                h_value = cornersHeuristic (node[2], problem, searchDirection)
                g_values.append(g_value)
                f_values.append(g_value + h_value)
        
        return min(g_values), min(f_values)



    
    
    frontierStatesForward=util.PriorityQueue()              # A priority list data structure to keep the frontier nodes for exploring from start state to the goal state.
                                                            #Its defined in the util. Nodes with least cost are prioritized first.

    frontierStatesBackward=util.PriorityQueue()             #A priority list data structure to keep the frontier nodes for exploring from goal state to the start state.
                                                            #Its defined in the util. Nodes with least cost are prioritized first.

    exploredStatesForward = []                              #List of all the states that have been explored/expanded while exploring from start state to goal state direction
    exploredStatesBackward = []                             #List of all the states that have been explored/expanded while exploring from goal state to start state direction

    
    
    initialState_Forward = problem.getStartState()          #Getting the intial State of the search problem for the forward direction search
    initialPacmanPosition_Forward = initialState_Forward[0] #Extracting the Pacman Position from the initial state
    
    allCorners=problem.corners                              #Fetching the coordinates of the corner positions
    

    cornerPositionDist={}                                   #Dictionary to store the manhattan distance of the corner position and its coordinate
    for corner in allCorners: 
        distance=manhatten(corner, initialPacmanPosition_Forward)
        cornerPositionDist[distance]=corner

    
    startStateBackward = cornerPositionDist[max(cornerPositionDist.keys())] # The corner which is farthest is used as the start position from the backward position   

    initialState_Backward = []
    initialState_Backward.append((tuple(startStateBackward)))#Getting the initial State of the search problem for the backward direction search

    for corner in allCorners:
        if(corner==startStateBackward):
            initialState_Backward.append((corner, True))
        else:
            initialState_Backward.append((corner, False))
    
    initialState_Backward=tuple(initialState_Backward)      #Initializing the food existance states for the backward direction search. False for all the positions except the start position of the reverse pacman
    

    frontierStatesForward.push(initialState_Forward,cornersHeuristic(initialState_Forward, problem, 'Forward'))      #Appending the intial state into the frontierStatesForward list
    frontierStatesBackward.push(initialState_Backward,cornersHeuristic(initialState_Backward,problem, 'Backward'))    #Appending the intial state into the frontierStatesBackward list
    
    
    U = float('inf')    # Initialzing the cost to the the goal state
    epsilon = 1         # Cost of the minimum edge in state space


    parentChildForward={}
    parentChildBackward={}
    """
        Description for the parentChild dictionary
        A dictionary to link the child node with the parent node, which will be used to backtrace the actions.
        - Keys in the dictionary represent the state represented by the child node.
        - The values are a set that represent the 'parent node state' and 'action' required to get from parent state to the respective child state.
    """


    while(1):

        # Checking if both the Priority Queues are empty.
        # If both are empty then no such path exist.
        if(frontierStatesForward.isEmpty() and frontierStatesBackward.isEmpty()):
            print('Both frontierStatesForward and frontierStatesBackward are empty.')
            print('No path from start to goal state exist.')
            return None

        else:
            
            gmin_Forward, fmin_Forward = get_both(frontierStatesForward.heap, initialState_Forward, parentChildForward, problem, 'Forward')
            gmin_Backward, fmin_Backward = get_both(frontierStatesBackward.heap, initialState_Backward, parentChildBackward, problem, 'Backward')
            
            # Fetching the highest priority values in each search directions
            minPriorityValueinForwardQueue = heapq.nsmallest(1,frontierStatesForward.heap)[0][0]
            minPriorityValueinBackwardQueue = heapq.nsmallest(1, frontierStatesBackward.heap)[0][0]

            
            minValue= min(minPriorityValueinForwardQueue, minPriorityValueinBackwardQueue)


            if(U <= max(minValue, fmin_Forward, fmin_Backward, gmin_Forward + gmin_Backward + epsilon)):

                seqAction1=backtrace(parentChildForward, initialState_Forward, midNode)
                seqAction2=backtrace(parentChildBackward, initialState_Backward, midNode)
                
                print(len(seqAction1))
                print(len(seqAction2))

                seqAction2New=[]

                for action in seqAction2:
                    if(action=='East'):
                        seqAction2New.insert(0,'West')

                    if(action=='West'):
                        seqAction2New.insert(0, 'East')    
                    
                    if(action=='South'):
                        seqAction2New.insert(0, 'North')    
                    
                    if(action=='North'):
                        seqAction2New.insert(0, 'South')    
    
                seqAction=seqAction1 + seqAction2New

                """Return the Action Seqeuence"""
                print("The cost of the path is: " + str(U))
                print("The mid-node is: " + str(midNode))
                return seqAction

                break

            else:

                if(minValue==minPriorityValueinForwardQueue):
                    parentNodeForward = frontierStatesForward.pop()  #Pop the node which highest priority        
                    if(parentNodeForward not in exploredStatesForward):
                        newFrontierNodes=problem.getSuccessors(parentNodeForward)  #Get the successor nodes of the node which was just popped 
                        exploredStatesForward.append(parentNodeForward)

                        for childNodes in newFrontierNodes:
                            """Verifying if each child node exists in explored states set or in the frontier set"""
                            if(childNodes[0] not in exploredStatesForward):

                                """We check if the childNode is already present in the childParent Dictionary"""
                                if(childNodes[0] not in parentChildForward.keys()):

                                    """Linking the ChildNode with the ParentNodeForward and keeping it in a dictionary"""

                                    parentChildForward[childNodes[0]]=(parentNodeForward, childNodes[1])

                                    """Calling the backtracing function to get the the sequence of actions which are required to go
                                       from initial state to the current childNode state
                                    """
                                    seqAction=backtrace(parentChildForward, initialState_Forward, childNodes[0])

                                    Cost_=problem.getCostOfActions(seqAction) + cornersHeuristic(childNodes[0], problem, 'Forward')

                                    priorityValue= max(Cost_, 2*problem.getCostOfActions(seqAction))

                                    """Adding the childNode state to the frontier list along with the Cost_ to reach the ChildNode"""
                                    frontierStatesForward.push(childNodes[0], priorityValue)


                                else:
                                    """ If the childNode is already present, we update the child:Parent link only if the current cost is lesser than the past cost"""

                                    """Getting the past cost"""
                                    Past_Cost=problem.getCostOfActions(backtrace(parentChildForward, initialState_Forward, childNodes[0]))
                                    """Getting the current cost"""
                                    Cost_=problem.getCostOfActions(backtrace(parentChildForward, initialState_Forward, parentNodeForward))+childNodes[2]

                                    """Adding the heuristic cost"""
                                    Cost_= Cost_ + cornersHeuristic(childNodes[0], problem, 'Forward')

                                    if(Past_Cost > Cost_):
                                        """Adding the childNode to the frontier list"""
                                        frontierStatesForward.push(childNodes[0],Cost_)
                                        """Updating the Child:Parent link in the parentChildForward linkup"""
                                        parentChildForward[childNodes[0]]=(parentNodeForward, childNodes[1])

                                        priorityValue= max(Cost_, 2*problem.getCostOfActions(seqAction))    
                                        frontierStatesForward.push(childNodes[0], priorityValue)

                        
                                if(childNodes[0] in exploredStatesBackward):
                                    costofStartStatetoNode = problem.getCostOfActions(backtrace(parentChildForward, initialState_Forward, childNodes[0]))
                                    #costofGoalStatetoNode  = problem.getCostOfActionsBackward(backtrace(parentChildBackward, initialState_Backward, childNodes[0]), initialState_Backward[0])
                                    costofGoalStatetoNode  = problem.getCostOfActions(backtrace(parentChildBackward, initialState_Backward, childNodes[0]), initialState_Backward[0])
                                    
                                    # Update U 
                                    U = min(U, costofStartStatetoNode+costofGoalStatetoNode)
                                    if((costofStartStatetoNode+costofGoalStatetoNode)==U):
                                        midNode = childNodes[0]

                else:
                    

                    parentNodeBackward = frontierStatesBackward.pop()  #Pop the node which highest priority        
                    if(parentNodeBackward not in exploredStatesBackward):
                        newFrontierNodes=problem.getSuccessors(parentNodeBackward, True) #Get the successor nodes of the node which was just popped
                        exploredStatesBackward.append(parentNodeBackward)

                        for childNodes in newFrontierNodes:
                            """Verifying if each child node exists in explored states set or in the frontier set"""
                            if(childNodes[0] not in exploredStatesBackward):

                                """We check if the childNode is already present in the childParent Dictionary"""
                                if(childNodes[0] not in parentChildBackward.keys()):

                                    """Linking the ChildNode with the parentNodeBackward and keeping it in a dictionary"""
                                    parentChildBackward[childNodes[0]]=(parentNodeBackward, childNodes[1])

                                    """Calling the backtracing function to get the the sequence of actions which are required to go
                                       from initial state to the current childNode state
                                    """
                                    seqAction=backtrace(parentChildBackward, initialState_Backward, childNodes[0])

                                    #Cost_=problem.getCostOfActionsBackward(seqAction,initialState_Backward[0]) + cornersHeuristic(childNodes[0], problem, 'Backward')
                                    Cost_=problem.getCostOfActions(seqAction,initialState_Backward[0]) + cornersHeuristic(childNodes[0], problem, 'Backward')

                                    #priorityValue= max(Cost_, 2*problem.getCostOfActionsBackward(seqAction, initialState_Backward[0]))
                                    priorityValue= max(Cost_, 2*problem.getCostOfActions(seqAction, initialState_Backward[0]))
                                   
                                    """Adding the childNode state to the frontier list along with the Cost_ to reach the ChildNode"""
                                    frontierStatesBackward.push(childNodes[0], priorityValue)


                                else:
                                    """ If the childNode is already present, we update the child:Parent link only if the current cost is lesser than the past cost"""

                                    """Getting the past cost"""
                                    #Past_Cost=problem.getCostOfActionsBackward(backtrace(parentChildBackward, initialState_Backward, childNodes[0]), initialState_Backward[0])
                                    Past_Cost=problem.getCostOfActions(backtrace(parentChildBackward, initialState_Backward, childNodes[0]), initialState_Backward[0])
                                    """Getting the current cost"""
                                   # Cost_=problem.getCostOfActionsBackward(backtrace(parentChildBackward, initialState_Backward, parentNodeBackward), initialState_Backward[0])+childNodes[2]
                                    Cost_=problem.getCostOfActions(backtrace(parentChildBackward, initialState_Backward, parentNodeBackward), initialState_Backward[0])+childNodes[2]
                                    """Adding the heuristic cost"""
                                    Cost_= Cost_ + cornersHeuristic(childNodes[0], problem, 'Backward')

                                    if(Past_Cost > Cost_):
                                        """Adding the childNode to the frontier list"""
                                        frontierStatesBackward.push(childNodes[0],Cost_)
                                        """Updating the Child:Parent link in the parentChildBackward linkup"""
                                        parentChildBackward[childNodes[0]]=(parentNodeBackward, childNodes[1])

                                        #priorityValue= max(Cost_, 2*problem.getCostOfActionsBackward(seqAction), initialState_Backward[0])
                                        priorityValue= max(Cost_, 2*problem.getCostOfActions(seqAction), initialState_Backward[0])
  
                                        frontierStatesBackward.push(childNodes[0], priorityValue)

                            
                                if(childNodes[0] in exploredStatesForward):
                                    
                                    costofStartStatetoNode = problem.getCostOfActions(backtrace(parentChildForward, initialState_Forward, childNodes[0]))
                                    #costofGoalStatetoNode  = problem.getCostOfActionsBackward(backtrace(parentChildBackward, initialState_Backward, childNodes[0]), initialState_Backward[0])
                                    costofGoalStatetoNode  = problem.getCostOfActions(backtrace(parentChildBackward, initialState_Backward, childNodes[0]), initialState_Backward[0])
                                    
                                    # Update U 

                                    U = min(U, costofStartStatetoNode+costofGoalStatetoNode)
                                    if((costofStartStatetoNode+costofGoalStatetoNode)==U):
                                        midNode = childNodes[0]

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
mm = meetInTheMiddle
mm1 = meetInMiddleCornerSearch