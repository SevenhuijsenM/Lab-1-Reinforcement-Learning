import time
import matplotlib.pyplot as plt
import numpy as np
from IPython import display

# All the methods that are currently implemented in the class Maze
methods = ['DynProg']

# Some colours
LIGHT_RED    = '#FFC4CC' # Color for the minotaur
LIGHT_GREEN  = '#95FD99' # Color of the start
BLACK        = '#000000' # Color of blocked walls
WHITE        = '#FFFFFF' # Color where there is nothing
LIGHT_PURPLE = '#E8D0FF' # Color of the exit
LIGHT_ORANGE = '#FAE0C3' # Color of the player

class Maze:
    """
    This class represents a Maze object, which is used to model a maze for a game. 
    The maze has a player and various actions that the player can perform.
    There is also a minotaur that moves around the maze and tries to catch the player.
    """
    # Actions of the player
    STAY       = 0
    MOVE_LEFT  = 1
    MOVE_RIGHT = 2
    MOVE_UP    = 3
    MOVE_DOWN  = 4

    # Actions of the minotaur
    MOVE_LEFT_MINOTAUR  = 0
    MOVE_RIGHT_MINOTAUR = 1
    MOVE_UP_MINOTAUR    = 2
    MOVE_DOWN_MINOTAUR  = 3

    # Give names to actions of the player
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }

    # Give names to actions of the minotaur
    actions_names_minotaur = {
        MOVE_LEFT_MINOTAUR: "move left",
        MOVE_RIGHT_MINOTAUR: "move right",
        MOVE_UP_MINOTAUR: "move up",
        MOVE_DOWN_MINOTAUR: "move down"
    }

    # Reward values
    STEP_REWARD = -1
    GOAL_REWARD = 0
    IMPOSSIBLE_REWARD = -1000
    CAUGHT_MINOTAUR = -1000

    # Initialise the maze environment
    def __init__(self, maze, minotaur_position = None):
        self.maze_layout                            = maze
        self.actions_player, self.actions_minotaur  = self.__actions()
        self.states, self.map                       = self.__states()
        self.n_actions                              = len(self.actions_player)
        self.n_states                               = len(self.states)
        self.n_actions_minotaur                     = len(self.actions_minotaur)
        self.min_probabilities                      = self.__minotaur_transition()
        self.transition_probabilities               = self.__transitions()
        self.rewards                                = self.__rewards()
        self.start_minautar, self.start, self.end   = \
                self.__get_start_positions(minotaur_position)

    def __actions(self):
        actions_player = dict()
        actions_player[self.STAY]       = (0, 0)
        actions_player[self.MOVE_LEFT]  = (0, -1)
        actions_player[self.MOVE_RIGHT] = (0, 1)
        actions_player[self.MOVE_UP]    = (-1, 0)
        actions_player[self.MOVE_DOWN]  = (1, 0)

        actions_minotaur = dict()
        actions_minotaur[self.MOVE_LEFT_MINOTAUR]  = (0, -1)
        actions_minotaur[self.MOVE_RIGHT_MINOTAUR] = (0, 1)
        actions_minotaur[self.MOVE_UP_MINOTAUR]    = (-1, 0)
        actions_minotaur[self.MOVE_DOWN_MINOTAUR]  = (1, 0)
        return actions_player, actions_minotaur

    def __states(self):
        states = dict()
        map_states = dict()
        s = 0

        # For each of the locations of the player and minotaur we have a state
        for i in range(self.maze_layout.shape[0]):
            for j in range(self.maze_layout.shape[1]):
                for k in range(self.maze_layout.shape[0]):
                    for l in range(self.maze_layout.shape[1]):
                        # There is no state where the player is in a wall (maze is 1)
                        if self.maze_layout[i, j] != 1:
                            states[s] = ((i, j), (k, l))
                            map_states[((i, j), (k, l))] = s
                            s += 1

        return states, map_states

    def __get_minotaur_move_possibilities(self, minotaur_state):
        """ Gets the amount of possible moves for the minotaur given a state
            :return int possibilities: The amount of possible moves for the minotaur
        """
        # The minotaur should have at least one possible move
        # Therefore the shape should be at least > 1x1
        assert  self.maze_layout.shape[0] > 1 and               \
                self.maze_layout.shape[1] > 1

        # Look at the coordinates of the minotaur, it cannot move into the boundaries
        possibilities = np.array([  self.MOVE_LEFT_MINOTAUR,    \
                                    self.MOVE_RIGHT_MINOTAUR,   \
                                    self.MOVE_UP_MINOTAUR,      \
                                    self.MOVE_DOWN_MINOTAUR]    )

        # Check if the minotaur is at the left or right boundary
        if minotaur_state[1] == 0:
            possibilities = possibilities[possibilities != self.MOVE_LEFT_MINOTAUR]
        if minotaur_state[1] == self.maze_layout.shape[1] - 1:
            possibilities = possibilities[possibilities != self.MOVE_RIGHT_MINOTAUR]

        # Check if the minotaur is at the top or bottom boundary
        if minotaur_state[0] == 0:
            possibilities = possibilities[possibilities != self.MOVE_UP_MINOTAUR]
        if minotaur_state[0] == self.maze_layout.shape[0] -1:
            possibilities = possibilities[possibilities != self.MOVE_DOWN_MINOTAUR]

        # If this returns 0 the minotaur is trapped, give a error message
        if possibilities.size == 0:
            print("Minotaur is trapped, this should not happen")

        return possibilities

    def __move(self, state, action, action_minotaur):
        """ Makes a step in the maze, given a current position and an action.
            Both the player and the minotaur make a step
            If the action STAY or an inadmissible action is used, the agent stays in place.
            :return tuple of tuples next_cell: Position ((y, x), (y_minotaur, x_minotaur))
            on the maze that agent and minotaur transitions to.
        """
        # Compute the future position given current (state, action, action_minotaur)
        current_cell = self.states[state]

        # get the rows and columns of the player and minotaur
        row_player = current_cell[0][0] + self.actions_player[action][0]
        col_player = current_cell[0][1] + self.actions_player[action][1]
        row_minotaur = current_cell[1][0] + self.actions_minotaur[action_minotaur][0]
        col_minotaur = current_cell[1][1] + self.actions_minotaur[action_minotaur][1]
        next_cell = ((row_player, col_player), (row_minotaur, col_minotaur))

        # Is the future position an impossible one ?
        player_hitting_maze_walls = (row_player == -1) or \
                                    (row_player == self.maze_layout.shape[0]) or \
                                    (col_player == -1) or \
                                    (col_player == self.maze_layout.shape[1]) or \
                                    (self.maze_layout[row_player, col_player] == 1)

        # Based on the impossiblity check return the next state.
        if player_hitting_maze_walls:
            # The player hit a wall, stay in place, but the minotaur can still move
            return self.map[(current_cell[0], next_cell[1])]
        else:
            # Both player and minotaur can move
            return self.map[next_cell]

    def __minotaur_transition(self):
        # For each state and action we compute the probability of the minotaur moving
        dimensions = (self.n_states, self.n_actions, self.n_actions_minotaur)
        minotaur_probabilities = np.zeros(dimensions)

        # For each state and action and minotaur action
        for s in range(self.n_states):
            for a in range(self.n_actions):
                # First look at the possible moves of the minotaur
                possibilities = self.__get_minotaur_move_possibilities(self.states[s][1])

                # For each possible move of the minotaur, compute the next state
                for a_m in range(self.n_actions_minotaur):
                    # If the move is in the possibilities then the probability is shared
                    if a_m in possibilities:
                        minotaur_probabilities[s, a, a_m] = 1 / len(possibilities)
                    else:
                        minotaur_probabilities[s, a, a_m] = 0
        return minotaur_probabilities

    def __transitions(self):
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        # Initialize the transition probailities tensor (S,S,A)
        dimensions = (self.n_states, self.n_states, self.n_actions)
        transition_probabilities = np.zeros(dimensions)

        # Compute the transition probabilities. Note that the transitions
        # are non-deterministic.
        for s in range(self.n_states):
            for a in range(self.n_actions):
                # For each each possible minotaur move
                for a_m in range(self.n_actions_minotaur):
                    if self.min_probabilities[s, a, a_m] > 0:
                        # Compute the next state given the current state and action
                        next_s = self.__move(s, a, a_m)
                        transition_probabilities[next_s, s, a] = self.min_probabilities[s, a, a_m]
        return transition_probabilities

    def __calculate_partial_reward(self, s, a, a_m):
        # Get the probability of this transaction
        probability = self.min_probabilities[s, a, a_m]

        # The current and the next state
        current_s = self.states[s]
        next_s = self.states[self.__move(s, a, a_m)]

        # If the minotaur catches the player
        if next_s[0] == next_s[1]:
            return probability * self.CAUGHT_MINOTAUR

        # If the player hits a wall
        elif next_s[0] == current_s[0] and a != self.STAY:
            return probability * self.IMPOSSIBLE_REWARD

        # If the player reaches the exit
        elif self.maze_layout[next_s[0]] == 2:
            return probability * self.GOAL_REWARD

        # If the player takes a step to an empty cell that is not the exit
        else:
            return probability * self.STEP_REWARD

    def __rewards(self):
        # Initialize the reward tensor (S,A)
        rewards = np.zeros((self.n_states, self.n_actions))

        # For each state and action the reward is computed
        for s in range(self.n_states):
            for a in range(self.n_actions):
                # Sum for the reward using a weighted average
                reward_sum = 0

                for a_m in self.actions_minotaur:
                    if self.min_probabilities[s, a, a_m] > 0:
                        reward_sum += self.__calculate_partial_reward(s, a, a_m)

                # Set the reward for this state and action
                rewards[s,a] = reward_sum
        return rewards

    def simulate_solo(self, policy, method, start_position_minotaur):
        """ Simulates the shortest path problem given a policy
            :param numpy.ndarray policy: The policy to follow
            :param str method: The method to use to solve the maze
            :return list path: The path to follow
        """

        if method not in methods:
            error = f'ERROR: the argument method must be in {methods}'
            raise NameError(error)

        # Create a starting state
        start = ((self.start[0], self.start[1]), start_position_minotaur)
                
        path = list()
        if method == 'DynProg':
            # Deduce the horizon from the policy shape
            horizon = policy.shape[1]
            # Initialize current state and time
            t = 0
            s = self.map[start]
            # Add the starting position in the maze to the path
            path.append(start)
            while t < horizon - 1:
                # Move to next state given the policy and the current state
                next_s = self.__move_without_minotaur(s, policy[s, t])
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                # Update time and state for next iteration
                t += 1
                s = next_s
        return path

    def simulate(self, policy, method):
            """ Simulates the shortest path given random actions of the minotaur
                :param numpy.ndarray policy: The policy to follow
                :param str method: The method to use to solve the maze
                :return list path: The path to follow
            """

            if method not in methods:
                error = f'ERROR: the argument method must be in {methods}'
                raise NameError(error)

            # Create a starting state
            start = ((self.start[0], self.start[1]), (self.start_minautar[0], self.start_minautar[1]))
            
            path = list()
            if method == 'DynProg':
                # Deduce the horizon from the policy shape
                horizon = policy.shape[1]

                # Initialize current state and time
                t = 0
                s = self.map[start]
                # Add the starting position in the maze to the path
                path.append(start)
                while t < horizon - 1:
                    # For each of the probabilities
                    prob_vector = self.transition_probabilities[:, s, int(policy[s, t])]
                    next_s = np.random.choice(self.n_states, p=prob_vector)
                    path.append(self.states[next_s])

                    # Update time and state for next iteration
                    t += 1
                    s = next_s
            return path

    def __get_start_positions(self, minotaur_position = None):
        """ Get all the positions from the grid
            :return tuple start_minautar: The position of the minotaur
            :return tuple start: The position of the player at the start
            :return tuple end: The position of the exit
        """
        # Give an error if the maze does not have an exit
        if 2 not in self.maze_layout:
            print("The maze does not have an exit")

        # Give an error if the maze does not have a start
        if 3 not in self.maze_layout:
            print("The maze does not have a start")

        # Get the coordinates of the exit
        coordinates_exit = np.argwhere(self.maze_layout == 2)[0]

        # Get the coordinates of the player at the start
        coordinates_player = np.argwhere(self.maze_layout == 3)[0]

        # Get the coordinates of the minotaur
        if minotaur_position:
            coordinates_minotaur = minotaur_position
        else:
            coordinates_minotaur = coordinates_exit

        # Return the coordinates of the exit, player and minotaur
        return coordinates_minotaur, coordinates_player, coordinates_exit

    def __move_without_minotaur(self, state, action):
        """ Make a step in the maze without the minotaur"""
        # Get the tuples of the current state
        current_cell = self.states[state]

        # Compute the future position given current (state, action)
        row = current_cell[0][0] + self.actions_player[action][0]
        col = current_cell[0][1] + self.actions_player[action][1]

        # Is the future position an impossible one ?
        hitting_maze_walls =  (row == -1) or (row == self.maze_layout.shape[0]) or \
                              (col == -1) or (col == self.maze_layout.shape[1]) or \
                              (self.maze_layout[row,col] == 1)

        # Update the state
        if not hitting_maze_walls:
            return self.map[((row, col), current_cell[1])]
        else:
            return state

def dynamic_programming(env, horizon):
    """ Solves the shortest path problem using dynamic programming
        :param Maze env: The environment to solve
        :param int horizon: The time horizon
        :return numpy.ndarray V: The optimal state value function
        :return numpy.ndarray policy: The optimal policy
    """
    # The dynamic prgramming requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p         = env.transition_probabilities
    r         = env.rewards
    n_states  = env.n_states
    n_actions = env.n_actions
    T         = horizon

    # The variables involved in the dynamic programming backwards recursions
    V      = np.zeros((n_states, T+1))
    policy = np.zeros((n_states, T+1))

    # Initialization
    Q            = np.copy(r)
    V[:, T]      = np.max(Q, 1)
    policy[:, T] = np.argmax(Q, 1)

    # The dynamic programming bakwards recursion
    for t in range(T - 1, -1 ,-1):
        # Update the value function acccording to the bellman equation
        for s in range(n_states):
            for a in range(n_actions):
                # Update of the temporary Q values
                Q[s, a] = r[s, a] + np.dot(p[:, s, a],V[:, t + 1])
        # Update by taking the maximum Q value w.r.t the action a
        V[:, t] = np.max(Q, 1)

        # The optimal action is the one that maximizes the Q function
        policy[:, t] = np.argmax(Q, 1)
    return V, policy

def draw_maze(maze):
    """Draws the maze environment
        :param numpy.ndarray maze: The maze to draw
    """

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_PURPLE, 3: LIGHT_GREEN, 4: LIGHT_RED}

    # Remove the axis ticks and add title title
    ax = plt.gca()
    ax.set_title('The Maze')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    rows,cols    = maze.shape
    colored_maze = [[col_map[maze[j, i]] for i in range(cols)] for j in range(rows)]

    # Create a table to color
    grid = plt.table(cellText=None,
                            cellColours=colored_maze,
                            cellLoc='center',
                            loc=(0,0),
                            edges='closed')

    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows)
        cell.set_width(1.0/cols)

def animate_solution(maze, path):
    """ Animates the shortest path found by the dynamic programming algorithm
        :param list path: The path to animate
        :param numpy.ndarray maze: The maze environment
    """
    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_PURPLE, 3: LIGHT_GREEN, 4: LIGHT_RED}

    # Size of the maze
    rows, cols = maze.shape

    # Remove the axis ticks and add title title
    ax = plt.gca()
    ax.set_title('Policy simulation')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    colored_maze = [[col_map[maze[j, i]] for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0,0),
                     edges='closed')

    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0 / rows)
        cell.set_width(1.0 / cols)

    # Create an empty function
    prev_cell = None

    # Update the color at each frame
    for coordinate in path:
        # Set the cell of the player with the color
        grid.get_celld()[coordinate[0]].set_facecolor(LIGHT_ORANGE)
        grid.get_celld()[coordinate[0]].get_text().set_text('Player')

        # Move the minotaur if the player is not out
        if prev_cell is  None or coordinate[0] != prev_cell[0]:
            grid.get_celld()[coordinate[1]].set_facecolor(LIGHT_RED)
            grid.get_celld()[coordinate[1]].get_text().set_text('Minotaur')
        

        if prev_cell is not None:
            # If the player does not move and it is at the exit then the player is out
            if coordinate[0] == prev_cell[0] and maze[coordinate[0]] == 2:
                grid.get_celld()[(coordinate[0])].set_facecolor(LIGHT_GREEN)
                grid.get_celld()[(coordinate[0])].get_text().set_text('Player is out')

            # Set the old cell with the previous color of the player
            elif coordinate[1] != prev_cell[0] and coordinate[0] != prev_cell[0] :
                grid.get_celld()[(prev_cell[0])].set_facecolor(col_map[maze[prev_cell[0]]])
                grid.get_celld()[(prev_cell[0])].get_text().set_text('')

            # Set the old cell with the previous color of the minotaur
            if coordinate[0] != prev_cell[1] and coordinate[0] != prev_cell[0] and coordinate[1] != prev_cell[1]:
                grid.get_celld()[(prev_cell[1])].set_facecolor(col_map[maze[prev_cell[1]]])
                grid.get_celld()[(prev_cell[1])].get_text().set_text('')


        prev_cell = coordinate
        display.display(fig)
        display.clear_output(wait=True)
        time.sleep(1)
