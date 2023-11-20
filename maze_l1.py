import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display

# All the methods that are currently implemented in the class Maze
methods = ['DynProg'];

# Some colours
LIGHT_RED    = '#FFC4CC';
LIGHT_GREEN  = '#95FD99';
BLACK        = '#000000';
WHITE        = '#FFFFFF';
LIGHT_PURPLE = '#E8D0FF';
LIGHT_ORANGE = '#FAE0C3';

class Maze:

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

    # Reward values
    STEP_REWARD = -1
    GOAL_REWARD = 0
    IMPOSSIBLE_REWARD = -100
    CAUGHT_MINOTAUR = -100

    # Initialise the maze environment
    def __init__(self, maze, weights=None, random_rewards=False):
        self.maze                                   = maze;
        self.actions_player, self.actions_minotaur  = self.__actions();
        self.states, self.map                       = self.__states();
        self.n_actions                              = len(self.actions_player);
        self.n_states                               = len(self.states);
        self.n_actions_minotaur                     = len(self.actions_minotaur);
        self.minotaur_probabilities                 = self.__minotaur_transition();
        self.transition_probabilities               = self.__transitions();
        self.rewards                  = self.__rewards(weights=weights,
                                                random_rewards=random_rewards);
        self.start_minautar                         = self.__get_start_minautar();

    def __actions(self):
        actions_player = dict();
        actions_player[self.STAY]       = (0, 0);
        actions_player[self.MOVE_LEFT]  = (0,-1);
        actions_player[self.MOVE_RIGHT] = (0, 1);
        actions_player[self.MOVE_UP]    = (-1,0);
        actions_player[self.MOVE_DOWN]  = (1,0);
        
        actions_minotaur = dict();
        actions_minotaur[self.MOVE_LEFT_MINOTAUR]  = (0,-1);
        actions_minotaur[self.MOVE_RIGHT_MINOTAUR] = (0, 1);
        actions_minotaur[self.MOVE_UP_MINOTAUR]    = (-1,0);
        actions_minotaur[self.MOVE_DOWN_MINOTAUR]  = (1,0);       
        
        return actions_player, actions_minotaur;

    def __states(self):
        states = dict();
        map = dict();
        s = 0;
        # For each of the locations of the player and minotaur we have a state
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                for k in range(self.maze.shape[0]):
                    for l in range(self.maze.shape[1]):
                        # There is no state where the player is in a wall (maze is 1)
                        if self.maze[i, j] != 1:
                            states[s] = ((i, j), (k, l));
                            map[((i, j), (k, l))] = s;
                            s += 1;
        return states, map
    
    def __get_minotaur_move_possibilities(self, minotaur_state):
        """ Gets the amount of possible moves for the minotaur given a state
            :return int possibilities: The amount of possible moves for the minotaur
        """
        # The minotaur should have at least one possible move, therefore the shape should be at least > 1x1
        assert self.maze.shape[0] > 1 and self.maze.shape[1] > 1;
        
        # Look at the coordinates of the minotaur, it cannot move into the boundaries
        possibilities = np.array([self.MOVE_LEFT_MINOTAUR, self.MOVE_RIGHT_MINOTAUR, self.MOVE_UP_MINOTAUR, self.MOVE_DOWN_MINOTAUR])
        
        # Check if the minotaur is at the left or right boundary
        if minotaur_state[1] == 0:
            possibilities = possibilities[possibilities != self.MOVE_LEFT_MINOTAUR];
        if minotaur_state[1] == self.maze.shape[1] - 1:
            possibilities = possibilities[possibilities != self.MOVE_RIGHT_MINOTAUR];
            
        # Check if the minotaur is at the top or bottom boundary
        if minotaur_state[0] == 0:
            possibilities = possibilities[possibilities != self.MOVE_UP_MINOTAUR];
        if minotaur_state[0] == self.maze.shape[0] -1:
            possibilities = possibilities[possibilities != self.MOVE_DOWN_MINOTAUR];
            
        # If this returns 0 the minotaur is trapped, give a error message
        if possibilities.size == 0:
            print("Minotaur is trapped, this should not happen");
        
        return possibilities;

    def __move(self, state, action, action_minotaur):
        """ Makes a step in the maze, given a current position and an action.
            Both the player and the minotaur make a step
            If the action STAY or an inadmissible action is used, the agent stays in place.
            :return tuple of tuples next_cell: Position ((y, x), (y_minotaur, x_minotaur))
            on the maze that agent and minotaur transitions to.
        """
        # Compute the future position given current (state, action, action_minotaur)
        current_cell = self.states[state];
        
        # get the rows and columns of the player and minotaur
        row_player = current_cell[0][0] + self.actions_player[action][0];
        col_player = current_cell[0][1] + self.actions_player[action][1];
        row_minotaur = current_cell[1][0] + self.actions_minotaur[action_minotaur][0];
        col_minotaur = current_cell[1][1] + self.actions_minotaur[action_minotaur][1];
        
        next_cell = ((row_player, col_player), (row_minotaur, col_minotaur));
        
        # Is the future position an impossible one ?
        player_hitting_maze_walls =  (row_player == -1) or (row_player == self.maze.shape[0]) or \
                              (col_player == -1) or (col_player == self.maze.shape[1]) or \
                              (self.maze[row_player, col_player] == 1);
                              
        # Based on the impossiblity check return the next state.
        if player_hitting_maze_walls:
            # The player hit a wall, stay in place, but the minotaur can still move
            return self.map[(current_cell[0], next_cell[1])];
        else:
            # Both player and minotaur can move
            return self.map[next_cell];
        
    def __minotaur_transition(self):
        # For each state and action we compute the probability of the minotaur moving
        dimensions = (self.n_states, self.n_actions, self.n_actions_minotaur);
        minotaur_probabilities = np.zeros(dimensions);
        
        # For each state and action and minotaur action
        for s in range(self.n_states):
            for a in range(self.n_actions):
                # First look at the possible moves of the minotaur
                possibilities = self.__get_minotaur_move_possibilities(self.states[s][1]);
                
                # For each possible move of the minotaur, compute the next state
                for a_m in range(self.n_actions_minotaur):
                    # If the move is in the possibilities then the probability is shared
                    if a_m in possibilities:
                        minotaur_probabilities[s, a, a_m] = 1 / len(possibilities);
                    else:
                        minotaur_probabilities[s, a, a_m] = 0;
        return minotaur_probabilities;
                 
    def __transitions(self):
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        # Initialize the transition probailities tensor (S,S,A)
        dimensions = (self.n_states, self.n_states, self.n_actions);
        transition_probabilities = np.zeros(dimensions);

        # Compute the transition probabilities. Note that the transitions
        # are non-deterministic.
        for s in range(self.n_states):
            for a in range(self.n_actions):
                # For each each possible minotaur move 
                for a_m in range(self.n_actions_minotaur):
                    if self.minotaur_probabilities[s, a, a_m] > 0:
                        # Compute the next state given the current state and action
                        next_s = self.__move(s,a,a_m);
                        
                        # The transition probability is the probability of the minotaur moving
                        transition_probabilities[s, next_s, a] = self.minotaur_probabilities[s, a, a_m];

        return transition_probabilities;
    
    def __rewards(self, weights=None, random_rewards=None):
        rewards = np.zeros((self.n_states, self.n_actions));

        # If the rewards are not described by a weight matrix
        if weights is None:
            # For each state and action the reward is computed
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    # Sum for the reward using a weighted average
                    reward_sum = 0;
                    
                    for a_m in self.actions_minotaur:
                        if self.minotaur_probabilities[s, a, a_m] > 0:
                            # Get the probability of this transaction
                            probability = self.minotaur_probabilities[s, a, a_m];
                            
                            # Next state
                            current_s = self.states[s];
                            next_s = self.states[self.__move(s, a, a_m)]
                            
                            # If the minotaur catches the player
                            if next_s[0] == next_s[1]:
                                reward_sum += probability * self.CAUGHT_MINOTAUR;
                                
                            # If the player hits a wall
                            elif next_s[0] == current_s[0] and a != self.STAY:
                                reward_sum += probability * self.IMPOSSIBLE_REWARD;
                                
                            # If the player reaches the exit
                            elif self.maze[next_s[0]] == 2:
                                reward_sum += probability * self.GOAL_REWARD;
                                
                            # If the player takes a step to an empty cell that is not the exit
                            else:
                                reward_sum += probability * self.STEP_REWARD;
                            
                    # Set the reward for this state and action
                    rewards[s,a] = reward_sum;
        return rewards;
    
    def simulate(self, start_position, minature_pos, policy, method):
        if method not in methods:
            error = 'ERROR: the argument method must be in {}'.format(methods);
            raise NameError(error);
        
        # Create a starting state
        start = (start_position, minature_pos);
        
        path = list();
        if method == 'DynProg':
            # Deduce the horizon from the policy shape
            horizon = policy.shape[1];
            # Initialize current state and time
            t = 0;
            s = self.map[start];
            # Add the starting position in the maze to the path
            path.append(start);
            while t < horizon-1:
                # Move to next state given the policy and the current state
                next_s = self.__move_no_minautar(s,policy[s, t]);
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                # Update time and state for next iteration
                t += 1;
                s = next_s;
        return path;

    def __get_start_minautar(self):
        """Look for the starting position of the minautar by looking at the map
        """
        
        # If self.maze does not contain the number 2 give an error
        if not np.any(self.maze == 2):
            print("The maze does not contain a starting position for the minotaur")
            
        # Otherwise find the coordinates
        else:
            # Get the coordinates of the minotaur
            coordinates = np.argwhere(self.maze == 2)[0];
            
            # Return the tuple
            return (coordinates[0], coordinates[1]);

    def __move_no_minautar(self, state, action):
        """ Make a step in the maze without the minotaur"""
        # Get the tuples of the current state
        current_cell = self.states[state];

        # Compute the future position given current (state, action)
        row = current_cell[0][0] + self.actions_player[action][0];
        col = current_cell[0][1] + self.actions_player[action][1];

        # Is the future position an impossible one ?
        hitting_maze_walls =  (row == -1) or (row == self.maze.shape[0]) or \
                              (col == -1) or (col == self.maze.shape[1]) or \
                              (self.maze[row,col] == 1);
        
        # Update the state
        if not hitting_maze_walls:
            return self.map[((row, col), current_cell[1])];
        else:
            return state;

def print_maze_directions(directions):
    # Print the directions array using arrows
    for x in range(directions.shape[0]):
        for y in range(directions.shape[1]):
            print(" %s" % directions[x,y], end="")
        print();
    print();

def draw_actions_position_minautar(env, policy, maze, minotaur_pos, t):
    # For each cell in the maze state that is not a wall, draw an arrow
    # corresponding to the action that would be taken when moving to that cell.

    # A variable containing the directions
    directions = np.zeros((maze.shape[0], maze.shape[1]), dtype=str);

    # For each cell in the maze
    for x in range(maze.shape[0]):
        for y in range(maze.shape[1]):
            # If the cell is not a wall
            if maze[x,y] != 1:
                # Get the index of this state
                state = env.map[((x,y), minotaur_pos)];
                # Get the action taken in this cell
                action = policy[state, t];
                # Depending on the action assign a direction
                if action == 0:
                    directions[x,y] = "s";
                elif action == 1:
                    directions[x,y] = "<";
                elif action == 2:
                    directions[x,y] = ">";
                elif action == 3:
                    directions[x,y] = "^";
                elif action == 4:
                    directions[x,y] = "v";
            else:
                directions[x,y] = "X";
    
    # Print an O  at the position of the minotaur
    directions[minotaur_pos[0], minotaur_pos[1]] = "O";

    # Print this array using a plot function
    print_maze_directions(directions);

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
    p         = env.transition_probabilities;
    r         = env.rewards;
    n_states  = env.n_states;
    n_actions = env.n_actions;
    T         = horizon;

    # The variables involved in the dynamic programming backwards recursions
    V      = np.zeros((n_states, T+1));
    policy = np.zeros((n_states, T+1));
    Q      = np.zeros((n_states, n_actions));

    # Initialization
    Q            = np.copy(r);
    V[:, T]      = np.max(Q,1);
    policy[:, T] = np.argmax(Q,1);

    # The dynamic programming bakwards recursion
    for t in range(T-1,-1,-1):
        # Update the value function acccording to the bellman equation
        for s in range(n_states):
            for a in range(n_actions):
                # Update of the temporary Q values
                Q[s, a] = r[s, a] + np.dot(p[:, s, a],V[:, t+1])
        # Update by taking the maximum Q value w.r.t the action a
        V[:, t] = np.max(Q, 1);
        # The optimal action is the one that maximizes the Q function
        policy[:, t] = np.argmax(Q, 1);
        print(policy[:, t])
    return V, policy;

def draw_maze(maze):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED};

    # Give a color to each cell
    rows,cols    = maze.shape;
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows));

    # Remove the axis ticks and add title title
    ax = plt.gca();
    ax.set_title('The Maze');
    ax.set_xticks([]);
    ax.set_yticks([]);

    # Give a color to each cell
    rows,cols    = maze.shape;
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                            cellColours=colored_maze,
                            cellLoc='center',
                            loc=(0,0),
                            edges='closed');
    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows);
        cell.set_width(1.0/cols);

def animate_solution(maze, path, min_pos):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED};

    # Size of the maze
    rows,cols = maze.shape;

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows));

    # Remove the axis ticks and add title title
    ax = plt.gca();
    ax.set_title('Policy simulation');
    ax.set_xticks([]);
    ax.set_yticks([]);

    # Give a color to each cell
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Color the minotaur cell red
    colored_maze[min_pos[0]][min_pos[1]] = LIGHT_RED;

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0,0),
                     edges='closed');

    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows);
        cell.set_width(1.0/cols);


    # Update the color at each frame
    for i in range(len(path)):
        grid.get_celld()[(path[i][0])].set_facecolor(LIGHT_ORANGE)
        grid.get_celld()[(path[i][0])].get_text().set_text('Player')
        if i > 0:
            if path[i][0] == path[i-1]:
                grid.get_celld()[(path[i][0])].set_facecolor(LIGHT_GREEN)
                grid.get_celld()[(path[i][0])].get_text().set_text('Player is out')
            else:
                grid.get_celld()[(path[i-1][0])].set_facecolor(col_map[maze[path[i-1][0]]])
                grid.get_celld()[(path[i-1][0])].get_text().set_text('')
        display.display(fig)
        display.clear_output(wait=True)
        time.sleep(1)


# maze = np.array([
#     [0, 0, 1, 0, 0, 0, 0],
#     [0, 0, 1, 0, 0, 0, 0],
#     [0, 0, 1, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0],
#     [1, 1, 1, 1, 1, 1, 0],
#     [0, 0, 0, 0, 0, 2, 0]
# ])

# env = Maze(maze)
# draw_maze(maze);
# # Finite horizon
# horizon = 10
# # Solve the MDP problem with dynamic programming 
# V, policy=  dynamic_programming(env,horizon);

# # # Simulate the shortest path starting from position A
# # method = 'DynProg';
# # start  = (0,0);
# # path = env.simulate(start, policy, method); 