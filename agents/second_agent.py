# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time


@register_agent("second_agent")
class SecondAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """
    
    global BOARD_SIZE

    def __init__(self):
        super(SecondAgent, self).__init__()
        self.name = "SecondAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.max_depth = 3

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """
        
        # Some simple code to help you with timing. Consider checking 
        # time_taken during your search and breaking with the best answer
        # so far when it nears 2 seconds.
        BOARD_SIZE = 8
        start_time = time.time()
        _, action = self.minimax(chess_board, my_pos, adv_pos, max_step, self.max_depth, float('-inf'), float('inf'), True)
        time_taken = time.time() - start_time
        
        
        print("My AI's turn took ", time_taken, "seconds.")

        # dummy return
        return my_pos, action
    
    #check all possible in a radius of max-depth
    def evaluate_board(self, chess_board, my_pos, adv_pos, max_step):
        my_area_size = self.calculate_area_size_reachable(chess_board, my_pos,max_step)
        adv_area_size = self.calculate_area_size_reachable(chess_board, adv_pos, max_step)

        return my_area_size - adv_area_size
    
    # sum of going only right, then only left then only up then only down
    def calculate_line_size(self, chess_board, start_pos):
        line_size = 0
        for direction in ["u", "r", "d", "l"]:
            new_pos = start_pos
            while self.valid_action(new_pos, direction, chess_board):
                new_pos = self.get_new_position(start_pos, direction)
                line_size += 1
        
        return line_size
        
    
    def calculate_area_size_reachable(self, chess_board, start_pos, max_step):
        visited = set()
        stack = [start_pos]
        area_size = 0

        while stack and max_step > 0:
            max_step -= 1
            current_pos = stack.pop()
            if current_pos in visited:
                continue

            visited.add(current_pos)
            area_size += 1

            x, y = current_pos
            for direction in ["u", "r", "d", "l"]:
                new_x, new_y = self.get_new_position(current_pos, direction)
                if self.is_valid_position(chess_board, (new_x, new_y)) and (new_x, new_y) not in visited:
                    stack.append((new_x, new_y))

        return area_size
    
    def calculate_area_size(self, chess_board, start_pos):
        visited = set()
        stack = [start_pos]
        area_size = 0

        while stack:
            current_pos = stack.pop()
            if current_pos in visited:
                continue

            visited.add(current_pos)
            area_size += 1

            x, y = current_pos
            for direction in ["u", "r", "d", "l"]:
                new_x, new_y = self.get_new_position(current_pos, direction)
                if self.is_valid_position(chess_board, (new_x, new_y)) and (new_x, new_y) not in visited:
                    stack.append((new_x, new_y))

        return area_size

    def is_valid_position(self, chess_board, pos):
        x, y = pos
        x_max, y_max, _ = chess_board.shape
        return 0 <= x < x_max and 0 <= y < y_max
    
    
    
    # check whether the action is valid
    def valid_action(self, pos, action, chess_board):
        (x, y) = pos
        if action == "u":
            return x > 0 and not chess_board[x, y, 0]
        elif action == "r":
            return y < chess_board.shape[1] and not chess_board[x, y, 1]
        elif action == "d":
            return x < chess_board.shape[0] and not chess_board[x, y, 2]
        elif action == "l":
            return y > 0 and not chess_board[x, y, 3]
        else:
            return False
        

    def is_terminal_node(self, depth):
        # Add your own conditions to check if it's a terminal node
        return depth == 0
    
    # current: for each neighbouring tile
    #               for each wall
    #                   evaluate move
    #                       simulate move

    # optimal: for each evaluated reachable tile
    #           for each action
    #               for each wall
    #                   evaluate move
    #                       simulate move

    # board = chess_board.copy()
    # board[my_pos] = how good this position is

    def minimax(self, chess_board, my_pos, adv_pos, max_step, depth, alpha, beta, maximizing_player):
        if self.is_terminal_node(depth):
            return self.evaluate_board(chess_board, my_pos, adv_pos, max_step), None

        valid_actions = ["u", "r", "d", "l"]

        if maximizing_player:
            max_eval = float('-inf')
            best_action = None
            for action in valid_actions:
                if(self.valid_action(my_pos, action, chess_board)):
                    new_pos = self.get_new_position(my_pos, action)
                    for wall in valid_actions:
                        new_board = self.simulate_move(chess_board, new_pos, wall)
                        eval, _ = self.minimax(new_board, new_pos, adv_pos, max_step, depth - 1, alpha, beta, False)
                        if eval > max_eval:
                            max_eval = eval
                            best_action = action
                        alpha = max(alpha, eval)
                        if beta <= alpha:
                            break
            return max_eval, best_action
        else:
            min_eval = float('inf')
            best_action = None
            for action in valid_actions:
                if(self.valid_action(adv_pos, action, chess_board)):
                    new_pos = self.get_new_position(adv_pos, action)
                    for wall in valid_actions:
                        new_board = self.simulate_move(chess_board, new_pos, wall)
                        eval, _ = self.minimax(new_board, my_pos, new_pos, max_step, depth - 1, alpha, beta, True)
                        if eval < min_eval:
                            min_eval = eval
                            best_action = action
                        beta = min(beta, eval)
                        if beta <= alpha:
                            break
            return min_eval, best_action

    def get_new_position(self, pos, action):
        if pos != None:
            (x ,y) = pos
            if action == "u":
                return (x - 1, y)
            elif action == "r":
                return (x, y + 1)
            elif action == "d":
                return (x + 1, y)
            elif action == "l":
                return (x, y - 1)
        else:
            print("pos is none")
    
    def check_valid_step(self, start_pos, end_pos, barrier_dir):
        """
        Check if the step the agent takes is valid (reachable and within max steps).

        Parameters
        ----------
        start_pos : tuple
            The start position of the agent.
        end_pos : np.ndarray
            The end position of the agent.
        barrier_dir : int
            The direction of the barrier.
        """
        # Endpoint already has barrier or is border
        r, c = end_pos
        if self.chess_board[r, c, barrier_dir]:
            return False
        if np.array_equal(start_pos, end_pos):
            return True

        # Get position of the adversary
        adv_pos = self.p0_pos if self.turn else self.p1_pos

        # BFS
        state_queue = [(start_pos, 0)]
        visited = {tuple(start_pos)}
        is_reached = False
        while state_queue and not is_reached:
            cur_pos, cur_step = state_queue.pop(0)
            r, c = cur_pos
            if cur_step == self.max_step:
                break
            for dir, move in enumerate(self.moves):
                if self.chess_board[r, c, dir]:
                    continue

                next_pos = cur_pos + move
                if np.array_equal(next_pos, adv_pos) or tuple(next_pos) in visited:
                    continue
                if np.array_equal(next_pos, end_pos):
                    is_reached = True
                    break

                visited.add(tuple(next_pos))
                state_queue.append((next_pos, cur_step + 1))

        return is_reached
            
    def simulate_move(self, chess_board, end_pos, action):
        x,y = end_pos
        new_board = chess_board.copy()
        new_board[x,y, self.dir_map[action]] = True
        

        return new_board
