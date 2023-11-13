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

    def __init__(self):
        super(SecondAgent, self).__init__()
        self.name = "SecondAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.max_depth = 6

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
        start_time = time.time()
        _, action = self.minimax(chess_board, my_pos, adv_pos, max_step, self.max_depth, float('-inf'), float('inf'), True)
        time_taken = time.time() - start_time
        
        
        print("My AI's turn took ", time_taken, "seconds.")

        # dummy return
        return my_pos, action
    
    def evaluate_board(self, chess_board, my_pos, adv_pos, max_step):
        my_area_size = self.calculate_area_size(chess_board, my_pos, max_step)
        adv_area_size = self.calculate_area_size(chess_board, adv_pos, max_step)

        # The goal is to trap the opponent in a smaller area, so we want to maximize the difference
        evaluation = my_area_size - adv_area_size

        return evaluation
    
    def calculate_area_size(self, chess_board, start_pos, max_step):
        visited = set()
        stack = [start_pos]
        area_size = 0

        while stack > 0 & (max_step > 0):
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

    def is_valid_position(self, chess_board, pos):
        x, y = pos
        x_max, y_max, _ = chess_board.shape
        return 0 <= x < x_max and 0 <= y < y_max

    def is_terminal_node(self, depth):
        # Add your own conditions to check if it's a terminal node
        return depth == 0

    def minimax(self, chess_board, my_pos, adv_pos, max_step, depth, alpha, beta, maximizing_player):
        if self.is_terminal_node(depth):
            return self.evaluate_board(chess_board, my_pos, adv_pos, max_step), None

        valid_actions = ["u", "r", "d", "l"]

        if maximizing_player:
            max_eval = float('-inf')
            best_action = None
            for action in valid_actions:
                new_pos = self.get_new_position(my_pos, action)
                new_board = self.simulate_move(chess_board, my_pos, new_pos, action)
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
                new_pos = self.get_new_position(adv_pos, action)
                new_board = self.simulate_move(chess_board, adv_pos, new_pos, action)
                eval, _ = self.minimax(new_board, my_pos, new_pos, max_step, depth - 1, alpha, beta, True)
                if eval < min_eval:
                    min_eval = eval
                    best_action = action
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval, best_action

    def get_new_position(self, pos, action):
        x, y = pos
        if action == "u":
            return x - 1, y
        elif action == "r":
            return x, y + 1
        elif action == "d":
            return x + 1, y
        elif action == "l":
            return x, y - 1

    def simulate_move(self, chess_board, start_pos, end_pos, action):
        # Add your own logic to simulate the move on the chess board
        # This may include updating the positions of both the bot and the opponent
        return chess_board.copy()
