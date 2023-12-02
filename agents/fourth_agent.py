# Student agent: Add your own agent here
import math
import random
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from collections import deque
import numpy as np


### Strategy meilleur ici que student agent
@register_agent("fourth_agent")
class FourthAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    global BOARD_SIZE

    def __init__(self):
        super(FourthAgent, self).__init__()
        self.name = "FourthAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.futures_game_states = []

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
        state = self.get_state(chess_board, my_pos, adv_pos, max_step)
        state.max_step = max_step
        root_board = Board(state, True)
        self.size = chess_board.shape[0] - 1
        # Some simple code to help you with timing. Consider checking
        # time_taken during your search and breaking with the best answer
        # so far when it nears 2 seconds.
        start_time = time.time()
        # we get around 50 visits, where should we search?
        while (time.time() - start_time) < 1.9:
            selected_node = root_board.select()
            expanded_node = root_board.expand(selected_node)
            simulation_result = root_board.simulate(expanded_node)
            root_board.backpropagate(expanded_node, simulation_result)
            #print("time : " + str(time.time() - start_time) + " visits : " + str(root_board.visits))
        time_taken = time.time() - start_time

        best_child_node = max(root_board.children, key=lambda x: x.visits)
        self.futures_game_states = []
        self.futures_game_states.append(best_child_node.children)
        pos, dir = best_child_node.state.last_action

        return pos, self.dir_map[dir]
    
    def get_state(self, chess_board, my_pos, adv_pos, max_step):
        # for boards in self.futures_game_states:
        #     for board in boards:
        #         if board.state.my_pos == my_pos and board.state.adv_pos == adv_pos:
        #             board.state.max_step = max_step
        #             board.state.maximization = True
        #             return board.state
        return GameState(chess_board, my_pos, adv_pos, max_step, True, 8)
    
class Board:
    def __init__(self, game_state, maximizing, parent=None):
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        self.state = game_state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.score = 0
        self.maximizing = maximizing
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
    
    def select(self):
        while not self.state.is_terminal():
            # this shoulds be a probability since we dont want to explore everything
            if (len(self.children) <= len(self.state.posible_moves) and random.random() < 0.5) or len(self.children) == 0:
                return self.expand(self)
            else:
                return self.best_child(self)
    
    def is_terminal(self):
        return self.state.is_terminal()

    def expand(self, node):
        move = node.state.next_action()
        if move:
            for wall in self.dir_map.keys():
               if not (self.state.check_valid_step(
                        self.state.my_pos, move, self.state.adv_pos, wall, self.state.current_board
                    ) if self.maximizing else self.state.check_valid_step(
                        self.state.adv_pos, move, self.state.mypos, wall, self.state.current_board
                    )):
                continue 
            new_state = node.state.perform_action(move, wall)
            new_state.last_action = (move, wall)
            new_node = Board(new_state, not node.maximizing, parent=node)
            node.children.append(new_node)
            return new_node
        else:
            return self.best_child(node)

    def best_child(self, node):
        exploration_weight = 1.4  # Adjust this parameter
        children_with_scores = [(child, child.state.get_score() / (child.visits + 1e-6) + exploration_weight * math.sqrt(math.log(node.visits + 1) / (child.visits + 1e-6))) for child in node.children]
        return max(children_with_scores, key=lambda x: x[1])[0] if len(children_with_scores) > 0 else node

    def simulate(self, node):
        # copy current state
        current_state = deepcopy(node.state)
        while not current_state.is_terminal():
            move = current_state.next_action()
            # if no more moves
            if not move:
                return current_state.get_score()
            # as long as there exists a move
            for wall in self.dir_map.keys():
                if not (self.state.check_valid_step(
                        current_state.my_pos, move, current_state.adv_pos, wall, current_state.current_board
                    ) if self.maximizing else self.state.check_valid_step(
                        current_state.adv_pos, move, current_state.my_pos, wall, current_state.current_board
                    )):
                    continue
                current_state = current_state.perform_action(move, wall)
                current_state.last_action = (move, wall)
                if current_state.is_terminal():
                    break
        return current_state.get_score()

    def backpropagate(self, node, score):
        print("backpropagating")
        while node is not None:
            node.visits += 1
            node.score += score if node.maximizing else -score
            node = node.parent

# Assume you have a GameState class with the necessary methods
class GameState:
    def __init__(self, chess_board, my_pos, adv_pos, max_step, maximizing, max_depth):
        # Initialize game state
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        self.current_board = chess_board
        self.posible_moves = self.sort_positions(chess_board, my_pos, adv_pos, max_step) # this takes 0.02 seconds
        self.current_index = 0
        self.best_action = None
        self.my_pos = my_pos
        self.adv_pos = adv_pos
        self.max_step = max_step
        self.maximizing = maximizing
        self.max_depth = max_depth
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

    def is_terminal(self):
        # Check if the game is in a terminal state
        return not self.check_valid_path(self.my_pos, self.adv_pos, self.current_board)
    
    # write a function that returns every move such that i + j <= max_step
    def iterate_positions_around(self, x, y, radius):
        positions = []
        for i in range(-radius, radius):
            for j in range(-radius, radius):
                if (
                    abs(i) + abs(j) <= radius
                    and 0 <= (x + i) <= self.size
                    and 0 <= (y + j) <= self.size
                ):
                    positions.append((x + i, y + j))
        return positions

    def evaluate_position(self, chess_board, my_pos, adv_pos):
        x, y = my_pos
        x2, y2 = adv_pos
        count = 0
        for i in chess_board[x][y]:
            if i == True:
                count += 1
        factor = abs(x2 - x) + abs(y2 - y)
        return count, factor, my_pos

    def sort_positions(self, chess_board, my_pos, adv_pos, max_step):
        positions = []
        current = time.time()
        self.size = chess_board.shape[0] - 1
        for pos in self.iterate_positions_around(my_pos[0], my_pos[1], max_step):
            self.max_step = max_step
            if self.check_valid_move(my_pos, pos, adv_pos, chess_board, True):
                positions.append(self.evaluate_position(chess_board, pos, adv_pos))
        positions.sort(key=lambda x: (x[0], x[1]))

        return list(map(lambda c: c[2], positions))
    
    def next_actions(self):
        self.current_index += 5
        return [position for position in self.posible_moves if self.posible_moves.index(position) >= self.current_index - 5 and self.posible_moves.index(position) < self.current_index]

    def next_action(self):
        self.current_index += 1
        return self.posible_moves[self.current_index - 1] if self.current_index <= len(self.posible_moves) else None
    
    def perform_action(self, move, action):
        x, y = move
        self.my_pos = move
        new_board = self.current_board.copy()
        new_board[x, y, self.dir_map[action]] = True
        return GameState(new_board, move, self.adv_pos, self.max_step, not self.maximizing, self.max_depth-1) if self.maximizing else GameState(new_board, self.my_pos, move, self.max_step, not self.maximizing, self.max_depth-1)

    def clone(self):
        # Create a copy of the current state
        return 0

    def get_score(self):
        current = time.time()
        my_area_size = self.calculate_area_size(self.current_board, self.my_pos)  # ,max_step)
        adv_area_size = self.calculate_area_size(self.current_board, self.adv_pos)  # , max_step)
        return my_area_size - adv_area_size
    
    # with numpy shit
    def calculate_area_size(self, chess_board, start_pos, limit=200):
        visited = np.zeros_like(chess_board[..., 0], dtype=bool)
        stack = deque([start_pos])
        area_size = 0
        while stack and limit > 0:
            limit -= 1
            current_pos = stack.pop()
            x, y = current_pos
            if visited[x, y]:
                continue

            visited[x, y] = True
            area_size += 1

            for dx, dy in self.moves:
                new_x, new_y = x + dx, y + dy
                if (
                    0 <= new_x < chess_board.shape[0]
                    and 0 <= new_y < chess_board.shape[1]
                    and not visited[new_x, new_y]
                ):
                    stack.append((new_x, new_y))

        return area_size
    
    def check_valid_move(self, start_pos, end_pos, adv_pos, chess_board, limit):
        """
        Check if the step the agent takes is valid (reachable and within max steps).

        Parameters
        ----------
        start_pos : tuple
            The start position of the agent.
        end_pos : np.ndarray
            The end position of the agent.
        """
        current = time.time()
        # BFS
        state_queue = [(start_pos, 0)]
        visited = {tuple(start_pos)}
        is_reached = False
        while state_queue and not is_reached:
            cur_pos, cur_step = state_queue.pop(0)
            r, c = cur_pos
            if cur_step == self.max_step and not limit:
                break
            for dir, move in enumerate(self.moves):
                if chess_board[r, c, dir]:
                    continue
                next_pos = (cur_pos[0] + move[0], cur_pos[1] + move[1])
                if (np.array_equal(next_pos, adv_pos) or tuple(next_pos) in visited):
                    if np.array_equal(adv_pos, end_pos):
                        is_reached = True
                        break
                    continue
                if np.array_equal(next_pos, end_pos):
                    is_reached = True
                    break

                visited.add(tuple(next_pos))
                state_queue.append((next_pos, cur_step + 1))
        return is_reached
    
    # Example BFS-based implementation of check_valid_path
    def check_valid_path(self, start_pos, end_pos, chess_board):
        visited = set()
        queue = [start_pos]

        while queue:
            current_pos = queue.pop(0)
            if current_pos == end_pos:
                return True

            if current_pos in visited:
                continue

            visited.add(current_pos)

            for move in self.moves:
                next_pos = (current_pos[0] + move[0], current_pos[1] + move[1])
                if self.check_valid_move(current_pos, next_pos, None, chess_board, False):
                    queue.append(next_pos)

        return False
    
    def check_valid_step(self, start_pos, end_pos, adv_pos, barrier_dir, chess_board):
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
        current = time.time()
        # Endpoint already has barrier or is border
        r, c = end_pos
        if chess_board[r, c, self.dir_map[barrier_dir]]:
            return False
        if np.array_equal(start_pos, end_pos):
            return True

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
                if chess_board[r, c, dir]:
                    continue
                next_pos = (cur_pos[0] + move[0], cur_pos[1] + move[1])
                if np.array_equal(next_pos, adv_pos) or tuple(next_pos) in visited:
                    continue
                if np.array_equal(next_pos, end_pos):
                    is_reached = True
                    break

                visited.add(tuple(next_pos))
                state_queue.append((next_pos, cur_step + 1))
        return is_reached