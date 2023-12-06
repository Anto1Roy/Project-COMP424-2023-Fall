# Student agent: Add your own agent here
from queue import PriorityQueue
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from collections import deque
import numpy as np


### Strategy meilleur ici que student agent
@register_agent("second_agent")
class SecondAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    global BOARD_SIZE
    global counter

    def __init__(self):
        super(SecondAgent, self).__init__()
        self.name = "SecondAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.max_depth = 4

    def step(self, chess_board, my_pos, adv_pos, max_step):
        global counter
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
        self.max_step = max_step
        self.size = chess_board.shape[0] - 1
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        counter = 0
        # Some simple code to help you with timing. Consider checking
        # time_taken during your search and breaking with the best answer
        # so far when it nears 2 seconds.
        start_time = time.time()
        # print(MaxFlow(chess_board).ford_fulkerson(my_pos, adv_pos))
        _, new_pos, wall = self.gyuminimax(
            chess_board,
            my_pos,
            adv_pos,
            max_step,
            self.max_depth,
            float("-inf"),
            float("inf"),
            True,
            start_time
        )
        time_taken = time.time() - start_time

        # new_pos = self.get_new_position(my_pos, new_pos)

        # print("My AI's turn took ", time_taken, "seconds.")
        # print(f"position , move :  {new_pos} , {self.dir_map[wall]}")

        if new_pos == None or wall == None:
            return self.random_move(chess_board, my_pos, adv_pos, max_step)

        return new_pos, self.dir_map[wall]

    # check all possible in a radius of max-depth
    def evaluate_board(self, chess_board, my_pos, adv_pos, max_step):
        my_area_size = self.calculate_area_size_reachable(chess_board, my_pos, max_step)
        adv_area_size = self.calculate_area_size_reachable(chess_board, adv_pos, max_step)

        return my_area_size - adv_area_size
    
    def evaluate_board_finish(self, chess_board, my_pos, adv_pos):
        my_area_size = self.calculate_area_size(chess_board, my_pos)  # ,max_step)
        adv_area_size = self.calculate_area_size(chess_board, adv_pos)  # , max_step)

        return my_area_size - adv_area_size

    # sum of going only right, then only left then only up then only down
    def calculate_line_size(self, chess_board, start_pos):
        line_size = 0
        for direction in ["u", "r", "d", "l"]:
            new_pos = start_pos
            while self.valid_action(new_pos, direction, chess_board):
                new_pos = self.get_new_position(new_pos, direction)
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

            for direction in ["u", "r", "d", "l"]:
                new_x, new_y = self.get_new_position(current_pos, direction)
                if (
                    self.is_valid_position(chess_board, (new_x, new_y))
                    and (new_x, new_y) not in visited
                ):
                    stack.append((new_x, new_y))

        return area_size

    def random_move(self, chess_board, my_pos, adv_pos, max_step):
        for i in self.iterate_positions_around(my_pos[0], my_pos[1], max_step):
            for j in self.dir_map.keys():
                if self.check_valid_step(my_pos, i, adv_pos, j, chess_board):
                    return i, self.dir_map[j]

    ## deque provides faster pop operations
    # with numpy shit
    def calculate_area_size(self, chess_board, start_pos, limit=50):
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

    ## op np maniere de compter les walls
    def count_walls(self, chess_board):
        return np.sum(chess_board)

    def calculate_area_size_og(self, chess_board, start_pos, limit=50):
        visited = set()
        stack = deque([start_pos])
        area_size = 0
        while stack and limit > 0:
            limit -= 1
            current_pos = stack.pop()
            if current_pos in visited:
                continue

            visited.add(current_pos)
            area_size += 1

            for direction in ["u", "r", "d", "l"]:
                new_x, new_y = self.get_new_position(current_pos, direction)
                if (
                    self.is_valid_position(chess_board, (new_x, new_y))
                    and (new_x, new_y) not in visited
                ):
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
            return y < self.size and not chess_board[x, y, 1]
        elif action == "d":
            return x < self.size and not chess_board[x, y, 2]
        elif action == "l":
            return y > 0 and not chess_board[x, y, 3]
        else:
            return False

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
        current = time.time()
        positions = []
        for pos in self.iterate_positions_around(my_pos[0], my_pos[1], max_step):
            if self.check_valid_move(my_pos, pos, adv_pos, chess_board, True):
                positions.append(self.evaluate_position(chess_board, pos, adv_pos))
        positions.sort(key=lambda x: (x[0], x[1]))

        # print("sort_positions took ", time.time() - current, "seconds.")

        return list(map(lambda c: c[2], positions[:10]))

    def is_terminal_node(self, depth, start_time, my_pos, adv_pos, chess_board):
        # Add your own conditions to check if it's a terminal node
        return 1 if depth == 0 or time.time() - start_time > 1.9 else (0 if self.check_valid_move(my_pos, adv_pos, adv_pos, chess_board, False) else -1)

    # if score is 1 vs 98 it biases the algorithm to go towards the 98 instead of going to the action with the most victories in total
    # live
    # being between the middle and the other agent is important
    # how to check for max flow in end game
    # a star path from my pos to adv pos

    def gyuminimax(
        self,
        chess_board,
        my_pos,
        adv_pos,
        max_step,
        depth,
        alpha,
        beta,
        maximizing_player,
        start_time,
    ):
        global counter
        counter += 1
        result = self.is_terminal_node(depth, start_time, my_pos, adv_pos, chess_board)
        if result == 1:
            return self.evaluate_board(chess_board, my_pos, adv_pos, max_step), None, None 
        elif result == 0:
            5 * self.evaluate_board_finish(chess_board, my_pos, adv_pos), None, None
        
        # does this make sense?
        current_eval = self.evaluate_board(chess_board, my_pos, adv_pos, max_step) if maximizing_player else self.evaluate_board(chess_board, adv_pos, my_pos, max_step)

        if maximizing_player:
            max_eval = float("-inf")
            best_action = None, None
            for move in self.sort_positions(chess_board, my_pos, adv_pos, max_step):
                for wall in self.dir_map.keys():
                    if not self.check_valid_step(
                        my_pos, move, adv_pos, wall, chess_board
                    ):
                        continue
                    new_board = self.simulate_move(chess_board, move, wall)
                    eval, _, _ = self.gyuminimax(
                        new_board,
                        move,
                        adv_pos,
                        max_step,
                        depth - 1,
                        alpha,
                        beta,
                        False,
                        start_time
                    )
                    # does this make sense?
                    eval += current_eval
                    if eval > max_eval:
                        max_eval = eval
                        best_action = (move, wall)
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
            action, wall = best_action
            return max_eval, action, wall
        else:
            min_eval = float("inf")
            best_action = None, None
            for move in self.sort_positions(chess_board, adv_pos, my_pos, max_step):
                for wall in self.dir_map.keys():
                    if not self.check_valid_step(
                        my_pos, move, adv_pos, wall, chess_board
                    ):
                        continue
                    new_board = self.simulate_move(chess_board, move, wall)
                    eval, _, _ = self.gyuminimax(
                        new_board, my_pos, move, max_step, depth - 1, alpha, beta, True, start_time
                    )
                    # does this make sense?
                    eval += current_eval
                    if eval < min_eval:
                        min_eval = eval
                        best_action = (move, wall)
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break

            action, wall = best_action
            return min_eval, action, wall

    def get_new_position(self, pos, action):
        if pos != None:
            (x, y) = pos
            if action == "u":
                return (x - 1, y)
            elif action == "r":
                return (x, y + 1)
            elif action == "d":
                return (x + 1, y)
            elif action == "l":
                return (x, y - 1)
            

    def check_valid_move(self, start_pos, end_pos, adv_pos, chess_board, limit):
        
        if np.array_equal(start_pos, end_pos):
            return True

        # A* algorithm
        state_queue = PriorityQueue()
        state_queue.put((0, start_pos, 0))  # (priority, position, cost)
        visited = {tuple(start_pos)}
        is_reached = False

        while not state_queue.empty() and not is_reached:
            _, cur_pos, cur_cost = state_queue.get()
            r, c = cur_pos

            if cur_cost == self.max_step and limit:
                break

            for dir, move in enumerate(self.moves):
                if chess_board[r, c, dir]:
                    continue

                next_pos = (cur_pos[0] + move[0], cur_pos[1] + move[1])

                if (np.array_equal(next_pos, adv_pos) and limit) or tuple(next_pos) in visited:
                    continue

                if np.array_equal(next_pos, end_pos):
                    is_reached = True
                    break

                visited.add(tuple(next_pos))
                priority = cur_cost + 1 + self.heuristic(next_pos, adv_pos)
                state_queue.put((priority, next_pos, cur_cost + 1))

        return is_reached

    def heuristic(self, pos, adv_pos):

        return abs(pos[0] - adv_pos[0]) + abs(pos[1] - adv_pos[1])
    
    def check_valid_step(self, start_pos, end_pos, adv_pos, barrier_dir, chess_board):
          # Endpoint already has barrier or is border
        r, c = end_pos
        if chess_board[r, c, self.dir_map[barrier_dir]]:
            return False
        if np.array_equal(start_pos, end_pos):
            return True

        # A* algorithm
        state_queue = PriorityQueue()
        state_queue.put((0, start_pos, 0))  # (priority, position, cost)
        visited = {tuple(start_pos)}
        is_reached = False

        while not state_queue.empty() and not is_reached:
            _, cur_pos, cur_cost = state_queue.get()
            r, c = cur_pos

            if cur_cost == self.max_step:
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
                priority = cur_cost + 1 + self.heuristic(next_pos, adv_pos)
                state_queue.put((priority, next_pos, cur_cost + 1))

        return is_reached

    def simulate_move(self, chess_board, end_pos, action):
        x, y = end_pos
        new_board = chess_board.copy()
        new_board[x, y, self.dir_map[action]] = True

        return new_board

class MaxFlow:
    def __init__(self, graph):
        self.graph = graph
        self.n = len(graph)
        self.m = len(graph[0])

    def ford_fulkerson(self, source, sink):
        parent = [[(-1, -1) for _ in range(self.m)] for _ in range(self.n)]
        max_flow = 0

        while True:
            # Find augmenting path using BFS
            path_flow = float('inf')
            visited = [[False] * self.m for _ in range(self.n)]
            queue = deque([(source, source)])
            visited[source[0]][source[1]] = True

            while queue:
                u, prev = queue.popleft()
                for v in self.get_adjacent_edges(u):
                    if not visited[v[0]][v[1]]:
                        queue.append((v, u))
                        visited[v[0]][v[1]] = True
                        parent[v[0]][v[1]] = (u, prev)
                        path_flow = 1  # Capacity is always 1

            if not visited[sink[0]][sink[1]]:
                break  # No more augmenting paths

            # Update residual capacities along the path
            max_flow += path_flow
            current = sink
            while current != source:
                u, prev = parent[current[0]][current[1]]
                self.update_capacity(u, current)
                current = u

        return max_flow

    def min_cut(self, source):
        # Perform DFS to find nodes reachable from source after max flow
        visited = [[False] * self.m for _ in range(self.n)]
        stack = [source]
        visited[source[0]][source[1]] = True

        while stack:
            u = stack.pop()
            for v in self.get_adjacent_edges(u):
                if not visited[v[0]][v[1]]:
                    stack.append(v)
                    visited[v[0]][v[1]] = True

        # Return the set of nodes reachable from the source
        return {(i, j) for i in range(self.n) for j in range(self.m) if visited[i][j]}

    def get_adjacent_edges(self, u):
        i, j = u
        edges = []
        if i > 0 and not self.graph[i-1, j, 2]:  # Up
            edges.append((i-1, j))
        if i < self.n - 1 and not self.graph[i, j, 0]:  # Down
            edges.append((i+1, j))
        if j > 0 and not self.graph[i, j-1, 1]:  # Left
            edges.append((i, j-1))
        if j < self.m - 1 and not self.graph[i, j, 3]:  # Right
            edges.append((i, j+1))
        return edges

    def update_capacity(self, u, v):
        i, j = u
        if u[0] == v[0] - 1:  # Move Up
            self.graph[v[0], v[1], 0] = True
        elif u[0] == v[0] + 1:  # Move Down
            self.graph[u[0], u[1], 0] = True
        elif u[1] == v[1] - 1:  # Move Left
            self.graph[v[0], v[1], 3] = True
        elif u[1] == v[1] + 1:  # Move Right
            self.graph[u[0], u[1], 1] = True
