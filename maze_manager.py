import heapq
import time
# import logging
import numpy as np
import random

from src.maze import Maze
from src.cell import Cell
from src.maze_viz import Visualizer
from src.solver import DepthFirstBacktracker
from src.solver import BiDirectional
from src.solver import BreadthFirst


class MazeManager(object):
    """A manager that abstracts the interaction with the library's components. The graphs, animations, maze creation,
    and solutions are all handled through the manager.

    Attributes:
        mazes (list): It is possible to have more than one maze. They are stored inside this variable.
        media_name (string): The filename for animations and images
        quiet_mode (bool): When true, information is not shown on the console
    """

    def __init__(self):
        self.mazes = []
        self.media_name = ""
        self.quiet_mode = False


    def add_maze(self, row, col, id=0):  # Maze 클래스의 생성자를 이용하여 미로 인스턴스를 생성하고 append로 리스트에 추가
        """Add a maze to the manager. We give the maze an index of
        the total number of mazes in the manager. As long as we don't
        add functionality to delete mazes from the manager, the ids will
        always be unique. Note that the id will always be greater than 0 because
        we add 1 to the length of self.mazes, which is set after the id assignment

        Args:
            row (int): The height of the maze
            col (int): The width of the maze
            id (int):  The optional unique id of the maze.

        Returns
            Maze: The newly created maze
        """

        if id is not 0:
            self.mazes.append(Maze(row, col, id))
        else:
            if len(self.mazes) < 1:
                self.mazes.append(Maze(row, col, 0))
            else:
                self.mazes.append(Maze(row, col, len(self.mazes) + 1))

        return self.mazes[-1]

    def add_existing_maze(self, maze, override=True):  # 동일 id를 가지고 있는 미로가 없다면 새로 추가 (이때, maze id는 override에 따라 정해짐)
        """Add an already existing maze to the manager.
        Note that it is assumed that the maze already has an id. If the id
        already exists, the function will fail. To assign a new, unique id to
        the maze, set the overwrite flag to true.

        Args:
            maze: The maze that will be added to the manager
            override (bool): A flag that you can set to bypass checking the id

        Returns:
            True: If the maze was added to the manager
            False: If the maze could not be added to the manager
        """

        # Check if there is a maze with the same id. If there is a conflict, return False
        if self.check_matching_id(maze.id) is None:  # 동일한 id를 갖는 미로가 없다면,
            if override:  # override(덮어쓰기)가 'True'라면, id를 새로 지정하고 후에 maze 인스턴스를 리스트에 추가. 'False'라면, 기존의 id를 그대로 사용. 
                if len(self.mazes) < 1:
                    maze.id = 0
                else:
                    maze.id = self.mazes.__len__() + 1
        else:  # 동일한 id를 갖는 미로가 있다면 바로 'False'를 반환 
            return False
        self.mazes.append(maze)
        return maze

    def get_maze(self, id):
        """Get a maze by its id.

            Args:
                id (int): The id of the desired maze

            Return:
                    Maze: Returns the maze if it was found.
                    None: If no maze was found
        """

        for maze in self.mazes:
            if maze.id == id:
                return maze
        print("Unable to locate maze")
        return None

    def get_mazes(self):
        """Get all of the mazes that the manager is holding"""
        return self.mazes

    def get_maze_count(self):
        """Gets the number of mazes that the manager is holding"""
        return self.mazes.__len__()

    def solve_maze(self, maze_id, method, neighbor_method="fancy"):
        """ Called to solve a maze by a particular method. The method
        is specified by a string. The options are
            1. DepthFirstBacktracker
            2.
            3.
        Args:
            maze_id (int): The id of the maze that will be solved
            method (string): The name of the method (see above)
            neighbor_method:

        """
        maze = self.get_maze(maze_id)
        if maze is None:
            print("Unable to locate maze. Exiting solver.")
            return None

        """DEVNOTE: When adding a new solution method, call it from here.
            Also update the list of names in the documentation above"""
        if method == "DepthFirstBacktracker":
            solver = DepthFirstBacktracker(maze, neighbor_method, self.quiet_mode)
            maze.solution_path = solver.solve()
        elif method == "BiDirectional":
            solver = BiDirectional(maze, neighbor_method, self.quiet_mode)
            maze.solution_path = solver.solve()
        elif method == "BreadthFirst":
            solver = BreadthFirst(maze, neighbor_method, self.quiet_mode)
            maze.solution_path = solver.solve()
        elif method == "A-Star":  # for a * search algorithm
            maze.optimal_solution_path, maze.solution_cost = a_star_search(maze, manhattan_distance)
        elif method == "A-Star special version":  # calculate a * search algorithm with more accurate heuristic function
            maze.optimal_solution_path, maze.solution_cost = a_star_search(maze, manhattan_distance_special_ver)
        elif method == "UCS":  # for uniform cost search algorithm
            maze.solution_path, maze.solution_cost = uniform_cost_search(maze)
        else:  # Q-learning
            #print(maze.entry_coor, " ", maze.exit_coor, "\n");
            q_table = q_learning(maze, 1000, 0.5, 0.95, 0.99)  # change parameter for experiment!
            #uniform_cost_search(maze)  # for finding optimal path & optimal cost
            print_q_table(q_table, maze)
            q_learning_path(maze, q_table)
            #filter_optimal_path(q_learning_path(maze, q_table), maze)  # find path using Q table and filter it with optimal path

    def show_maze(self, id, cell_size=1):
        """Just show the generation animation and maze"""
        vis = Visualizer(self.get_maze(id), cell_size, self.media_name)
        vis.show_maze()

    def show_generation_animation(self, id, cell_size=1):
        vis = Visualizer(self.get_maze(id), cell_size, self.media_name)
        vis.show_generation_animation()

    def show_solution(self, id, cell_size=1):
        vis = Visualizer(self.get_maze(id), cell_size, self.media_name)
        vis.show_maze_solution()

    def show_solution_animation(self, id, cell_size=1):
        """
        Shows the animation of the path that the solver took.

        Args:
            id (int): The id of the maze whose solution will be shown
            cell_size (int):
        """
        vis = Visualizer(self.get_maze(id), cell_size, self.media_name)
        vis.animate_maze_solution()

    def check_matching_id(self, id):
        """Check if the id already belongs to an existing maze

        Args:
            id (int): The id to be checked

        Returns:

        """
        return next((maze for maze in self.mazes if maze.id == id),
                    None)  # generator(리스트 컴프리헨션과 유사)를 통해 동일한 id를 갖는 첫 번째 미로만을 반환. 만약 없을 시, 두 번째 인자인 'None'을 반환.

    def set_filename(self, filename):
        """
        Sets the filename for saving animations and images
        Args:
            filename (string): The name of the file without an extension
        """

        self.media_name = filename

    def set_quiet_mode(self, enabled):
        """
        Enables/Disables the quiet mode
        Args:
            enabled (bool): True when quiet mode is on, False when it is off
        """
        self.quiet_mode = enabled


def manhattan_distance(coord1, coord2):  # 두 좌표 사이의 맨허튼 거리를 계산
    return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])


def manhattan_distance_special_ver(coord1, coord2):  # 실제 이동 비용을 감안한 맨허튼 거리 함수 (special version)
    return 1.1 * abs(coord1[0] - coord2[0]) + 0.9 * abs(coord1[1] - coord2[1])


def adj_distance(coord1, coord2):  # 두 인접한 노드 사이의 이동 비용을 계산 (수평은 0.9, 수직은 1.1 penalty)
    return 1.1 * abs(coord1[0] - coord2[0]) + 0.9 * abs(coord1[1] - coord2[1])


def a_star_search(maze, heuristic_function):  # a * 탐색 알고리즘으로 최적해 구하기
    start = maze.entry_coor
    goal = maze.exit_coor
    priority_queue = []  # 우선순위 큐 선언 (for f의 최솟값 찾기)
    heapq.heappush(priority_queue, (0 + heuristic_function(start, goal), start))  # (f, coord) 튜플 형태로 우선순위에 저장
    parent = {}  # 최적 해 경로 역추적용. 딕셔너리 자료형을 이용하여 '[a] -> b' 형태로 저장.
    cost = {start: 0.0}  # start로부터 실제로 든 비용(g 값) (so far). 딕셔너리 자료형을 이용하여 '[a] -> cost' 형태로 저장.
    visited_cells = []  # 방문한 노드들을 모두 저장.

    maze.solution_path = []  # 미로 해결 경로 초기화

    print("\nSolving the maze with a-star search...")
    time_start = time.time()  # 걸린 시간 check!

    def relaxation(a, b, tentative_cost):  # 노드 a와 b 사이에서 relaxation 연산
        parent[b] = a
        cost[b] = tentative_cost

    while len(priority_queue) != 0:  # 우선순위 큐에 남아있는 cell이 없을 때까지 (=더 이상 탐색 후보인 fringe가 없을 때까지)
        f_curr, curr = heapq.heappop(priority_queue)  # 우선순위 큐에서 pop
        visited_cells.append(curr)  # 방문 기록에 추가

        if curr != goal:  # 아직 goal에 도착하지 않았다면,
            neighbours = maze.find_neighbours(curr[0], curr[1])  # 현재 위치에서 이웃 찾기
            neighbours = maze._validate_neighbours_generate(neighbours)  # 이웃 셀 필터링 1
            if neighbours is not None:  # None 객체 참조 방지
                neighbours = maze.validate_neighbours_solve(neighbours, curr[0], curr[1], goal[0], goal[1],
                                                            "brute-force")  # 이웃 셀 필터링 2

            if neighbours is not None:  # 만약, 추가적으로 탐색 가능한 셀들이 없다면 동작 무시
                for neighbour in neighbours:
                    temp_cost = cost[curr] + adj_distance(neighbour, curr)
                    if neighbour not in cost or temp_cost < cost[
                        neighbour]:  # 잠정적 cost가 더 작은 경우에만 연산을 수행하기에 업데이트가 안 된 old data는 자동적으로 무시됨.
                        relaxation(curr, neighbour, temp_cost)  # relaxation 연산
                        maze.grid[neighbour[0]][neighbour[1]].visited = True  # 방문 표시
                        heapq.heappush(priority_queue,
                                       (temp_cost + heuristic_function(neighbour, goal), neighbour))  # 우선순위 큐에 push

        else:  # 만약, goal에 도착했다면,
            while curr in parent:  # 최적 해 경로 역추적
                maze.optimal_solution_path.append(curr)
                curr = parent[curr]  # 해당 노드의 부모 노드를 참조함으로써 역추적
            maze.optimal_solution_path.append(start)
            maze.optimal_solution_path.reverse()

            for curr in visited_cells:  # 방문 cell들 필터링 작업.
                if curr in maze.optimal_solution_path:  # 방문 경로 저장
                    maze.solution_path.append((curr, False))  # 만약 해당 셀이 최적 해에 포함되어 있다면 활성상태 False로 설정.
                else:
                    maze.solution_path.append((curr, True))  # 만약 해당 셀이 최적 해에 포함되어 있지 않다면 활성상태 True로 설정.

            print("optimal total cost: {:.4f}".format(cost[goal]))
            print("Number of moves performed: {}".format(len(maze.solution_path)))
            print("Execution time for algorithm: {:.4f}".format(time.time() - time_start))

            # file output
            with open('episodes.txt', 'w') as file:
                for coord in maze.optimal_solution_path:
                    file.write(f"{coord}\n")

            return maze.optimal_solution_path, cost[goal]  # 최적 해와 비용 return

    return None, -1  # 만약 해가 존재하지 않다면,


def uniform_cost_search(maze):  # ucs 알고리즘으로 최적해 구하기
    start = maze.entry_coor
    goal = maze.exit_coor
    priority_queue = []  # 우선순위 큐 선언 (for f의 최솟값 찾기)
    heapq.heappush(priority_queue, (0, start))  # (g, coord) 형태로 우선순위에 저장
    parent = {}  # 경로 역추적용
    cost = {start: 0.0}  # start로부터 실제로 든 비용 (so far)
    visited_cells = []  # 방문한 노드들을 모두 저장.
    path = []  # 방문한 노드들을 저장하되, 최적 해에 포함되지 않는 cell은 True로 설정.

    print("\nSolving the maze with uniform cost search...")
    time_start = time.time()  # 걸린 시간 check!

    def relaxation(a, b, tentative_cost):  # 노드 a와 b 사이에서 relaxation 연산
        parent[b] = a
        cost[b] = tentative_cost

    while len(priority_queue) != 0:  # 우선순위 큐에 남아있는 cell이 없을 때까지 (=더 이상 탐색 후보인 fringe가 없을 때까지)
        f_curr, curr = heapq.heappop(priority_queue)  # 우선순위 큐에서 pop
        visited_cells.append(curr)  # 방문 기록에 추가

        if curr != goal:  # 아직 goal에 도착하지 않았다면,
            neighbours = maze.find_neighbours(curr[0], curr[1])  # 현재 위치에서 이웃 찾기
            neighbours = maze._validate_neighbours_generate(neighbours)  # 이웃 셀 필터링 1
            if neighbours is not None:  # None 객체 참조 방지
                neighbours = maze.validate_neighbours_solve(neighbours, curr[0], curr[1], goal[0], goal[1],
                                                            "brute-force")  # 이웃 셀 필터링 2

            if neighbours is not None:  # 만약, 추가적으로 탐색 가능한 셀들이 없다면 동작 무시
                for neighbour in neighbours:
                    temp_cost = cost[curr] + adj_distance(neighbour, curr)
                    if neighbour not in cost or temp_cost < cost[
                        neighbour]:  # 잠정적 cost가 더 작은 경우에만 연산을 수행하기에 업데이트가 안 된 old data는 자동적으로 무시됨.
                        relaxation(curr, neighbour, temp_cost)  # relaxation 연산
                        maze.grid[neighbour[0]][neighbour[1]].visited = True  # 방문 표시
                        heapq.heappush(priority_queue, (temp_cost, neighbour))  # 우선순위 큐에 push

        else:  # 만약, goal에 도착했다면,
            while curr in parent:  # 최적 해 경로 역추적
                maze.optimal_solution_path.append(curr)
                curr = parent[curr]  # 해당 노드의 부모 노드를 참조함으로써 역추적
            maze.optimal_solution_path.append(start)
            maze.optimal_solution_path.reverse()

            for curr in visited_cells:  # 방문 cell들 필터링 작업.
                if curr in maze.optimal_solution_path:  # 방문 기록 저장
                    path.append((curr, False))  # 만약 해당 셀이 최적 해에 포함되어 있다면 활성상태 False로 설정.
                else:
                    path.append((curr, True))  # 만약 해당 셀이 최적 해에 포함되어 있지 않다면 활성상태 True로 설정.

            print("optimal total cost: {:.4f}".format(cost[goal]))
            print("Number of moves performed: {}".format(len(path)))
            print("Execution time for algorithm: {:.4f}".format(time.time() - time_start))

            return path, cost[goal]  # 최적 해와 비용 return

    return None, -1  # 만약 해가 존재하지 않다면,


# Q-learning (HW#2)
def S2I(state, cnt):  # state -> index
    return state[0] * cnt + state[1]

def go(state, action, maze):  # return next state, reward, flag
    reward = 0
    next_state = None

    row, col = state

    # 먼저 다음 state 구하기
    if action == 0:  # 상
        next_state = (row + 1, col)
    elif action == 1:  # 하
        next_state = (row - 1, col)
    elif action == 2:  # 좌
        next_state = (row, col - 1)
    else:  # 우
        next_state = (row, col + 1)

    # 다음 state가 유효한 지 확인
    if next_state[0] < 0 or next_state[0] >= maze.num_rows or next_state[1] < 0 or next_state[1] >= maze.num_cols:
        #print(state, " invalid operation! \n")  # for debugging purpose
        reward = -1
        next_state = state

    # wall 존재 유무 판단
    elif maze.grid[state[0]][state[1]].is_walls_between(maze.grid[next_state[0]][next_state[1]]):
        reward = -1
        next_state = state  # 벽이 있어 다음 state로 가지 X
    else:
        reward = -0.1  # 다음 state로 넘어갈 수 있다면,

    # goal에 도달했는지 판단
    if next_state == maze.exit_coor:
        return next_state, 100, True
    else:
        return next_state, reward, False


def get_valid_action1(state, maze):  # 현재 state에서 가능한 action들만 추출 --- 1
    row, col = state
    valid_action = [0, 1, 2, 3]

    if row == maze.num_rows - 1:
        valid_action.remove(0)  # 상 X
    if col == maze.num_cols - 1:
        valid_action.remove(3)  # 우 X
    if row == 0:
        valid_action.remove(1)  # 하 X
    if col == 0:
        valid_action.remove(2)  # 좌 X

    return valid_action

def get_valid_action2(state, actions, maze):  # 현재 state에서 가능한 action들만 추출 --- 2
    row, col = state
    temp = []

    for a in actions:
        if a == 0:  # 상
            next_state = (row + 1, col)
            if maze.grid[state[0]][state[1]].is_walls_between(maze.grid[next_state[0]][next_state[1]]):
                temp.append(a)
        if a == 1:  # 하
            next_state = (row - 1, col)
            if maze.grid[state[0]][state[1]].is_walls_between(maze.grid[next_state[0]][next_state[1]]):
                temp.append(a)
        if a == 2:  # 좌
            next_state = (row, col - 1)
            if maze.grid[state[0]][state[1]].is_walls_between(maze.grid[next_state[0]][next_state[1]]):
                temp.append(a)
        if a == 3:  # 우
            next_state = (row, col + 1)
            if maze.grid[state[0]][state[1]].is_walls_between(maze.grid[next_state[0]][next_state[1]]):
                temp.append(a)

    return [element for element in actions if element not in temp]

def q_learning(maze, episodes, alpha, gamma, epsilon):  # create Q table
    total_time_start = time.time()  # for experimenting

    S = maze.num_rows * maze.num_cols  # 총 가능한 state의 개수
    A = 4
    """
    action: <integer>
    - 0: 상
    - 1: 하
    - 2: 좌
    - 3: 우
    
    state: <tuple>
    - (i, j): i행 j열
    
    q_table: <2D array>
    - [s, a]: Q value 
    """
    q_table = np.zeros((S, A))  # init Q table with '0'

    file = open("results.txt", "a")  # open file stream
    file.write(f"{alpha},{gamma}: \n")
    # episodes번 반복
    for i in range(episodes):
        episode_start = time.time()  # for experimenting
        # init state & flag
        """
        state 정보:
        - s, s_index : 현재 state 정보
        - ns, ns_index : 다음 state 정보
        
        action 정보:
        - action
        
        학습 종료 조건 정보:
        - flag
        """
        s = maze.entry_coor
        flag = False

        # print(s, "\n")

        cnt = 0
        while not flag:
            cnt += 1
            # init 작업
            s_index = S2I(s, maze.num_cols)

            #valid_action = get_valid_action1(s, maze)

            # by epsilon greedy policy
            if random.random() < epsilon:
                #action = random.choice(valid_action)
                action = random.choice([0, 1, 2, 3])
            else:
                #q_values = [q_table[s_index, a] for a in valid_action]
                #action = valid_action[np.argmax(q_values)]  # find maximum Q value among current state (meaning row)
                action = np.argmax(q_table[s_index])

            # (next state, reward, 종료 조건) 구하기
            ns, reward, flag = go(s, action, maze)
            ns_index = S2I(ns, maze.num_cols)

            old_value = q_table[s_index, action]
            new_value = reward + gamma * np.max(q_table[ns_index])
            q_table[s_index, action] = (1 - alpha) * old_value + alpha * new_value  # update Q value

            # for debugging purpose
            #if s == ns:
                #print("Loop!", s, ns, "\n")

            s = ns  # 다음 state로 이동!

        print("successfully reached goal!", "(", i, ") -> ", cnt, "\n")  # for debugging purpose
        # file.write(f"{i} episode learning exec time: {time.time() - episode_start}\n")  # for experimenting
        epsilon *= 0.99  # reduce epsilon
        epsilon = max(epsilon, 0.1)  # set under-bound

    # file.write("Q-learning finished!\n")
    file.write(f"Total learning time: {time.time()- total_time_start}\n")  # for experimenting
    file.close()  # close file stream

    return q_table

def print_q_table(q_table, maze):
    with open('q_table.txt', 'w') as file:
        for i in range(maze.num_rows-1, -1, -1):
            for j in range(maze.num_cols):
                # file output
                q_values = []
                state = (i, j)
                s_index = S2I(state, maze.num_cols)
                for k in range(4):
                    q_values.append(q_table[s_index, k])
                #file.write(f"{q_values} ")
            file.write("\n")


def q_learning_path(maze, q_table):  # find path using Q table
    time_start = time.time()  # for experimenting (time execution)

    path = []
    state = maze.entry_coor  # 시작 위치
    maze.solution_path = []

    with open('results.txt', 'a') as file:  # for experiment
        while state != maze.exit_coor:
            path.append(state)
            #file.write(f"{state}\n ")
            maze.solution_path.append((state, False))

            s_index = S2I(state, maze.num_cols)
            valid_actions = get_valid_action1(state, maze)  # filtering 1
            valid_actions = get_valid_action2(state, valid_actions, maze)  # filtering 2
            q_values = [q_table[s_index, a] for a in valid_actions]
            action = valid_actions[np.argmax(q_values)]  # find maximum Q value among current state (meaning row)
            #action = np.argmax(q_table[s_index])  # find optimal action
            state, reward, flag = go(state, action, maze)  # go next state! (cf. "reward", "flag" is useless var)
            print(state, action, reward, flag, "\n")  # for debugging purpose

        path.append(state)
        #file.write(f"{state}\n ")
        maze.solution_path.append((state, False))

        #file.write("Finish finding path!\n")
        exec_time = time.time() - time_start  # total execution time
        file.write("Total execution time for finding path using Q-table {:.4f}\n".format(exec_time))
        file.write("path length: {}\n\n".format(len(path)))

    return path

def filter_optimal_path(path, maze):  # path 중에서 최적 경로를 filtering
    maze.solution_path = []
    for curr in path:
        if curr in maze.optimal_solution_path:  # 방문 기록 저장
            maze.solution_path.append((curr, False))  # 만약 해당 셀이 최적 해에 포함되어 있다면 활성상태 False로 설정.
        else:
            maze.solution_path.append((curr, True))  # 만약 해당 셀이 최적 해에 포함되어 있지 않다면 활성상태 True로 설정.
