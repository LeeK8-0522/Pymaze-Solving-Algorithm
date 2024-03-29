import heapq
import time
import logging

from src.maze import Maze
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
                    maze.id = self.mazes.__len__()+1
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
        elif method == "A-Star":
            maze.solution_path, maze.solution_cost = a_star_search(maze, manhattan_distance)
        else:
            maze.solution_path, maze.solution_cost = uniform_cost_search(maze)

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

    def show_solution_animation(self, id, cell_size =1):
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
        return next((maze for maze in self.mazes if maze .id == id), None)  # generator(리스트 컴프리헨션과 유사)를 통해 동일한 id를 갖는 첫 번째 미로만을 반환. 만약 없을 시, 두 번째 인자인 'None'을 반환.

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
        self.quiet_mode=enabled

def manhattan_distance(coord1, coord2):  # 두 좌표 사이의 맨허튼 거리를 계산 (수평은 0.9, 수직은 1.1 penalty)
     return 0.9 * abs(coord1[0] - coord2[0]) + 1.1 * abs(coord1[1] - coord2[1])

def a_star_search(maze, heuristic_function):  # a * 탐색 알고리즘으로 최적해 구하기
    start = maze.entry_coor
    goal = maze.exit_coor
    maze.grid[start[0]][start[1]].visited = True  # start 노드 방문 표시
    priority_queue = []  # 우선순위 큐 선언 (for f의 최솟값 찾기)
    heapq.heappush(priority_queue, (0 + heuristic_function(start, goal), start))  # (f, coord) 튜플 형태로 우선순위에 저장
    parent = {}  # 경로 역추적용. 딕셔너리 자료형을 이용하여 '[a] -> b' 형태로 저장.
    cost = {start: 0.0}  # start로부터 실제로 든 비용 (so far). 딕셔너리 자료형을 이용하여 '[a] -> cost' 형태로 저장.
    step = 0  # 최적 해를 구하기 위해 수행한 연산 단계의 수

    print("\nSolving the maze with a-star search...")
    time_start = time.time()  # 걸린 시간 check!

    def relaxation(a, b, tentative_cost):  # 노드 a와 b 사이에서 relaxation 연산
        parent[b] = a
        cost[b] = tentative_cost

    while len(priority_queue) != 0:
        step += 1
        f_curr, curr = heapq.heappop(priority_queue)  # 우선순위 큐에서 pop
        maze.grid[curr[0]][curr[1]].visited = True  # 방문 표시

        if curr == goal:  # 만약, goal에 도착했다면,
            path = []  # for solution path 저장
            while curr in parent:  # 경로 역추적
                path.append((curr, False))
                curr = parent[curr]  # 해당 노드의 부모 노드를 참조함으로써 역추적
            path.append((start, False))
            path.reverse()

            print("optimal total cost: {:.4f}".format(cost[goal]))
            print("Number of moves performed: {}".format(step))
            print("Execution time for algorithm: {:.4f}".format(time.time() - time_start))

            return path, cost[goal]  # 최적 해와 비용 return

        neighbours = maze.find_neighbours(curr[0], curr[1])  # 현재 위치에서 이웃 찾기
        neighbours = maze._validate_neighbours_generate(neighbours)  # 이웃 셀 필터링 1
        if neighbours is not None:  # None 객체 참조 방지
            neighbours = maze.validate_neighbours_solve(neighbours, curr[0], curr[1], goal[0], goal[1], "brute-force")  # 이웃 셀 필터링 2

        if neighbours is not None:  # 만약, 추가적으로 탐색 가능한 셀들이 없다면 동작 무시
            for neighbour in neighbours:
                temp_cost = cost[curr] + heuristic_function(neighbour,
                                                            curr)  # 주의!) 여기서 heuristic 값을 구하는 것은 아니지만 동일한 효과를 낼 수 있기에 맨허튼 거리 함수 사용
                if neighbour not in cost or temp_cost < cost[neighbour]:  # 잠정적 cost가 더 작은 경우에만 연산을 수행하기에 업데이트가 안 된 old data는 자동적으로 무시됨.
                    relaxation(curr, neighbour, temp_cost)  # relaxation 연산
                    heapq.heappush(priority_queue,(temp_cost + heuristic_function(neighbour, goal), neighbour))  # 우선순위 큐에 push

    return None, -1  # 만약 해가 존재하지 않다면,

def uniform_cost_search(maze):  # ucs 알고리즘으로 최적해 구하기
    start = maze.entry_coor
    goal = maze.exit_coor
    maze.grid[start[0]][start[1]].visited = True  # start 노드 방문 표시
    priority_queue = []  # 우선순위 큐 선언 (for f의 최솟값 찾기)
    heapq.heappush(priority_queue, (0, start))  # (g, coord) 형태로 우선순위에 저장
    parent = {}  # 경로 역추적용
    cost = {start: 0.0} # start로부터 실제로 든 비용 (so far)
    step = 0  # 최적 해를 구하기 위해 수행한 연산 단계의 수

    print("\nSolving the maze with uniform cost search...")
    time_start = time.time()  # 걸린 시간 check!

    def relaxation(a, b, tentative_cost):  # 노드 a와 b 사이에서 relaxation 연산
        parent[b] = a
        cost[b] = tentative_cost

    while len(priority_queue) != 0:
        step += 1
        f_curr, curr = heapq.heappop(priority_queue)  # 우선순위 큐에서 pop
        maze.grid[curr[0]][curr[1]].visited = True  # 방문 표시

        if curr == goal:  # 만약, goal에 도착했다면,
            path = []  # for solution path 저장
            while curr in parent:  # 경로 역추적
                path.append((curr, False))
                curr = parent[curr]  # 해당 노드의 부모 노드를 참조함으로써 역추적
            path.append((start, False))
            path.reverse()

            print("optimal total cost: {:.4f}".format(cost[goal]))
            print("Number of moves performed: {}".format(step))
            print("Execution time for algorithm: {:.4f}".format(time.time() - time_start))

            return path, cost[goal]  # 최적 해와 비용 return

        neighbours = maze.find_neighbours(curr[0], curr[1])  # 현재 위치에서 이웃 찾기
        neighbours = maze._validate_neighbours_generate(neighbours)  # 이웃 셀 필터링 1
        if neighbours is not None:  # None 객체 참조 방지
            neighbours = maze.validate_neighbours_solve(neighbours, curr[0], curr[1], goal[0], goal[1], "brute-force")  # 이웃 셀 필터링 2

        if neighbours is not None:  # 만약, 추가적으로 탐색 가능한 셀들이 없다면 동작 무시
            for neighbour in neighbours:
                temp_cost = cost[curr] + manhattan_distance(neighbour,
                                                            curr)  # 주의!) 여기서 heuristic 값을 구하는 것은 아니지만 동일한 효과를 낼 수 있기에 맨허튼 거리 함수 사용
                if neighbour not in cost or temp_cost < cost[neighbour]:  # 잠정적 cost가 더 작은 경우에만 연산을 수행하기에 업데이트가 안 된 old data는 자동적으로 무시됨.
                    relaxation(curr, neighbour, temp_cost)  # relaxation 연산
                    heapq.heappush(priority_queue,(temp_cost, neighbour))  # 우선순위 큐에 push

    return None, -1  # 만약 해가 존재하지 않다면,
