# Pymaze-Solving-Algorithm
with A* search algorithm and UCS algorithm

random seed value: 
-> for algorithm.py seed(0)
-> maze.py seed(1)

modified part:
- maze.py: In class maze, I added a field "optimal_solution_path" which store optimal solution path data using list and tupe structure. (data format: (row, col))
- maze.manager.py: I added three different functions. 'manhattan_distance', 'a_star_search', and 'uniform_cost_search'.
  'manhattan_distance' function gets two coodrinate for arguments and return distance between two coordinates. (applied 1.1 for vertical steps, 0.9 for horizontal cost)
  'a_star_search' function gets maze instance and heuristic_function for arguments and return return its path and optimal
