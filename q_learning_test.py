from __future__ import absolute_import
import matplotlib
from src.maze_manager import MazeManager
from src.maze_viz import Visualizer
from src.maze import Maze
matplotlib.use('TkAgg')

if __name__ == "__main__":

    # Create the manager
    manager = MazeManager()

    # Add a 20x20 maze to the manager
    maze = manager.add_maze(20, 20)

    # Solve the maze using A Star search algorithm special version
    manager.solve_maze(maze.id, "Q-learning")

    # Show how the maze was solved
    manager.show_solution_animation(maze.id)


