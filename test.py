from __future__ import absolute_import
from src.maze_manager import MazeManager
from src.maze import Maze

if __name__ == "__main__":

    # Create the manager
    manager = MazeManager()

    # Add a 20x20 maze to the manager
    maze = manager.add_maze(20, 20)

    # Solve the maze using the Depth First Backtracker algorithm
    manager.solve_maze(maze.id, "A-Star")
    # manager.solve_maze(maze.id, "A-Star special version")
    # manager.solve_maze(maze.id, "UCS")

    # Show how the maze was solved
    manager.show_solution_animation(maze.id)
