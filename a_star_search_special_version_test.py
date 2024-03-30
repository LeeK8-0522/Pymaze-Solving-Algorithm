from __future__ import absolute_import
from src.maze_manager import MazeManager
from src.maze import Maze

if __name__ == "__main__":

    # Create the manager
    manager = MazeManager()

    # Add a 20x20 maze to the manager
    maze = manager.add_maze(20, 20)

    # Solve the maze using A Star search algorithm special version
    manager.solve_maze(maze.id, "A-Star special version")

    # Show how the maze was solved
    manager.show_solution_animation(maze.id)
