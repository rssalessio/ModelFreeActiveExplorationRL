from envs.maze import Maze, MazeParameters, Action
DISCOUNT_FACTOR = 0.99
MAZE_PARAMETERS = MazeParameters(
    num_rows=16,
    num_columns=16,
    slippery_probability=0.4,
    walls=[(1,1), (2,2), (0,4), (1,4),  (4,0), (4,1), 
           (5,5), (5,6), (5,7), (5, 8),
           (6,5), (6,6), (6,7), (6, 8),
           (7,5), (7,6), (7,7), (7, 8),
           (8,5), (8,6), (8,7), (8, 8)],
    random_walls=False
)