import pygame
import random
import numpy as np

import torch
import torch.nn as nn

# Define the maze size
MAZE_CELLS_X, MAZE_CELLS_Y = 5, 5
MAZE_WIDTH, MAZE_HEIGHT = 20 * MAZE_CELLS_X, 20 * MAZE_CELLS_Y
CELL_WIDTH = (MAZE_WIDTH - (MAZE_CELLS_X - 1) * 2) // MAZE_CELLS_X
CELL_HEIGHT = (MAZE_HEIGHT - (MAZE_CELLS_Y - 1) * 2) // MAZE_CELLS_Y

# Define NN structure
INPUT_SIZE = 2 + MAZE_CELLS_X * MAZE_CELLS_Y * 4
HIDDEN_SIZE = 2 * INPUT_SIZE
OUTPUT_SIZE = 4
NUM_HIDDEN = 5

# Initialize Pygame
pygame.init()
window = pygame.display.set_mode((MAZE_WIDTH, MAZE_HEIGHT))
pygame.display.set_caption("Maze Solving AI")
clock = pygame.time.Clock()

# Define some colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

DROPOUT_RATE = 0

# Test parameters (feel free to change)
NUM_ROUNDS = 20
FRAME_RATE = 10


# Define the MazeCell class
class MazeCell:
    def __init__(self):
        self.visited = False
        self.top = True
        self.bottom = True
        self.left = True
        self.right = True


# Define the Maze class
class Maze:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.cells = [[MazeCell() for _ in range(width)] for _ in range(height)]
        self.path = []

    def get_neighbors(self, x, y):
        neighbors = []
        if y > 0:
            neighbors.append((x, y - 1))  # Top neighbor
        if y < self.height - 1:
            neighbors.append((x, y + 1))  # Bottom neighbor
        if x > 0:
            neighbors.append((x - 1, y))  # Left neighbor
        if x < self.width - 1:
            neighbors.append((x + 1, y))  # Right neighbor
        return neighbors

    def generate(self):  # Generate maze using Randomized Depth-First Search
        stack = [(0, 0)]
        visited = set()

        while stack:
            x, y = stack[-1]
            self.cells[x][y].visited = True
            visited.add((x, y))

            neighbors = [(nx, ny) for nx, ny in self.get_neighbors(x, y) if (nx, ny) not in visited]
            if neighbors:
                nx, ny = random.choice(neighbors)
                if nx == x:  # Move vertically
                    if ny > y:  # Move down
                        self.cells[x][y].bottom = False
                        self.cells[nx][ny].top = False
                    else:  # Move up
                        self.cells[x][y].top = False
                        self.cells[nx][ny].bottom = False
                else:  # Move horizontally
                    if nx > x:  # Move right
                        self.cells[x][y].right = False
                        self.cells[nx][ny].left = False
                    else:  # Move left
                        self.cells[x][y].left = False
                        self.cells[nx][ny].right = False

                stack.append((nx, ny))
            else:
                stack.pop()

    def draw(self):
        for x in range(self.width):
            for y in range(self.height):
                cell = self.cells[x][y]
                cell_x, cell_y = x * (CELL_WIDTH + 2), y * (CELL_HEIGHT + 2)

                # Draw top, bottom, left, and right walls
                if cell.top:
                    pygame.draw.rect(window, BLACK, (cell_x, cell_y, CELL_WIDTH + 2, 2))
                if cell.bottom:
                    pygame.draw.rect(window, BLACK, (cell_x, cell_y + CELL_HEIGHT + 2, CELL_WIDTH + 2, 2))
                if cell.left:
                    pygame.draw.rect(window, BLACK, (cell_x, cell_y, 2, CELL_HEIGHT + 2))
                if cell.right:
                    pygame.draw.rect(window, BLACK, (cell_x + CELL_WIDTH + 2, cell_y, 2, CELL_HEIGHT + 2))


# Define Deep Q Neural Network class
class DeepQNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DeepQNetwork, self).__init__()

        # Prepare hidden layer definitions for unpacking
        hidden_layers = []
        for _ in range(NUM_HIDDEN - 1):
            hidden_layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(DROPOUT_RATE)])

        # Define layers
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # Input
            nn.ReLU(),
            *hidden_layers,  # Main hidden layers (unpacked)
            nn.Linear(hidden_size, hidden_size // 2),  # Ramp-down layer
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(hidden_size // 2, hidden_size // 4),  # Ramp-down layer
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(hidden_size // 4, output_size),  # Output layer
            nn.Sigmoid()  # Sigmoid to encourage 'threshold' behaviour of actions
        )
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        return self.layers(x)


# Define Agent class
class Agent:
    def __init__(self, input_size, hidden_size, output_size):
        self.action_space = [i for i in range(output_size)]

        self.Q_eval = DeepQNetwork(input_size, hidden_size, output_size)  # Evaluated network

    def choose_action(self, observation):
        state = observation.unsqueeze(0).to(self.Q_eval.device)
        actions = self.Q_eval.forward(state)
        action = torch.argmax(actions).item()

        return action

    @staticmethod
    def draw(x, y):  # Draw a black dot to represent the agent
        cell_x, cell_y = x * (CELL_WIDTH + 2), y * (CELL_HEIGHT + 2)
        pygame.draw.circle(window, BLACK, (cell_x + CELL_WIDTH // 2 + 2, cell_y + CELL_HEIGHT // 2 + 2), 3)


def eval_agent(agent):
    # Generate maze
    maze = Maze(MAZE_CELLS_X, MAZE_CELLS_Y)
    maze.generate()

    # Reset the agent
    x, y = 0, 0
    loc_hist = [(x, y)]

    # Initialize the cell knowledge matrix, -1 means unknown wall state
    cell_hist = [[-1] * 4 for _ in range(MAZE_CELLS_X * MAZE_CELLS_Y)]

    # Add starting cell to cell history. The '* 1' converts the boolean to an integer
    cell_hist[x + y * MAZE_CELLS_X] = [maze.cells[x][y].top * 1,
                                       maze.cells[x][y].bottom * 1,
                                       maze.cells[x][y].left * 1,
                                       maze.cells[x][y].right * 1]

    # Formulate initial input state (observation)
    x_tensor = torch.tensor(x, dtype=torch.float32).view(1, -1)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(1, -1)
    cell_hist_tensor = torch.tensor(cell_hist, dtype=torch.float32).view(1, -1)
    observation = torch.cat((x_tensor, y_tensor, cell_hist_tensor), dim=1)

    # Epoch-specific variables for calculating total reward
    score = 0
    num_steps = 0

    while (x, y) != (MAZE_CELLS_X - 1, MAZE_CELLS_Y - 1) and num_steps < 200:
        # Make move choice
        action = agent.choose_action(observation)

        x_old = x
        y_old = y

        # Move the agent
        if action == 0:  # Up
            if y > 0 and not maze.cells[x][y].top:
                y -= 1
        elif action == 1:  # Down
            if y < MAZE_CELLS_Y - 1 and not maze.cells[x][y].bottom:
                y += 1
        elif action == 2:  # Left
            if x > 0 and not maze.cells[x][y].left:
                x -= 1
        elif action == 3:  # Right
            if x < MAZE_CELLS_X - 1 and not maze.cells[x][y].right:
                x += 1

        num_steps += 1  # Increase step count

        # Calculate new observation
        cell_hist[x + y * MAZE_CELLS_X] = [maze.cells[x][y].top * 1,
                                           maze.cells[x][y].bottom * 1,
                                           maze.cells[x][y].left * 1,
                                           maze.cells[x][y].right * 1]

        x_tensor_ = torch.tensor(x, dtype=torch.float32).view(1, -1)
        y_tensor_ = torch.tensor(y, dtype=torch.float32).view(1, -1)
        cell_hist_tensor_ = torch.tensor(cell_hist, dtype=torch.float32).view(1, -1)
        observation_ = torch.cat((x_tensor_, y_tensor_, cell_hist_tensor_), dim=1)

        # Give reward if exit has been reached
        if (x, y) == (MAZE_CELLS_X - 1, MAZE_CELLS_Y - 1):
            reward = 1  # Reward if reaches exit
        elif (x, y) == (x_old, y_old):
            reward = -0.2  # Punish if moving into wall
        elif (x, y) in loc_hist:
            reward = -0.1  # Punish if moving into known cells
        elif (x, y) not in loc_hist:
            reward = 0  # Reward if discovering new cells
            loc_hist.append((x, y))
        else:
            reward = None

        score += reward

        # Update agent
        observation = observation_

        # Update pygame window
        window.fill(WHITE)
        maze.draw()
        agent.draw(x, y)
        clock.tick(FRAME_RATE)
        pygame.display.flip()

    return score, num_steps


def main():
    print('Using GPU acceleration:', torch.cuda.is_available())

    agent = Agent(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)  # Init agent

    # Load existing state dictionary (optional)
    load_path = "models/example_model.pth"
    agent.Q_eval.load_state_dict(torch.load(load_path))

    scores = []
    for rnd in range(NUM_ROUNDS):
        score, num_steps = eval_agent(agent)
        scores.append(score)
        print(f"round {rnd} - steps: {num_steps} score: {score}")

    print(f"mean: {np.mean(scores)}, std: {np.std(scores)}")

    pygame.quit()


if __name__ == "__main__":
    main()
