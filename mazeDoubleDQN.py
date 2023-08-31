import os
import datetime
import pygame
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

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

# Define training hyperparameters
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 1.00
EPS_END = 0.05
EPS_DECAY = 0.997
LR = 1e-4
MAX_MEM_SIZE = 1000000
TARGET_UPDATE_FREQ = 1
DROPOUT_RATE = 0.5
TAU = 0.01

# Define the training regime
NUM_EPOCHS = 10
MAX_STEPS = 200


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
        self.optimizer = optim.Adam(self.parameters(), lr=LR)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        return self.layers(x)


# Define Agent class
class Agent:
    def __init__(self, input_size, hidden_size, output_size):
        self.loss = None
        self.gamma = GAMMA
        self.epsilon = EPS_START
        self.eps_min = EPS_END
        self.eps_dec = EPS_DECAY
        self.lr = LR
        self.action_space = [i for i in range(output_size)]
        self.mem_size = MAX_MEM_SIZE
        self.batch_size = BATCH_SIZE
        self.mem_cntr = 0

        self.Q_eval = DeepQNetwork(input_size, hidden_size, output_size)  # Evaluated network
        self.Q_target = DeepQNetwork(input_size, hidden_size, output_size)  # Target network

        # Initialize arrays used for batch learning
        self.state_memory = np.zeros((self.mem_size, input_size), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_size), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.done_memory = np.zeros(self.mem_size, dtype=np.uint8)

    def store_transition(self, state, action, reward, state_, done):  # Store actions and consequences
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.done_memory[index] = done

        self.mem_cntr += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = observation.unsqueeze(0).to(self.Q_target.device)
            actions = self.Q_target.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        state_batch = torch.tensor(self.state_memory[batch], dtype=torch.float32, device=self.Q_eval.device)
        new_state_batch = torch.tensor(self.new_state_memory[batch], dtype=torch.float32, device=self.Q_eval.device)
        reward_batch = torch.tensor(self.reward_memory[batch], dtype=torch.float32, device=self.Q_eval.device)
        action_batch = torch.tensor(self.action_memory[batch], dtype=torch.int64, device=self.Q_eval.device)
        done_batch = torch.tensor(self.done_memory[batch], dtype=torch.int8, device=self.Q_eval.device)

        q_eval = self.Q_eval.forward(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        q_next = self.Q_target.forward(new_state_batch)
        q_next_max, _ = q_next.max(dim=1)

        q_target = reward_batch + (1 - done_batch) * self.gamma * q_next_max

        self.loss = self.Q_eval.loss(q_target, q_eval)
        self.loss.backward()

        for param in self.Q_eval.parameters():  # Clip high gradients for stability
            param.grad.data.clamp_(-0.5, 0.5)

        self.Q_eval.optimizer.step()

    def update_target(self):  # Copy evaluated network state to target network
        self.Q_target.load_state_dict(self.Q_eval.state_dict())

    def soft_update_target(self):  # Soft update variant
        eval_state_dict = self.Q_eval.state_dict()
        target_state_dict = self.Q_target.state_dict()
        for key in eval_state_dict:
            target_state_dict[key] = eval_state_dict[key] * TAU + (1 - TAU) * target_state_dict[key]
        self.Q_target.load_state_dict(target_state_dict)

    def decay_epsilon(self):
        self.epsilon = self.epsilon * EPS_DECAY if self.epsilon > self.eps_min else self.eps_min

    @staticmethod
    def draw(x, y):  # Draw a black dot to represent the agent
        cell_x, cell_y = x * (CELL_WIDTH + 2), y * (CELL_HEIGHT + 2)
        pygame.draw.circle(window, BLACK, (cell_x + CELL_WIDTH // 2 + 2, cell_y + CELL_HEIGHT // 2 + 2), 3)


# Q-learning algorithm
def train_agent(agent, num_epochs, max_steps):
    # Set up live plot
    scores, eps_hist, avg_scores, losses = [], [], [], []
    plt.ion()
    fig1, (ax1, ax3) = plt.subplots(2, 1)
    ax2 = ax1.twinx()

    for epoch in range(num_epochs):
        # Generate maze
        maze = Maze(MAZE_CELLS_X, MAZE_CELLS_Y)
        maze.generate()

        # Reset the agent
        x, y = 0, 0
        loc_hist = [(x, y)]

        # Initialize the cell knowledge matrix, -1 means unknown wall state
        cell_hist = [[-1] * 4 for _ in range(MAZE_CELLS_X * MAZE_CELLS_Y)]

        # Add starting cell to cell history. The '* 1' is a trick that converts the boolean to an integer
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

        while (x, y) != (MAZE_CELLS_X - 1, MAZE_CELLS_Y - 1) and num_steps < max_steps:
            # Store current state (before action)
            x_old = x
            y_old = y

            # Make move choice
            action = agent.choose_action(observation)

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

            # Give reward
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
                reward = None  # To suppress a warning (and just in case, will raise an error)

            score += reward

            # Update agent
            done = (x, y) == (MAZE_CELLS_X - 1, MAZE_CELLS_Y - 1)
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_

            # Update pygame window
            window.fill(WHITE)
            maze.draw()
            agent.draw(x, y)
            pygame.display.flip()

            # Update loss list
            if agent.loss is not None:
                losses.append(agent.loss.item())

        # Update target network
        if epoch % TARGET_UPDATE_FREQ == 0:
            agent.soft_update_target()

        # Add data to list
        scores.append(score)
        eps_hist.append(agent.epsilon)
        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)

        # Update plots
        ax1.clear()
        ax1.plot(range(epoch+1), scores, color='tab:blue')
        ax1.plot(range(epoch+1), avg_scores, color='midnightblue')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Score', color='tab:blue')

        ax2.clear()
        ax2.plot(range(epoch+1), eps_hist, color='tab:orange')
        ax2.set_ylabel('Epsilon', color='tab:orange')
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.tick_right()
        ax2.set_ylim([0, 1])

        ax3.clear()
        ax3.plot(losses)
        ax3.set_xticks([])
        ax3.set_ylim(bottom=0)
        ax3.set_ylabel('Loss')

        # Refresh the plots and pause for a short duration
        plt.pause(0.1)

        # Print the total reward and loss for this epoch
        print('epoch ', epoch, ': score = %.2f' % score, 'avg %.2f' % avg_score, 'eps %.3f' % agent.epsilon)
        agent.decay_epsilon()

    # Save agent model
    print('Saving model...')
    current_datetime = datetime.datetime.now().strftime("%d_%m_%H%M")
    save_name = f"{MAZE_CELLS_X}x{MAZE_CELLS_Y}_{NUM_HIDDEN}+2h_{HIDDEN_SIZE}n_ep{num_epochs}_{current_datetime}.pth"
    save_path = os.path.join("models", save_name)
    torch.save(agent.Q_target.state_dict(), save_path)
    print(f"Model state dictionary saved at {save_path}")

    plt.ioff()
    plt.show()


def main():
    print('Using GPU acceleration:', torch.cuda.is_available())

    agent = Agent(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)  # Init agent
    train_agent(agent, NUM_EPOCHS, MAX_STEPS)  # Train agent

    pygame.quit()


if __name__ == "__main__":
    main()
