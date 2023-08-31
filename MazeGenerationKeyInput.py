import pygame
import random

# Initialize the maze size and agent properties
maze_cells_x, maze_cells_y = 5, 5
maze_width, maze_height = 20*maze_cells_x, 20*maze_cells_y
cell_width = (maze_width - (maze_cells_x - 1) * 2) // maze_cells_x
cell_height = (maze_height - (maze_cells_y - 1) * 2) // maze_cells_y

# Initialize Pygame
pygame.init()
window = pygame.display.set_mode((maze_width, maze_height))
pygame.display.set_caption("Randomized Depth-First Search Maze")
clock = pygame.time.Clock()

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Toggle maze exit path
show_path = False


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

    def generate(self):
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

    def dfs(self, start, end):
        stack = [start]
        visited = set()

        while stack:
            x, y = stack[-1]

            if (x, y) == end:
                return stack

            visited.add((x, y))
            neighbors = [(nx, ny) for nx, ny in self.get_neighbors(x, y) if
                         (nx, ny) not in visited and not self.has_wall(x, y, nx, ny)]
            if neighbors:
                stack.append(random.choice(neighbors))
            else:
                stack.pop()

        return []

    def has_wall(self, x1, y1, x2, y2):
        # Check if there's a wall between two adjacent cells
        if x1 == x2:  # Moving vertically
            if y2 > y1:
                return self.cells[x1][y1].bottom
            else:
                return self.cells[x1][y1].top
        elif y1 == y2:  # Moving horizontally
            if x2 > x1:
                return self.cells[x1][y1].right
            else:
                return self.cells[x1][y1].left


# Define the Agent class
class Agent:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def move(self, dx, dy, maze):
        new_x, new_y = self.x + dx, self.y + dy
        if 0 <= new_x < maze.width and 0 <= new_y < maze.height and not maze.has_wall(self.x, self.y, new_x, new_y):
            self.x, self.y = new_x, new_y


def draw_maze(maze, visited):
    for x in range(maze.width):
        for y in range(maze.height):
            cell = maze.cells[x][y]
            cell_x, cell_y = x * (cell_width + 2), y * (cell_height + 2)

            if (x, y) in visited:
                # Draw top, bottom, left, and right walls
                if cell.top:
                    pygame.draw.rect(window, BLACK, (cell_x, cell_y, cell_width + 2, 2))
                if cell.bottom:
                    pygame.draw.rect(window, BLACK, (cell_x, cell_y + cell_height + 2, cell_width + 2, 2))
                if cell.left:
                    pygame.draw.rect(window, BLACK, (cell_x, cell_y, 2, cell_height + 2))
                if cell.right:
                    pygame.draw.rect(window, BLACK, (cell_x + cell_width + 2, cell_y, 2, cell_height + 2))

            else:
                pygame.draw.rect(window, (100, 100, 100), (cell_x + 2, cell_y + 2, cell_width, cell_height))

            # Draw the path
            if (x, y) in maze.path and show_path:
                pygame.draw.rect(window, (0, 255, 0), (cell_x + 2, cell_y + 2, cell_width, cell_height))


def draw_agent(agent):
    cell_x, cell_y = agent.x * (cell_width + 2), agent.y * (cell_height + 2)
    pygame.draw.circle(window, (0, 0, 0), (cell_x + cell_width // 2, cell_y + cell_height // 2), cell_width // 2 - 4)


def main():
    for _ in range(3):
        # Generate the maze
        maze = Maze(maze_cells_x, maze_cells_y)
        maze.generate()

        # Store the path from the starting point to the exit
        if show_path:
            maze.path = maze.dfs((0, 0), (maze.width - 1, maze.height - 1))

        # Create the agent at the starting position (0, 0)
        agent = Agent(0, 0)
        visited = [(0, 0)]

        running = True
        while not (agent.x == maze_cells_x-1 and agent.y == maze_cells_y-1) and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        agent.move(0, -1, maze)  # Move up
                    elif event.key == pygame.K_DOWN:
                        agent.move(0, 1, maze)  # Move down
                    elif event.key == pygame.K_LEFT:
                        agent.move(-1, 0, maze)  # Move left
                    elif event.key == pygame.K_RIGHT:
                        agent.move(1, 0, maze)  # Move right

            visited.append((agent.x, agent.y))

            # Clear the window
            window.fill(WHITE)

            # Draw the maze
            draw_maze(maze, visited)

            # Draw the agent
            draw_agent(agent)

            pygame.display.flip()
            clock.tick(10)  # Set the frame rate (you can adjust this value)

    pygame.quit()


if __name__ == "__main__":
    main()
