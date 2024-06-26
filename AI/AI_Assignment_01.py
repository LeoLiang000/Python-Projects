import numpy as np
import heapq
import random
import matplotlib.pyplot as plt
import copy
import random
import time
import tracemalloc

from PIL import Image, ImageDraw
from collections import deque
from matplotlib.colors import ListedColormap
from collections import defaultdict

# moving directions
dirs = [
    (0, 1),  # right
    (1, 0),  # down
    (0, -1),  # left
    (-1, 0)  # up
]


def reconstruct_path(pred, start, end):
    """
    Reconstructs the path from start to end using the predecessors recorded during BFS
    @param pred:  predecessor of each node
    @param start: start point
    @param end: end point
    @return: the shortest path
    """

    current = end
    path = []
    while current != start:
        path.append(current)
        current = pred[current]
    path.append(start)  # add the start node
    path.reverse()  # reverse the path to start->end order
    return path


def DFS(maze):
    """
    @function DFS: Depth-First Search for maze solver
    @param maze: 2D numpy array with maze cell values: 1 for path, 0 for wall, each cell is GridCell type
    @return: path list/None
    """

    h, w = maze.shape[0], maze.shape[1]  # real height and width
    start, end = (0, 0), (h - 1, w - 1)  # set start point and end point

    # explore as deep as possible
    stack = [(start, [start])]  # tuple: position + path
    while stack:
        (x, y), path = stack.pop()
        if (x, y) == end:
            return path

        if not maze[x, y].visited:
            maze[x, y].visited = True

            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w and maze[nx, ny].val == 1:
                    stack.append(
                        ((nx, ny), path + [(nx, ny)])
                    )

    return None


def BFS(maze):
    """
    @function BFS: Breadth-First Search
    @param maze: 2D numpy array with maze cell values: 1 for path, 0 for wall, each cell is GridCell type
    @return: path/None
    """

    h, w = maze.shape[0], maze.shape[1]  # real height and width
    start, end = (0, 0), (h - 1, w - 1)  # set start point and end point
    visited = set()
    pred = {start: None}  # predecessor of each node
    visited.add(start)
    q = deque([start])

    while q:
        x, y = q.popleft()
        if (x, y) == end:
            return reconstruct_path(pred, start, end)

        if not maze[x, y].visited:
            maze[x, y].visited = True

            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w and maze[nx, ny].val == 1:
                    if (nx, ny) not in visited:
                        q.append((nx, ny))
                        visited.add((nx, ny))
                        pred[(nx, ny)] = (x, y)

    return []


def AStar(maze):
    """
    @function AStar: extension of BFS, determine moving direction based on cost
    @param maze: 2D numpy array with maze cell values: 1 for path, 0 for wall, each cell is GridCell type
    @return: True/False
    """

    def getH(start, end):
        """
        @function getH: calculate Heuristic-cost by Manhattan distance
        @param start: current node
        @param end: goal
        @return: Manhattan distance between 2 nodes
        """
        return abs(start[0] - end[0]) + abs(start[1] - end[1])

    h, w = maze.shape[0], maze.shape[1]  # real height and width
    start, end = (0, 0), (h - 1, w - 1)  # set start point and end point
    visited = set()
    pred = {start: None}  # predecessor of each node
    visited.add(start)

    priorQueue = [
        (0 + getH(start, end),  # f: total cost (g+h), where g: current path cost (default 0), h heuristic cost
         0,  # g: current path cost
         0,  # x: current node coordinate x
         0)  # y: current node coordinate y
    ]

    while priorQueue:
        f, g, x, y = heapq.heappop(priorQueue)  # pop the smallest cost node

        if (x, y) == end: return reconstruct_path(pred, start, end)

        if not maze[x, y].visited:
            maze[x, y].visited = True
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w and maze[nx, ny].val == 1 and (nx, ny) not in visited:
                    heapq.heappush(priorQueue, ((g + 1) + getH((nx, ny), end), (g + 1), nx, ny))
                    visited.add((nx, ny))
                    pred[(nx, ny)] = (x, y)
    return []


def MDP_VI(maze, gamma=0.9, threshold=0.01):
    """
    @function MDP_VI: Markov Decision Process with Value Iteration
    @param maze: 2D numpy array with maze cell values: 1 for path, 0 for wall, each cell is GridCell type
    @param gamma: discount factor
    @param threshold: ending condition
    @return:
    """

    h, w = maze.shape[0], maze.shape[1]  # real height and width
    start, end = (0, 0), (h - 1, w - 1)  # set start point and end point

    states = [(x, y) for x in range(h) for y in range(w) if maze[x, y].val != 0]  # movable places (NOT wall)
    valueMap = np.zeros_like(maze, dtype=np.float32)
    reward = -1  # reward policy

    while True:
        delta = 0
        for x, y in states:
            if (x, y) == end: continue  # skip ending point

            tmp_v = valueMap[x, y]  # temp value
            valueMap[x, y] = max([(reward + gamma * valueMap[x + dx, y + dy]) if 0 <= x + dx < h and 0 <= y + dy < w and
                                                                                 maze[
                                                                                     x + dx, y + dy].val != 0 else float(
                '-inf') for dx, dy in dirs])
            delta = max(delta, abs(tmp_v - valueMap[x, y]))

        if delta < threshold: break

    policy = np.zeros_like(maze, dtype=int)  # reflect the shortest path
    for x, y in states:
        if (x, y) == end: continue

        values = [valueMap[x + dx, y + dy] if 0 <= x + dx < h and 0 <= y + dy < w and maze[
            x + dx, y + dy].val != 0 else float('-inf') for dx, dy in dirs]
        policy[x, y] = np.argmax(values)

    return policy


def MDP_PI(maze, gamma=0.9):
    """
    @function MDP_PI: Markov Decision Process with Policy Iteration
    @param policy: movable path
    @param valueMap: Markov Value Chain for each movable grid
    @param maze:
    @param gamma:
    @return:
    """

    def iterValue(policy, valueMap, maze, gamma=0.9, reward=-1, threshold=0.01):
        while True:
            delta = 0
            for x in range(h):
                for y in range(w):
                    if maze[x, y].val == 0 or (x, y) == end: continue

                    tmp_v = valueMap[x, y]
                    dx, dy = dirs[policy[x, y]]
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and maze[nx, ny].val != 0:
                        valueMap[x, y] = reward + gamma * valueMap[nx, ny]
                    else:
                        valueMap[x, y] = reward
                    delta = max(delta, abs(tmp_v - valueMap[x, y]))
            if delta < threshold:
                break
        return valueMap

    h, w = maze.shape[0], maze.shape[1]  # real height and width
    start, end = (0, 0), (h - 1, w - 1)  # set start point and end point
    states = [(x, y) for x in range(h) for y in range(w) if maze[x, y].val != 0]  # movable places (NOT wall)
    valueMap = np.zeros_like(maze, dtype=np.float32)
    policy = np.random.choice(len(dirs), size=maze.shape)  # Initial random policy

    while True:
        valueMap = iterValue(policy, valueMap, maze, gamma)
        policyStable = True

        for x, y in states:
            if (x, y) == end: continue

            # old_action = policy[x, y]
            # values = [valueMap[x + dx, y + dy] if 0 <= x + dx < h and 0 <= y + dy < w and maze[x + dx, y + dy].val == 1 else float('-inf') for dx, dy in dirs]
            # policy[x, y] = np.argmax(values)
            # if old_action != policy[x, y]:
            #     policyStable = False

            oldAction = policy[x, y]
            actionValues = [float('-inf')] * 4

            for actionID, (dx, dy) in enumerate(dirs):
                if 0 <= x + dx < h and 0 <= y + dy < w and maze[x + dx, y + dy].val != 0:
                    actionValues[actionID] = valueMap[x + dx, y + dy]
            maxAction = np.argmax(actionValues)
            policy[x, y] = maxAction

            if oldAction != maxAction:
                policyStable = False

        if policyStable:
            break

    return policy


def plot_policy_on_maze(maze, policy, title=''):
    # Define a simple mapping from policy actions to arrow directions
    arrows = {0: '→', 1: '↓', 2: '←', 3: '↑'}

    fig, ax = plt.subplots(figsize=(maze.shape[1], maze.shape[0]))
    data_clean = maze.astype(np.float32)
    ax.imshow(data_clean, cmap='Pastel1', interpolation='nearest')

    for (i, j), action in np.ndenumerate(policy):
        if maze[i, j] == 1:  # Only plot arrows for open path cells
            ax.text(j, i, arrows[action], ha='center', va='center', fontsize=12, color='black')

    # Hide gridlines and ticks
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    if title:
        plt.title(f'{title} Solving Maze: {int((maze.shape[0] + 1) / 2)}x{int((maze.shape[1] + 1) / 2)}')
    plt.show()


def plotPath(maze, path, title=''):
    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = ListedColormap(['black', 'white'])
    ax.imshow(maze, cmap=cmap)  # 'binary' colormap for black and white

    # Extract X and Y coordinates from the path
    xs, ys = zip(*path)

    # Plot the path on the maze
    ax.plot(ys, xs, color='red', linewidth=2)  # Note the order of ys and xs due to the way matrices are plotted

    # Customize the plot
    ax.set_xticks(np.arange(-.5, maze.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, maze.shape[0], 1), minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", size=0)

    # Hide the axes
    plt.axis('off')
    if title:
        plt.title(f'{title} Solving Maze: {int((maze.shape[0] + 1) / 2)}x{int((maze.shape[1] + 1) / 2)}')
    plt.show()


class GridCell:
    def __init__(self, x, y, val):
        """
        @class GridCell: cell of grid
        @param x: position x
        @param y: position y
        @param val: current point value
        """
        self.x = x
        self.y = y
        self.val = val
        self.visited = False


"""
MazeGenerator is referring to online open source: https://github.com/johnsliao/python-maze-generator
I (Hanwen Liang) have added a transform function for transferring elements of original matrix into GridCell elements 
for usage of implementation of DFS/BFS ...
"""


class Cell:
    def __init__(self):
        self.north = True
        self.south = True
        self.east = True
        self.west = True
        self.visited = False


class Maze:
    def __init__(self, width=20, height=20, cell_width=20):
        self.width = width
        self.height = height
        self.cell_width = cell_width
        self.cells = [[Cell() for _ in range(height)] for _ in range(width)]
        self.generate()
        self.matOrigin = ''
        self.mat = self.to_matrix()  # transform into GridCell (author: Hanwen Liang)

    def generate(self):
        x, y = random.choice(range(self.width)), random.choice(range(self.height))
        self.cells[x][y].visited = True
        path = [(x, y)]

        while not all(all(c.visited for c in cell) for cell in self.cells):
            x, y = path[len(path) - 1][0], path[len(path) - 1][1]

            good_adj_cells = []
            if self.exists(x, y - 1) and not self.cells[x][y - 1].visited:
                good_adj_cells.append('north')
            if self.exists(x, y + 1) and not self.cells[x][y + 1].visited:
                good_adj_cells.append('south')
            if self.exists(x + 1, y) and not self.cells[x + 1][y].visited:
                good_adj_cells.append('east')
            if self.exists(x - 1, y) and not self.cells[x - 1][y].visited:
                good_adj_cells.append('west')

            if good_adj_cells:
                go = random.choice(good_adj_cells)
                if go == 'north':
                    self.cells[x][y].north = False
                    self.cells[x][y - 1].south = False
                    self.cells[x][y - 1].visited = True
                    path.append((x, y - 1))
                if go == 'south':
                    self.cells[x][y].south = False
                    self.cells[x][y + 1].north = False
                    self.cells[x][y + 1].visited = True
                    path.append((x, y + 1))
                if go == 'east':
                    self.cells[x][y].east = False
                    self.cells[x + 1][y].west = False
                    self.cells[x + 1][y].visited = True
                    path.append((x + 1, y))
                if go == 'west':
                    self.cells[x][y].west = False
                    self.cells[x - 1][y].east = False
                    self.cells[x - 1][y].visited = True
                    path.append((x - 1, y))
            else:
                path.pop()

    def exists(self, x, y):
        if x < 0 or x > self.width - 1 or y < 0 or y > self.height - 1:
            return False
        return True

    def get_direction(self, direction, x, y):
        if direction == 'north':
            return x, y - 1
        if direction == 'south':
            return x, y + 1
        if direction == 'east':
            return x + 1, y
        if direction == 'west':
            return x - 1, y

    def draw(self):
        canvas_width, canvas_height = self.cell_width * self.width, self.cell_width * self.height
        im = Image.new('RGB', (canvas_width, canvas_height))
        draw = ImageDraw.Draw(im)

        for x in range(self.width):
            for y in range(self.height):
                if self.cells[x][y].north:
                    draw.line(
                        (x * self.cell_width, y * self.cell_width, (x + 1) * self.cell_width, y * self.cell_width))
                if self.cells[x][y].south:
                    draw.line((x * self.cell_width, (y + 1) * self.cell_width, (x + 1) * self.cell_width,
                               (y + 1) * self.cell_width))
                if self.cells[x][y].east:
                    draw.line(((x + 1) * self.cell_width, y * self.cell_width, (x + 1) * self.cell_width,
                               (y + 1) * self.cell_width))
                if self.cells[x][y].west:
                    draw.line(
                        (x * self.cell_width, y * self.cell_width, x * self.cell_width, (y + 1) * self.cell_width))

        im.show()

    def to_matrix(self):
        matrix = [[0 for _ in range(self.width * 2 + 1)] for _ in range(self.height * 2 + 1)]

        for x in range(self.width):
            for y in range(self.height):
                cell_x, cell_y = x * 2 + 1, y * 2 + 1
                matrix[cell_y][cell_x] = 1  # Mark the cell as path

                if not self.cells[x][y].north and y > 0:
                    matrix[cell_y - 1][cell_x] = 1  # Remove north wall
                if not self.cells[x][y].south and y < self.height - 1:
                    matrix[cell_y + 1][cell_x] = 1  # Remove south wall
                if not self.cells[x][y].east and x < self.width - 1:
                    matrix[cell_y][cell_x + 1] = 1  # Remove east wall
                if not self.cells[x][y].west and x > 0:
                    matrix[cell_y][cell_x - 1] = 1  # Remove west wall

        # Add the outer walls
        for i in range(self.width * 2 + 1):
            matrix[0][i] = 0  # Top wall
            matrix[self.height * 2][i] = 0  # Bottom wall

        for i in range(self.height * 2 + 1):
            matrix[i][0] = 0  # Left wall
            matrix[i][self.width * 2] = 0  # Right wall

        # tarnsform matrix
        self.matOrigin = matrix = np.array(matrix)[1:-1, 1:-1]
        # print(self.matOrigin)

        matrix = np.array([GridCell(i, j, matrix[i, j]) for i, row in enumerate(matrix) for j, col in enumerate(row)])
        matrix = matrix.reshape(2 * self.width - 1, 2 * self.height - 1)

        return matrix


def main():
    width = 10
    height = 10

    isSearchAlgo = False  # switch for searching algorithms: DFS, BFS, A*
    isMdp = True # switch for MDP: value iteration, policy iterations

    if isSearchAlgo:
        timeComplex = defaultdict(list)  # record execute time
        spaceComplex = defaultdict(list)  # record memory consumption

        # execute 5 times to get average
        for i in range(5):
            maze = Maze(width=width, height=height)
            maze_bfs = copy.deepcopy(maze)
            maze_dfs = copy.deepcopy(maze)
            maze_AStar = copy.deepcopy(maze)

            # BFS
            tracemalloc.start()
            ts = time.perf_counter()
            pBFS = BFS(maze_bfs.mat)
            te = time.perf_counter()
            t = te - ts  # get execution time
            curMemo, peakMemo = tracemalloc.get_traced_memory()  # get the current memory usage

            timeComplex['BFS'].append(t)
            spaceComplex['BFS'].append((curMemo, peakMemo))
            plotPath(maze_bfs.matOrigin, pBFS, title='BFS')

            # DFS
            tracemalloc.start()
            ts = time.perf_counter()
            pDFS = DFS(maze_dfs.mat)
            te = time.perf_counter()
            t = te - ts
            curMemo, peakMemo = tracemalloc.get_traced_memory()  # get the current memory usage

            timeComplex['DFS'].append(t)
            spaceComplex['DFS'].append((curMemo, peakMemo))
            plotPath(maze_dfs.matOrigin, pDFS, title='DFS')

            # A Star
            tracemalloc.start()
            ts = time.perf_counter()
            pAStar = AStar(maze_AStar.mat)
            te = time.perf_counter()
            t = te - ts
            curMemo, peakMemo = tracemalloc.get_traced_memory()  # get the current memory usage

            timeComplex['AStar'].append(t)
            spaceComplex['AStar'].append((curMemo, peakMemo))
            plotPath(maze_AStar.matOrigin, pAStar, title='A Star')

        printTimeSpaceInfo(timeComplex, spaceComplex)
        # compute average time and space complexity
        # for k, timeList in timeComplex.items():
        #     print(f'{k} time: {sum(timeList) / len(timeList)}')
        #
        # print()
        #
        # for k, spaceList in spaceComplex.items():
        #     sum_curMemo = 0
        #     sum_peakMemo = 0
        #     for curMemo, peakMemo in spaceList:
        #         sum_curMemo += curMemo
        #         sum_peakMemo += peakMemo
        #
        #     averMemo = sum_curMemo / len(spaceList) / 10 ** 6
        #     averPeak = sum_peakMemo / len(spaceList) / 10 ** 6
        #
        #     print(f'{k} current memo: {averMemo} MB. \n{k} peak memo: {averPeak} MB\n')

    if isMdp:
        timeComplex = defaultdict(list)  # record execute time
        spaceComplex = defaultdict(list)  # record memory consumption

        for i in range(5):
            maze = Maze(width=width, height=height)
            maze_MDP_VI = copy.deepcopy(maze)
            maze_MDP_PI = copy.deepcopy(maze)

            # MDP Value Iteration
            tracemalloc.start()
            ts = time.perf_counter()
            pMDP_VI = MDP_VI(maze_MDP_VI.mat)
            te = time.perf_counter()
            t = te - ts
            curMemo, peakMemo = tracemalloc.get_traced_memory()  # get the current memory usage
            timeComplex['MDP_VI'].append(t)
            spaceComplex['MDP_VI'].append((curMemo, peakMemo))
            plot_policy_on_maze(maze_MDP_VI.matOrigin, pMDP_VI, title='MDP-Value Iteration')

            # MDP Policy Iteration
            tracemalloc.start()
            ts = time.perf_counter()
            pMDP_PI = MDP_PI(maze_MDP_PI.mat)
            te = time.perf_counter()
            t = te - ts
            curMemo, peakMemo = tracemalloc.get_traced_memory()  # get the current memory usage
            timeComplex['MDP_PI'].append(t)
            spaceComplex['MDP_PI'].append((curMemo, peakMemo))
            plot_policy_on_maze(maze_MDP_PI.matOrigin, pMDP_PI, title='MDP-Policy Iteration')

        printTimeSpaceInfo(timeComplex, spaceComplex)


def printTimeSpaceInfo(timeComplex, spaceComplex):
    # compute average time and space complexity
    for k, timeList in timeComplex.items():
        print(f'{k} time: {sum(timeList) / len(timeList)}')

    print()

    for k, spaceList in spaceComplex.items():
        sum_curMemo = 0
        sum_peakMemo = 0
        for curMemo, peakMemo in spaceList:
            sum_curMemo += curMemo
            sum_peakMemo += peakMemo

        averMemo = sum_curMemo / len(spaceList) / 10 ** 6
        averPeak = sum_peakMemo / len(spaceList) / 10 ** 6

        print(f'{k} current memo: {averMemo} MB. \n{k} peak memo: {averPeak} MB\n')


if __name__ == '__main__':
    main()
