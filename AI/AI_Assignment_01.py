import numpy as np
import random
import matplotlib.pyplot as plt

import random
import sys
from PIL import Image, ImageDraw

"""
MazeGenerator is referring to online open source: https://github.com/johnsliao/python-maze-generator
I (Hanwen Liang) have added a transform function for transferring elements of original matrix into GridCell elements 
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
        matrix = np.array(matrix)[1:-1, 1:-1]
        print(matrix)

        matrix = np.array([GridCell(i, j, matrix[i, j]) for i, row in enumerate(matrix) for j, col in enumerate(row)])
        matrix = matrix.reshape(2 * self.width - 1, 2 * self.height - 1)

        return matrix


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


def DFS(maze):
    """
    @ function DFS: Depth-First Search for maze solver
    @param maze: 2D numpy array with maze cell values: 1 for path, 0 for wall, each cell is GridCell type
    @return: True/False
    """
    h = maze.shape[0]  # real height
    w = maze.shape[1]  # real width

    # set start point and end point
    start = (0, 0)
    end = (h - 1, w - 1)

    # moving directions
    dirs = [
        (0, 1),  # right
        (1, 0),  # down
        (0, -1),  # left
        (-1, 0)  # up
    ]

    # explore as deep as possible
    stack = [start]
    while stack:
        x, y = stack.pop()
        if (x, y) == end: return True

        curCell = maze[x, y]
        if not curCell.visited:
            curCell.visited = True

            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w and maze[nx, ny].val == 1:
                    stack.append((nx, ny))

    return False


def main():
    width = 10
    height = 10
    maze = Maze(width=width, height=height).mat
    ret = DFS(maze)
    print(ret)


if __name__ == '__main__':
    main()
