#!/usr/bin/env python
# coding: utf-8

# In[3]:


import matplotlib.pyplot as plt
import numpy as np
import time
from matplotlib.patches import Circle

class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

def get_path(cur_node, maze):
    path = []
    no_row, no_col = np.shape(maze)
    result = [[-1 for _ in range(no_col)] for _ in range(no_row)]

    current = cur_node

    while current is not None:
        path.append(current.position)
        current = current.parent

    path = path[::-1]
    start_value = 0

    for i in range(len(path)):
        result[path[i][0]][path[i][1]] = start_value
        start_value += 1

    return result

def search_A_Star(maze, start, end, cost):
    start_time = time.perf_counter()
    start_node = Node(None, tuple(start))
    start_node.f = start_node.g = start_node.h = 0
    end_node = Node(None, tuple(end))
    end_node.f = end_node.g = end_node.h = 0
    to_visit = []
    visited_nodes = []
    to_visit.append(start_node)
    outer_iterations = 0
    max_iterations = (len(maze) // 2) ** 10

    moves = [
        [-1,  0],  # Up
        [ 0, -1],  # Left
        [ 1,  0],  # Down
        [ 0,  1]   # Right
    ]

    no_row, no_col = np.shape(maze)

    while len(to_visit) > 0:
        outer_iterations += 1
        current_node = to_visit[0]
        current_index = 0

        for index, item in enumerate(to_visit):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        if outer_iterations > max_iterations:
            print("Too many iterations - no path found")
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            print(f"Execution time: {execution_time:.6f} seconds")
            return None

        to_visit.pop(current_index)
        visited_nodes.append(current_node)

        if current_node == end_node:
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            print(f"Execution time: {execution_time:.6f} seconds")
            return get_path(current_node, maze), execution_time 

        children = []

        for new_move in moves:
            node_position = (current_node.position[0] + new_move[0],
                             current_node.position[1] + new_move[1])

            if (node_position[0] > (no_row - 1) or
                node_position[0] < 0 or
                node_position[1] > (no_col - 1) or
                node_position[1] < 0):
                continue

            if maze[node_position[0]][node_position[1]] != 0:
                continue

            new_node = Node(current_node, node_position)
            children.append(new_node)

        for child in children:
            if len([visited_child for visited_child in visited_nodes if visited_child == child]) > 0:
                continue

            child.g = current_node.g + cost
            child.h = (((child.position[0] - end_node.position[0]) ** 2) +
                       ((child.position[1] - end_node.position[1]) ** 2))
            child.f = child.g + child.h

            if len([i for i in to_visit if child == i and child.g > i.g]) > 0:
                continue

            to_visit.append(child)

    end_time = time.perf_counter()  # Вимір часу закінчення виконання
    execution_time = end_time - start_time  # Обчислення тривалості виконання

    print(f"Execution time: {execution_time:.6f} seconds")  # Виведення часу в форматі з шістьма десятковими знаками

    return None

###############################################################################################################################

def visualize_maze_with_path(maze, path, execution_time):
    SQUARE_SIZE = 1
    cmap = plt.cm.get_cmap("Pastel1")
    no_row, no_col = np.shape(maze)

    fig, ax = plt.subplots(figsize=(no_col, no_row))
    ax.set_xlim([0, no_col])
    ax.set_ylim([0, no_row])
    ax.set_aspect("equal")

    for row in range(no_row):
        for col in range(no_col):
            if maze[row][col] == 1:
                ax.add_patch(plt.Rectangle((col, row), SQUARE_SIZE, SQUARE_SIZE, facecolor="grey"))
            elif path[row][col] >= 0:
                ax.add_patch(plt.Rectangle((col, row), SQUARE_SIZE, SQUARE_SIZE, facecolor="yellow"))

    start_circle = Circle((start[1] + 0.5, start[0] + 0.5), 0.4, edgecolor="green", facecolor="green")
    ax.add_patch(start_circle)

    end_circle = Circle((end[1] + 0.5, end[0] + 0.5), 0.4, edgecolor="red", facecolor="red")
    ax.add_patch(end_circle)

    plt.title("A* Pathfinding Visualization")
    plt.text(0, -1, f"Execution time: {execution_time:.6f} seconds", fontsize=10)

    plt.show()

maze = [
    [0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0]
]

start = [0, 0]
end = [5, 5]
cost = 1

path, execution_time = search_A_Star(maze, start, end, cost)

print("\nA* path:")
for row in path:
    for element in row:
        print("{:3d}".format(element), end=" ")
    print()

print("\nStart maze:")
for row in maze:
    for element in row:
        print("{:3d}".format(element), end=" ")
    print()

visualize_maze_with_path(maze, path, execution_time)


# In[5]:


import time
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

    def __eq__(self, other):
        return self.position == other.position

    def __hash__(self):
        return hash(self.position)


def reconstruct_path(node, maze):
    path = []
    no_row, no_col = np.shape(maze)
    result = [[-1 for _ in range(no_col)] for _ in range(no_row)]
    current = node

    while current is not None:
        path.append(current.position)
        current = current.parent
    path = list(reversed(path))

    for i, position in enumerate(path):
        result[position[0]][position[1]] = i

    return result


def search_BFS(maze, start, end):
    start_time = time.perf_counter()

    start_node = Node(None, tuple(start))
    end_node = Node(None, tuple(end))
    queue = deque()
    visited_nodes = set()
    queue.append(start_node)
    outer_iterations = 0
    max_iterations = (len(maze) // 2) ** 10
    move = [
        [-1,  0],
        [ 0, -1],
        [ 1,  0],
        [ 0,  1]
    ]

    no_row, no_col = np.shape(maze)

    while queue:
        outer_iterations += 1
        cur_node = queue.popleft()
        visited_nodes.add(cur_node)

        if cur_node == end_node:
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            print(f"Execution time: {execution_time:.6f} seconds")
            return reconstruct_path(cur_node, maze), execution_time

        if outer_iterations > max_iterations:
            print("Занадто багато ітерацій - шлях не знайдено")
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            print(f"Execution time: {execution_time:.6f} seconds")
            return None

        for new_position in move:
            node_position = (cur_node.position[0] + new_position[0], cur_node.position[1] + new_position[1])

            if node_position[0] not in range(no_row) or node_position[1] not in range(no_col):
                continue

            if maze[node_position[0]][node_position[1]] != 0:
                continue

            new_node = Node(cur_node, node_position)

            if new_node in visited_nodes:
                continue

            queue.append(new_node)

    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.6f} seconds")
    return None

def visualize_maze_with_path(maze, path, execution_time):
    SQUARE_SIZE = 1
    cmap = plt.cm.get_cmap("Pastel1")
    no_row, no_col = np.shape(maze)

    fig, ax = plt.subplots(figsize=(no_col, no_row))
    ax.set_xlim([0, no_col])
    ax.set_ylim([0, no_row])
    ax.set_aspect("equal")

    for row in range(no_row):
        for col in range(no_col):
            if maze[row][col] == 1:
                ax.add_patch(plt.Rectangle((col, row), SQUARE_SIZE, SQUARE_SIZE, facecolor="grey"))
            elif path[row][col] >= 0:
                ax.add_patch(plt.Rectangle((col, row), SQUARE_SIZE, SQUARE_SIZE, facecolor="yellow"))

    start_circle = Circle((start[1] + 0.5, start[0] + 0.5), 0.4, edgecolor="green", facecolor="green")
    ax.add_patch(start_circle)

    end_circle = Circle((end[1] + 0.5, end[0] + 0.5), 0.4, edgecolor="red", facecolor="red")
    ax.add_patch(end_circle)

    plt.title("BFS Pathfinding Visualization")
    plt.text(0, -1, f"Execution time: {execution_time:.6f} seconds", fontsize=10)

    plt.show()



maze = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0]
])

start = [0, 0]
end = [5, 5]

result = search_BFS(maze, start, end)

if result is not None:
    path, execution_time = result

    print("\nBFS path:")
    for row in path:
        for element in row:
            print("{:3d}".format(element), end=" ")
        print()

    print("\nStart maze:")
    for row in maze:
        for element in row:
            print("{:3d}".format(element), end=" ")
        print()

    visualize_maze_with_path(maze, path, execution_time)


# In[ ]:




