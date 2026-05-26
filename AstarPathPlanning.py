import heapq
import numpy as np


class AStarPathPlanner:
    def __init__(self, grid, allow_diagonal=True):
        self.grid = np.asarray(grid)
        if self.grid.ndim != 2:
            raise ValueError("grid must be a 2D matrix")
        self.rows, self.cols = self.grid.shape
        self.allow_diagonal = allow_diagonal

        if allow_diagonal:
            self.moves = [
                (-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1),
            ]
        else:
            self.moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def _in_bounds(self, node):
        r, c = node
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _is_free(self, node):
        return self.grid[node] == 0

    def _heuristic(self, a, b):
        dr = abs(a[0] - b[0])
        dc = abs(a[1] - b[1])
        if self.allow_diagonal:
            # Octile distance: admissible for 8-connected grids with unit/√2 costs
            return (dr + dc) + (np.sqrt(2) - 2) * min(dr, dc)
        return dr + dc

    def _neighbors(self, node):
        r, c = node
        for dr, dc in self.moves:
            nr, nc = r + dr, c + dc
            neighbor = (nr, nc)
            if not self._in_bounds(neighbor) or not self._is_free(neighbor):
                continue
            # Block diagonal squeezes between two obstacles
            if dr != 0 and dc != 0:
                if self.grid[r + dr, c] == 1 and self.grid[r, c + dc] == 1:
                    continue
            step = np.sqrt(2) if dr != 0 and dc != 0 else 1.0
            yield neighbor, step

    def find_path(self, start, goal):
        start, goal = tuple(start), tuple(goal)

        for name, node in (("start", start), ("goal", goal)): #checks if start and goal are outside grid or on obstacle and throws error
            if not self._in_bounds(node):
                raise ValueError(f"{name} {node} is out of bounds")
            if not self._is_free(node):
                raise ValueError(f"{name} {node} is on an obstacle")

        if start == goal:
            return [start], 0.0

        open_heap = []
        counter = 0
        heapq.heappush(open_heap, (self._heuristic(start, goal), counter, start))

        came_from = {}
        g_score = {start: 0.0}
        closed = set()

        while open_heap:
            _, _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            if current == goal:
                return self._reconstruct(came_from, current), g_score[current]
            closed.add(current)

            for neighbor, step_cost in self._neighbors(current):
                if neighbor in closed:
                    continue
                tentative_g = g_score[current] + step_cost
                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._heuristic(neighbor, goal)
                    counter += 1
                    heapq.heappush(open_heap, (f, counter, neighbor))

        return None, float("inf")

    def _reconstruct(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path


if __name__ == "__main__":
    grid = np.array([
        [0, 0, 0, 0, 1, 0],
        [1, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ])

    planner = AStarPathPlanner(grid, allow_diagonal=True)
    path, cost = planner.find_path((0, 0), (4, 5))
    print(f"cost: {cost:.3f}")
    print(f"path: {path}")
