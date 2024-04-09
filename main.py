# -------------------------------------------------
# Алгоритм Краскала для поиска минимального остова графа
# -------------------------------------------------
import numpy as np

matrix = np.loadtxt('./data/input_13.in', skiprows=1)

file = open("input_13.out", "w")

class DisjointSet:
    def __init__(self, n):
        self.parent = [i for i in range(n + 1)]
        self.rank = [0] * (n + 1)

    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]

    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu == pv:
            return False
        if self.rank[pu] < self.rank[pv]:
            self.parent[pu] = pv
        elif self.rank[pu] > self.rank[pv]:
            self.parent[pv] = pu
        else:
            self.parent[pv] = pu
            self.rank[pu] += 1
        return True

def kruskal(graph):
    n = len(graph) - 1
    dsu = DisjointSet(n)
    edges = sorted(graph)
    mst = []

    for weight, u, v in edges:
        if dsu.union(u, v):
            mst.append((weight, u, v))

    return mst

graph = [(int(value), row_idx + 1, col_idx + 1) for row_idx, row in enumerate(matrix) for col_idx, value in enumerate(row) if value != 0]
mst = kruskal(graph)

# Сумма общего веса остова
edges = [(row_idx, col_idx) for (value, row_idx, col_idx) in mst]
Summ = sum(value for (value, row_idx, col_idx) in mst)
print(Summ, file=file)

# Отрисовка матрицы смежности
result = ' '.join(['(' + ', '.join(map(str, tpl)) + ')' for tpl in edges])
num_vertices = max(max(pair) for pair in edges) + 1
adj_matrix = [[0] * num_vertices for _ in range(num_vertices)]

for edge in edges:
    start, end = edge
    adj_matrix[start][end] = 1
    adj_matrix[end][start] = 1

# Вывод матрицы смежности
for row in adj_matrix[1:]:
    print(", ".join(map(str, row[1:])), file=file)

# Вывод ребер минимального остова
for weight, u, v in mst:
    print(f"({u}, {v})", end=" ", file=file)


