import networkx as nx
import matplotlib.pyplot as plt

# Создаем неориентированный граф
G = nx.Graph()

# Добавляем вершины от v1 до v10
nodes = [f"v{i}" for i in range(1, 11)]
G.add_nodes_from(nodes)

# Создаем список ребер, замыкая их в цикл
edges = [(f"v{i}", f"v{i+1}") for i in range(1, 10)]
edges.append(("v10", "v1"))  # Замыкающее ребро e10
G.add_edges_from(edges)

# Используем круговое расположение вершин для наглядности цикла
pos = nx.circular_layout(G)

# Настраиваем размер окна
plt.figure(figsize=(8, 8))

# Рисуем вершины и сами линии ребер
nx.draw(
    G, pos,
    with_labels=True,
    node_color='skyblue',
    node_size=1200,
    font_size=12,
    font_weight='bold',
    edge_color='gray',
    width=2
)

# Добавляем подписи для ребер (e1, e2 ... e10)
edge_labels = {}
for i, (u, v) in enumerate(edges):
    edge_labels[(u, v)] = f"e{i+1}"

nx.draw_networkx_edge_labels(
    G, pos,
    edge_labels=edge_labels,
    font_color='red',
    font_size=10
)

plt.title("Циклический граф (10 вершин, 10 ребер)", fontsize=14)
plt.axis('off') # Скрываем оси координат
plt.show()