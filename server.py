from flask import Flask, jsonify, request
import networkx as nx
import numpy as np
import time
import math
import heapq
import json

app = Flask(__name__)

# Global variables to store the graph and state
graph = None
pos = {}
source_node = None
destination_node = None
traffic_applied = False
paths = {
    "Dijkstra": [],
    "Bellman-Ford": [],
    "A* Search": []
}
execution_times = {
    "Dijkstra": 0,
    "Bellman-Ford": 0,
    "A* Search": 0
}
path_lengths = {
    "Dijkstra": 0,
    "Bellman-Ford": 0,
    "A* Search": 0
}

def create_graph():
    """Create a grid-like road network"""
    global graph, pos
    graph = nx.Graph()
    pos = {}
    
    # Create a grid-like road network
    rows, cols = 5, 5
    
    # Create nodes
    node_id = 1
    for i in range(rows):
        for j in range(cols):
            graph.add_node(node_id)
            pos[node_id] = (j, -i)  # Position for visualization
            node_id += 1
    
    # Connect nodes horizontally and vertically with random weights
    for i in range(rows):
        for j in range(cols):
            current_node = i * cols + j + 1
            
            # Connect to right neighbor
            if j < cols - 1:
                right_node = current_node + 1
                weight = np.random.randint(1, 10)
                graph.add_edge(current_node, right_node, weight=weight, original_weight=weight)
            
            # Connect to bottom neighbor
            if i < rows - 1:
                bottom_node = current_node + cols
                weight = np.random.randint(1, 10)
                graph.add_edge(current_node, bottom_node, weight=weight, original_weight=weight)
    
    # Add some diagonal connections for more interesting paths
    for i in range(rows - 1):
        for j in range(cols - 1):
            current_node = i * cols + j + 1
            diagonal_node = (i + 1) * cols + (j + 1) + 1
            if np.random.random() < 0.3:  # 30% chance to add a diagonal edge
                weight = np.random.randint(1, 15)
                graph.add_edge(current_node, diagonal_node, weight=weight, original_weight=weight)

def serialize_graph():
    """Convert the graph to a serializable format"""
    nodes = list(graph.nodes())
    edges = []
    for u, v, data in graph.edges(data=True):
        edges.append({
            'source': u,
            'target': v,
            'weight': data['weight'],
            'original_weight': data['original_weight']
        })
    
    # Convert positions dict to serializable format
    positions = {str(k): v for k, v in pos.items()}
    
    return {
        'nodes': nodes,
        'edges': edges,
        'positions': positions
    }

@app.route('/api/init', methods=['GET'])
def initialize_graph():
    """Initialize or reset the graph"""
    global graph, pos, source_node, destination_node, traffic_applied, paths, execution_times, path_lengths
    
    create_graph()
    source_node = None
    destination_node = None
    traffic_applied = False
    paths = {algo: [] for algo in paths}
    execution_times = {algo: 0 for algo in execution_times}
    path_lengths = {algo: 0 for algo in path_lengths}
    
    return jsonify({
        'status': 'success',
        'graph': serialize_graph(),
        'source_node': source_node,
        'destination_node': destination_node,
        'traffic_applied': traffic_applied
    })

@app.route('/api/graph', methods=['GET'])
def get_graph():
    """Get the current state of the graph"""
    if graph is None:
        create_graph()
    
    return jsonify({
        'graph': serialize_graph(),
        'source_node': source_node,
        'destination_node': destination_node,
        'traffic_applied': traffic_applied,
        'paths': paths,
        'execution_times': execution_times,
        'path_lengths': path_lengths
    })

@app.route('/api/set_nodes', methods=['POST'])
def set_nodes():
    """Set source and destination nodes"""
    global source_node, destination_node
    
    data = request.json
    source_node = data.get('source_node')
    destination_node = data.get('destination_node')
    
    return jsonify({
        'status': 'success',
        'source_node': source_node,
        'destination_node': destination_node
    })

@app.route('/api/apply_traffic', methods=['POST'])
def apply_traffic():
    """Apply traffic to the graph"""
    global graph, traffic_applied, paths
    
    if traffic_applied:
        return jsonify({
            'status': 'warning',
            'message': 'Traffic already applied'
        })
    
    # Increase some edge weights to simulate traffic
    for u, v, data in graph.edges(data=True):
        if np.random.random() < 0.3:  # 30% chance of traffic on each edge
            traffic_factor = np.random.uniform(1.5, 3.0)
            data['weight'] = int(data['original_weight'] * traffic_factor)
    
    traffic_applied = True
    paths = {algo: [] for algo in paths}  # Clear paths
    
    return jsonify({
        'status': 'success',
        'graph': serialize_graph(),
        'traffic_applied': traffic_applied
    })

@app.route('/api/reset_traffic', methods=['POST'])
def reset_traffic():
    """Reset traffic on the graph"""
    global graph, traffic_applied, paths
    
    # Reset edge weights to original values
    for u, v, data in graph.edges(data=True):
        data['weight'] = data['original_weight']
    
    traffic_applied = False
    paths = {algo: [] for algo in paths}  # Clear paths
    
    return jsonify({
        'status': 'success',
        'graph': serialize_graph(),
        'traffic_applied': traffic_applied
    })

def run_dijkstra():
    """Run Dijkstra's algorithm"""
    global graph, source_node, destination_node, paths, execution_times, path_lengths
    
    start_time = time.time()
    
    # Implementation of Dijkstra's algorithm
    distances = {node: float('infinity') for node in graph.nodes()}
    predecessors = {node: None for node in graph.nodes()}
    distances[source_node] = 0
    priority_queue = [(0, source_node)]
    
    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)
        
        if current_distance > distances[current_node]:
            continue
        
        if current_node == destination_node:
            break
        
        for neighbor in graph.neighbors(current_node):
            weight = graph[current_node][neighbor]['weight']
            distance = current_distance + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                predecessors[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))
    
    # Reconstruct path
    path = []
    current = destination_node
    while current is not None:
        path.append(current)
        current = predecessors[current]
    path.reverse()
    
    end_time = time.time()
    
    paths["Dijkstra"] = path
    execution_times["Dijkstra"] = end_time - start_time
    path_lengths["Dijkstra"] = distances[destination_node]

def run_bellman_ford():
    """Run Bellman-Ford algorithm"""
    global graph, source_node, destination_node, paths, execution_times, path_lengths
    
    start_time = time.time()
    
    # Implementation of Bellman-Ford algorithm
    distances = {node: float('infinity') for node in graph.nodes()}
    predecessors = {node: None for node in graph.nodes()}
    distances[source_node] = 0
    
    # Relax edges |V| - 1 times
    for _ in range(len(graph.nodes()) - 1):
        for u, v, data in graph.edges(data=True):
            weight = data['weight']
            if distances[u] + weight < distances[v]:
                distances[v] = distances[u] + weight
                predecessors[v] = u
            if distances[v] + weight < distances[u]:
                distances[u] = distances[v] + weight
                predecessors[u] = v
    
    # Reconstruct path
    path = []
    current = destination_node
    while current is not None:
        path.append(current)
        current = predecessors[current]
    path.reverse()
    
    end_time = time.time()
    
    paths["Bellman-Ford"] = path
    execution_times["Bellman-Ford"] = end_time - start_time
    path_lengths["Bellman-Ford"] = distances[destination_node]

def run_a_star():
    """Run A* Search algorithm"""
    global graph, source_node, destination_node, paths, execution_times, path_lengths, pos
    
    start_time = time.time()
    
    # Implementation of A* Search algorithm
    def heuristic(node1, node2):
        # Euclidean distance heuristic
        x1, y1 = pos[node1]
        x2, y2 = pos[node2]
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    open_set = [(0, source_node)]  # Priority queue of (f_score, node)
    closed_set = set()
    
    g_score = {node: float('infinity') for node in graph.nodes()}
    g_score[source_node] = 0
    
    f_score = {node: float('infinity') for node in graph.nodes()}
    f_score[source_node] = heuristic(source_node, destination_node)
    
    predecessors = {node: None for node in graph.nodes()}
    
    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current == destination_node:
            break
        
        if current in closed_set:
            continue
        
        closed_set.add(current)
        
        for neighbor in graph.neighbors(current):
            if neighbor in closed_set:
                continue
            
            tentative_g_score = g_score[current] + graph[current][neighbor]['weight']
            
            if tentative_g_score < g_score[neighbor]:
                predecessors[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, destination_node)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    # Reconstruct path
    path = []
    current = destination_node
    while current is not None:
        path.append(current)
        current = predecessors[current]
    path.reverse()
    
    end_time = time.time()
    
    paths["A* Search"] = path
    execution_times["A* Search"] = end_time - start_time
    path_lengths["A* Search"] = g_score[destination_node]

@app.route('/api/find_path', methods=['POST'])
def find_path():
    """Find paths using all algorithms"""
    global source_node, destination_node, paths
    
    if source_node is None or destination_node is None:
        return jsonify({
            'status': 'error',
            'message': 'Please select both source and destination nodes'
        })
    
    data = request.json
    algorithm = data.get('algorithm', 'all')
    
    if algorithm == 'Dijkstra' or algorithm == 'all':
        run_dijkstra()
    
    if algorithm == 'Bellman-Ford' or algorithm == 'all':
        run_bellman_ford()
    
    if algorithm == 'A* Search' or algorithm == 'all':
        run_a_star()
    
    return jsonify({
        'status': 'success',
        'paths': paths,
        'execution_times': execution_times,
        'path_lengths': path_lengths
    })

if __name__ == '__main__':
    create_graph()  # Initialize the graph on startup
    app.run(debug=True, port=5000)
