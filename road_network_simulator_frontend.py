import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button, RadioButtons, CheckButtons
import numpy as np
import time
import heapq
import math
from matplotlib.backend_bases import MouseEvent

class RoadNetworkSimulator:
    def __init__(self):
        self.G = nx.Graph()
        self.pos = {}
        self.source_node = None
        self.destination_node = None
        self.selected_algorithm = "Dijkstra"
        self.traffic_applied = False
        self.path_colors = {
            "Dijkstra": "red",
            "Bellman-Ford": "green",
            "A* Search": "blue"
        }
        self.paths = {
            "Dijkstra": [],
            "Bellman-Ford": [],
            "A* Search": []
        }
        self.execution_times = {
            "Dijkstra": 0,
            "Bellman-Ford": 0,
            "A* Search": 0
        }
        self.path_lengths = {
            "Dijkstra": 0,
            "Bellman-Ford": 0,
            "A* Search": 0
        }
        
        self.create_graph()
        self.setup_ui()
        
    def create_graph(self):
        # Create a grid-like road network
        rows, cols = 5, 5
        
        # Create nodes
        node_id = 1
        for i in range(rows):
            for j in range(cols):
                self.G.add_node(node_id)
                self.pos[node_id] = (j, -i)  # Position for visualization
                node_id += 1
        
        # Connect nodes horizontally and vertically with random weights
        for i in range(rows):
            for j in range(cols):
                current_node = i * cols + j + 1
                
                # Connect to right neighbor
                if j < cols - 1:
                    right_node = current_node + 1
                    weight = np.random.randint(1, 10)
                    self.G.add_edge(current_node, right_node, weight=weight, original_weight=weight)
                
                # Connect to bottom neighbor
                if i < rows - 1:
                    bottom_node = current_node + cols
                    weight = np.random.randint(1, 10)
                    self.G.add_edge(current_node, bottom_node, weight=weight, original_weight=weight)
        
        # Add some diagonal connections for more interesting paths
        for i in range(rows - 1):
            for j in range(cols - 1):
                current_node = i * cols + j + 1
                diagonal_node = (i + 1) * cols + (j + 1) + 1
                if np.random.random() < 0.3:  # 30% chance to add a diagonal edge
                    weight = np.random.randint(1, 15)
                    self.G.add_edge(current_node, diagonal_node, weight=weight, original_weight=weight)
    
    def setup_ui(self):
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(left=0.1, bottom=0.2, right=0.7)
        
        # Create buttons
        self.apply_traffic_button_ax = plt.axes([0.1, 0.05, 0.2, 0.05])
        self.apply_traffic_button = Button(self.apply_traffic_button_ax, 'Apply Traffic')
        self.apply_traffic_button.on_clicked(self.apply_traffic)
        
        self.reset_traffic_button_ax = plt.axes([0.35, 0.05, 0.2, 0.05])
        self.reset_traffic_button = Button(self.reset_traffic_button_ax, 'Reset Traffic')
        self.reset_traffic_button.on_clicked(self.reset_traffic)
        
        self.find_path_button_ax = plt.axes([0.6, 0.05, 0.2, 0.05])
        self.find_path_button = Button(self.find_path_button_ax, 'Find Path')
        self.find_path_button.on_clicked(self.find_path)
        
        # Algorithm selection radio buttons
        self.algorithm_radio_ax = plt.axes([0.75, 0.5, 0.2, 0.15])
        self.algorithm_radio = RadioButtons(
            self.algorithm_radio_ax, 
            ('Dijkstra', 'Bellman-Ford', 'A* Search'),
            active=0
        )
        self.algorithm_radio.on_clicked(self.set_algorithm)
        
        # Results text area
        self.results_ax = plt.axes([0.75, 0.2, 0.2, 0.25])
        self.results_ax.axis('off')
        
        # Connect click event for node selection
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        self.draw_graph()
        
    def draw_graph(self):
        self.ax.clear()
        
        # Draw edges with weights
        for u, v, data in self.G.edges(data=True):
            x1, y1 = self.pos[u]
            x2, y2 = self.pos[v]
            self.ax.plot([x1, x2], [y1, y2], 'k-', linewidth=1.5, alpha=0.7)
            
            # Display edge weight
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            self.ax.text(mid_x, mid_y, str(data['weight']), fontsize=8, 
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        # Draw paths for each algorithm
        for algo, path in self.paths.items():
            if path:
                path_edges = list(zip(path[:-1], path[1:]))
                color = self.path_colors[algo]
                for u, v in path_edges:
                    x1, y1 = self.pos[u]
                    x2, y2 = self.pos[v]
                    self.ax.plot([x1, x2], [y1, y2], color=color, linewidth=3, alpha=0.7)
        
        # Draw nodes
        node_colors = ['lightgray'] * len(self.G.nodes())
        
        # Highlight source and destination
        if self.source_node is not None:
            source_idx = list(self.G.nodes()).index(self.source_node)
            node_colors[source_idx] = 'green'
        
        if self.destination_node is not None:
            dest_idx = list(self.G.nodes()).index(self.destination_node)
            node_colors[dest_idx] = 'red'
        
        nx.draw_networkx_nodes(self.G, self.pos, ax=self.ax, node_color=node_colors, 
                              node_size=500, alpha=0.8)
        nx.draw_networkx_labels(self.G, self.pos, ax=self.ax)
        
        # Add legend for paths
        legend_patches = []
        for algo, color in self.path_colors.items():
            if self.paths[algo]:
                patch = mpatches.Patch(color=color, label=f"{algo}")
                legend_patches.append(patch)
        
        if legend_patches:
            self.ax.legend(handles=legend_patches, loc='upper right')
        
        # Update results text
        self.update_results_text()
        
        self.ax.set_title("Road Network Simulator")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.fig.canvas.draw_idle()
    
    def update_results_text(self):
        self.results_ax.clear()
        self.results_ax.axis('off')
        
        text = "Results:\n\n"
        for algo in self.paths.keys():
            if self.paths[algo]:
                text += f"{algo}:\n"
                text += f"  Path length: {self.path_lengths[algo]:.2f}\n"
                text += f"  Time: {self.execution_times[algo]:.6f} sec\n\n"
        
        self.results_ax.text(0, 1, text, verticalalignment='top')
    
    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        
        # Find the closest node to the click
        click_pos = (event.xdata, event.ydata)
        closest_node = None
        min_dist = float('inf')
        
        for node, pos in self.pos.items():
            dist = math.sqrt((pos[0] - click_pos[0])**2 + (pos[1] - click_pos[1])**2)
            if dist < min_dist:
                min_dist = dist
                closest_node = node
        
        if min_dist < 0.2:  # Threshold for node selection
            if self.source_node is None:
                self.source_node = closest_node
                print(f"Source node set to {closest_node}")
            elif self.destination_node is None:
                self.destination_node = closest_node
                print(f"Destination node set to {closest_node}")
            else:
                # Reset selections
                self.source_node = closest_node
                self.destination_node = None
                self.paths = {algo: [] for algo in self.paths}
                print(f"Reset. Source node set to {closest_node}")
            
            self.draw_graph()
    
    def set_algorithm(self, label):
        self.selected_algorithm = label
        print(f"Selected algorithm: {label}")
    
    def apply_traffic(self, event):
        if self.traffic_applied:
            print("Traffic already applied")
            return
        
        print("Applying traffic...")
        # Increase some edge weights to simulate traffic
        for u, v, data in self.G.edges(data=True):
            if np.random.random() < 0.3:  # 30% chance of traffic on each edge
                traffic_factor = np.random.uniform(1.5, 3.0)
                data['weight'] = int(data['original_weight'] * traffic_factor)
        
        self.traffic_applied = True
        self.paths = {algo: [] for algo in self.paths}  # Clear paths
        self.draw_graph()
    
    def reset_traffic(self, event):
        print("Resetting traffic...")
        # Reset edge weights to original values
        for u, v, data in self.G.edges(data=True):
            data['weight'] = data['original_weight']
        
        self.traffic_applied = False
        self.paths = {algo: [] for algo in self.paths}  # Clear paths
        self.draw_graph()
    
    def find_path(self, event):
        if self.source_node is None or self.destination_node is None:
            print("Please select both source and destination nodes")
            return
        
        print(f"Finding path from {self.source_node} to {self.destination_node} using {self.selected_algorithm}")
        
        # Run all algorithms for comparison
        self.run_dijkstra()
        self.run_bellman_ford()
        self.run_a_star()
        
        self.draw_graph()
    
    def run_dijkstra(self):
        start_time = time.time()
        
        # Implementation of Dijkstra's algorithm
        distances = {node: float('infinity') for node in self.G.nodes()}
        predecessors = {node: None for node in self.G.nodes()}
        distances[self.source_node] = 0
        priority_queue = [(0, self.source_node)]
        
        while priority_queue:
            current_distance, current_node = heapq.heappop(priority_queue)
            
            if current_distance > distances[current_node]:
                continue
            
            if current_node == self.destination_node:
                break
            
            for neighbor in self.G.neighbors(current_node):
                weight = self.G[current_node][neighbor]['weight']
                distance = current_distance + weight
                
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    predecessors[neighbor] = current_node
                    heapq.heappush(priority_queue, (distance, neighbor))
        
        # Reconstruct path
        path = []
        current = self.destination_node
        while current is not None:
            path.append(current)
            current = predecessors[current]
        path.reverse()
        
        end_time = time.time()
        
        self.paths["Dijkstra"] = path
        self.execution_times["Dijkstra"] = end_time - start_time
        self.path_lengths["Dijkstra"] = distances[self.destination_node]
    
    def run_bellman_ford(self):
        start_time = time.time()
        
        # Implementation of Bellman-Ford algorithm
        distances = {node: float('infinity') for node in self.G.nodes()}
        predecessors = {node: None for node in self.G.nodes()}
        distances[self.source_node] = 0
        
        # Relax edges |V| - 1 times
        for _ in range(len(self.G.nodes()) - 1):
            for u, v, data in self.G.edges(data=True):
                weight = data['weight']
                if distances[u] + weight < distances[v]:
                    distances[v] = distances[u] + weight
                    predecessors[v] = u
                if distances[v] + weight < distances[u]:
                    distances[u] = distances[v] + weight
                    predecessors[u] = v
        
        # Reconstruct path
        path = []
        current = self.destination_node
        while current is not None:
            path.append(current)
            current = predecessors[current]
        path.reverse()
        
        end_time = time.time()
        
        self.paths["Bellman-Ford"] = path
        self.execution_times["Bellman-Ford"] = end_time - start_time
        self.path_lengths["Bellman-Ford"] = distances[self.destination_node]
    
    def run_a_star(self):
        start_time = time.time()
        
        # Implementation of A* Search algorithm
        def heuristic(node1, node2):
            # Euclidean distance heuristic
            x1, y1 = self.pos[node1]
            x2, y2 = self.pos[node2]
            return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        open_set = [(0, self.source_node)]  # Priority queue of (f_score, node)
        closed_set = set()
        
        g_score = {node: float('infinity') for node in self.G.nodes()}
        g_score[self.source_node] = 0
        
        f_score = {node: float('infinity') for node in self.G.nodes()}
        f_score[self.source_node] = heuristic(self.source_node, self.destination_node)
        
        predecessors = {node: None for node in self.G.nodes()}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == self.destination_node:
                break
            
            if current in closed_set:
                continue
            
            closed_set.add(current)
            
            for neighbor in self.G.neighbors(current):
                if neighbor in closed_set:
                    continue
                
                tentative_g_score = g_score[current] + self.G[current][neighbor]['weight']
                
                if tentative_g_score < g_score[neighbor]:
                    predecessors[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, self.destination_node)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        # Reconstruct path
        path = []
        current = self.destination_node
        while current is not None:
            path.append(current)
            current = predecessors[current]
        path.reverse()
        
        end_time = time.time()
        
        self.paths["A* Search"] = path
        self.execution_times["A* Search"] = end_time - start_time
        self.path_lengths["A* Search"] = g_score[self.destination_node]

# Create and run the simulator
simulator = RoadNetworkSimulator()
plt.show()
