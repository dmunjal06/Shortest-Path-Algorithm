import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button, RadioButtons
import numpy as np
import math
import requests
import json

class RoadNetworkSimulator:
    def __init__(self):
        self.backend_url = "http://localhost:5000/api"
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
        
        # Initialize the graph from the backend
        self.initialize_graph()
        self.setup_ui()
        
    def initialize_graph(self):
        """Initialize the graph from the backend"""
        response = requests.get(f"{self.backend_url}/init")
        data = response.json()
        
        self.graph_data = data['graph']
        self.source_node = data['source_node']
        self.destination_node = data['destination_node']
        self.traffic_applied = data['traffic_applied']
        
        # Convert positions from string keys to int keys
        self.pos = {int(k): v for k, v in self.graph_data['positions'].items()}
    
    def get_graph_state(self):
        """Get the current state of the graph from the backend"""
        response = requests.get(f"{self.backend_url}/graph")
        data = response.json()
        
        self.graph_data = data['graph']
        self.source_node = data['source_node']
        self.destination_node = data['destination_node']
        self.traffic_applied = data['traffic_applied']
        self.paths = data['paths']
        self.execution_times = data['execution_times']
        self.path_lengths = data['path_lengths']
        
        # Convert positions from string keys to int keys
        self.pos = {int(k): v for k, v in self.graph_data['positions'].items()}
    
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
        for edge in self.graph_data['edges']:
            u, v = edge['source'], edge['target']
            x1, y1 = self.pos[u]
            x2, y2 = self.pos[v]
            self.ax.plot([x1, x2], [y1, y2], 'k-', linewidth=1.5, alpha=0.7)
            
            # Display edge weight
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            self.ax.text(mid_x, mid_y, str(edge['weight']), fontsize=8, 
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
        nodes = self.graph_data['nodes']
        node_colors = ['lightgray'] * len(nodes)
        
        # Highlight source and destination
        if self.source_node is not None:
            source_idx = nodes.index(self.source_node)
            node_colors[source_idx] = 'green'
        
        if self.destination_node is not None:
            dest_idx = nodes.index(self.destination_node)
            node_colors[dest_idx] = 'red'
        
        # Draw nodes
        x_coords = [self.pos[node][0] for node in nodes]
        y_coords = [self.pos[node][1] for node in nodes]
        self.ax.scatter(x_coords, y_coords, c=node_colors, s=500, alpha=0.8, zorder=10)
        
        # Add node labels
        for node in nodes:
            x, y = self.pos[node]
            self.ax.text(x, y, str(node), fontsize=10, ha='center', va='center', zorder=11)
        
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
            
            # Update the backend
            requests.post(
                f"{self.backend_url}/set_nodes",
                json={
                    'source_node': self.source_node,
                    'destination_node': self.destination_node
                }
            )
            
            self.draw_graph()
    
    def set_algorithm(self, label):
        self.selected_algorithm = label
        print(f"Selected algorithm: {label}")
    
    def apply_traffic(self, event):
        print("Applying traffic...")
        response = requests.post(f"{self.backend_url}/apply_traffic")
        data = response.json()
        
        if data['status'] == 'success':
            self.graph_data = data['graph']
            self.traffic_applied = data['traffic_applied']
            self.paths = {algo: [] for algo in self.paths}  # Clear paths
            self.draw_graph()
        else:
            print(data['message'])
    
    def reset_traffic(self, event):
        print("Resetting traffic...")
        response = requests.post(f"{self.backend_url}/reset_traffic")
        data = response.json()
        
        if data['status'] == 'success':
            self.graph_data = data['graph']
            self.traffic_applied = data['traffic_applied']
            self.paths = {algo: [] for algo in self.paths}  # Clear paths
            self.draw_graph()
    
    def find_path(self, event):
        if self.source_node is None or self.destination_node is None:
            print("Please select both source and destination nodes")
            return
        
        print(f"Finding path from {self.source_node} to {self.destination_node} using {self.selected_algorithm}")
        
        response = requests.post(
            f"{self.backend_url}/find_path",
            json={'algorithm': self.selected_algorithm}
        )
        data = response.json()
        
        if data['status'] == 'success':
            self.paths = data['paths']
            self.execution_times = data['execution_times']
            self.path_lengths = data['path_lengths']
            self.draw_graph()
        else:
            print(data['message'])

# Create and run the simulator
if __name__ == "__main__":
    simulator = RoadNetworkSimulator()
    plt.show()
