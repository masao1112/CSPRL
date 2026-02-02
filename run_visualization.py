import os
import sys
import pickle
import numpy as np
import matplotlib
# Use Agg backend for headless server (must be before importing pyplot)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
import osmnx as ox
from matplotlib.lines import Line2D

# Add parent directory to path to allow importing env_plus and power_grid
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
# Also add parent of CSPRL for power_grid if needed
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import env_plus as ev
import evaluation_framework as ef

LOCATION = "DongDa"

def find_best_model(log_dir):
    """Find the latest/best model zip file in the directory."""
    files = [f for f in os.listdir(log_dir) if f.endswith(".zip")]
    if not files:
        raise FileNotFoundError(f"No .zip model files found in {log_dir}")
    # Sort by modification time (latest first) or by step number if possible
    # Assuming format: best_model_DongDa_STEP.zip
    files.sort(key=lambda x: os.path.getmtime(os.path.join(log_dir, x)), reverse=True)
    return os.path.join(log_dir, files[0])

def nodesize(station_list, my_graph, my_plan):
    ns = []
    for node in my_graph.nodes():
        if node not in station_list:
            ns.append(2)
        else:
            i = station_list.index(node)
            station = my_plan[i]
            try:
                capacity = station[2]["capability"]
            except KeyError:
                capacity = np.sum(ef.CHARGING_POWER * station[1])
            if capacity < 100:
                ns.append(6)
            elif 100 <= capacity < 200:
                ns.append(11)
            elif 200 <= capacity < 300:
                ns.append(16)
            else:
                ns.append(21)
    return ns

def visualise_stations(my_graph, my_plan, output_path):
    print(f"Generating plot to {output_path}...")
    station_list = [station[0][0] for station in my_plan]
    colours = ['r', 'grey']
    nc = [colours[0] if node in station_list else colours[1] for node in my_graph.nodes()]
    labels = ['Charging station', 'Normal road junction']
    legend_elements = [Line2D([0], [0], marker='o', color='w', lw=0, markerfacecolor=colours[0], markersize=7),
                       Line2D([0], [0], marker='o', color='w', lw=0, markerfacecolor=colours[1], markersize=4)]
    
    ns = nodesize(station_list, my_graph, my_plan)
    fig, ax = ox.plot_graph(my_graph, node_color=nc, save=False, node_size=ns, edge_linewidth=0.2, edge_alpha=0.8,
                            show=False, close=False)
    ax.legend(legend_elements, labels, loc=2, prop={"size": 12})
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("Plot saved successfully.")

def run_evaluation_and_viz():
    print(f"--- Starting Visualization for {LOCATION} ---")
    
    # 1. Setup Paths
    base_dir = current_dir
    graph_file = os.path.join(base_dir, "Graph", LOCATION, LOCATION + ".graphml")
    node_file = os.path.join(base_dir, "Graph", LOCATION, f"nodes_extended_{LOCATION}.txt")
    plan_file = os.path.join(base_dir, "Graph", LOCATION, f"existingplan_{LOCATION}.pkl")
    log_dir = os.path.join(base_dir, f"tmp_{LOCATION}")
    results_dir = os.path.join(base_dir, "Results", LOCATION)
    images_dir = os.path.join(base_dir, "Images", "Result_Plots", LOCATION)
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    # 2. Find Model
    try:
        model_path = find_best_model(log_dir)
        print(f"Loading model: {model_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # 3. Load Environment and Model
    env = ev.StationPlacement(graph_file, node_file, plan_file, location=LOCATION)
    env = Monitor(env, log_dir)
    model = DQN.load(model_path)

    # 4. Run Inference
    print("Running inference...")
    obs, _ = env.reset()
    done = False
    best_plan = None
    action_history = []
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action_history.append(action.item())
        obs, reward, done, info, _ = env.step(action)
        if done:
            _node_list_dummy, best_plan = env.render()
    
    # 5. Save Action Distribution (Headless friendly)
    plt.figure()
    sns.countplot(x=action_history)
    plt.title('Frequency of Chosen Actions')
    action_plot_path = os.path.join(results_dir, "action_distribution.png")
    plt.savefig(action_plot_path)
    plt.close()
    print(f"Action distribution saved to {action_plot_path}")

    # 6. Save Plan
    plan_output_path = os.path.join(results_dir, "latest_best_plan.pkl")
    with open(plan_output_path, "wb") as f:
        pickle.dump(best_plan, f)
    print(f"Plan saved to {plan_output_path}")

    # 7. Generate Map Visualization
    print("Loading graph for visualization...")
    G = ox.load_graphml(graph_file)
    viz_output_path = os.path.join(images_dir, "latest_visualization.png")
    visualise_stations(G, best_plan, viz_output_path)
    print(f"Done! Check {viz_output_path}")

if __name__ == "__main__":
    run_evaluation_and_viz()
