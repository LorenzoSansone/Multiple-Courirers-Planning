import matplotlib.pyplot as plt
import numpy as np
import json
import os
from sklearn.manifold import MDS

def ensure_symmetric(matrix):
    matrix = np.array(matrix)
    symmetric_matrix = (matrix + matrix.T) / 2
    return symmetric_matrix

def plot_couriers_routes(instance_file, ax, SOL):
    # Extract the base name and instance number from the input file name
    base_name = os.path.basename(instance_file)
    instance_number = base_name.replace('.dat', '')  # Remove the .dat suffix

    # Extract only the numeric part of the instance number
    instance_number = ''.join(filter(str.isdigit, instance_number))
    
    # Format the instance number for JSON file naming
    if int(instance_number) < 10:
        json_file_name = f'{instance_number.zfill(2)}.json'
    else:
        json_file_name = f'{instance_number}.json'
    
    # Define paths for the JSON file
    json_file_path = f'res/{SOL}/{json_file_name}'
    
    # Step 1: Read the input instance data
    with open(instance_file, 'r') as file:
        lines = file.readlines()
    m = int(lines[0].strip())  # number of couriers
    n = int(lines[1].strip())  # number of items
    l = list(map(int, lines[2].strip().split()))  # max load size for each courier
    s = list(map(int, lines[3].strip().split()))  # sizes of the items
    D = [list(map(int, line.strip().split())) for line in lines[4:]]  # distances

    # Convert distance matrix to numpy array
    D = np.array(D)
    
    # Ensure the matrix is symmetric
    D = ensure_symmetric(D)

    # Step 2: Read the solution from the JSON file
    try:
        with open(json_file_path, 'r') as file:
            solution_data = json.load(file)
            
            # Get the first key-value pair, whatever the key is
            solution = next(iter(solution_data.values()))
    except FileNotFoundError:
        print(f"JSON file {json_file_path} not found.")
        return
    except json.JSONDecodeError:
        print(f"Error decoding JSON file {json_file_path}.")
        return


    # Step 3: Apply Multidimensional Scaling (MDS) to get 2D coordinates
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    coordinates = mds.fit_transform(D)

    # The depot is at index n (the last index in coordinates)
    origin_coord = coordinates[-1]
    
    # Translate all coordinates so the origin is at (0,0)
    coordinates -= origin_coord
    origin_coord = np.array([0, 0])  # Set the origin to (0,0)

    # Step 4: Create the plot in the provided subplot axis (`ax`)

    # Plot the depot
    ax.scatter(*origin_coord, color='red', label='Depot (Origin)', s=100, edgecolor='black')
    ax.text(*origin_coord, 'Origin', fontsize=12, ha='right')

    # Plot the locations of items
    for i in range(n):
        ax.scatter(*coordinates[i], color='blue', s=100, edgecolor='black')
        ax.text(coordinates[i][0], coordinates[i][1], f'{i + 1}', fontsize=12, ha='right')

    # Plot the routes for each courier
    colors = plt.cm.get_cmap('tab10', m)  # Colormap for different couriers
    for courier_id, items_picked in enumerate(solution['sol']):
        if not items_picked:
            continue

        color = colors(courier_id)
        route = [origin_coord] + [coordinates[item - 1] for item in items_picked] + [origin_coord]
        route = np.array(route)

        # Plot the route with arrows
        for start, end in zip(route[:-1], route[1:]):
            ax.annotate(
                '', 
                xy=end, 
                xytext=start, 
                arrowprops=dict(facecolor=color, edgecolor=color, arrowstyle='->', lw=2),
                fontsize=10
            )
        
        # Plot the route line
        ax.plot(route[:, 0], route[:, 1], color=color, linestyle='-', linewidth=2, marker='o',
                label=f'Courier {courier_id}')
        
    # Step 5: Add labels and legends to the subplot
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title(f'Instance {instance_number}\nObjective (Max Distance): {solution["obj"]}')
    ax.legend()
    ax.grid(True)


def visualize(instance_range, SOL):
    # Determine the grid size for subplots
    n_instances = len(instance_range)
    n_cols = 3
    n_rows = (n_instances + n_cols - 1) // n_cols

    # Create the subplots
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    fig.suptitle(f'Multiple Couriers Planning (MCP) Problem: Couriers Routes Visualization \nfor Instances {instance_range.start} to {instance_range.stop - 1}, \nsolver: {"gurobi (Mixed-Integer Linear)" if SOL=="MIP" else "gecode" if SOL == "CP" else "SAT" }',
                 fontsize=16, fontweight='bold', y=0.98)

    # Flatten the array of axes for easy indexing
    axs = axs.flatten()

    # Loop through the instances and plot each one
    for i, instance_number in enumerate(instance_range):
        instance_file = f'instances/inst{instance_number:02d}.dat'
        plot_couriers_routes(instance_file, axs[i], SOL)

    # Hide any empty subplots if there are any
    for j in range(i + 1, len(axs)):
        axs[j].set_visible(False)

    # Adjust layout and save the plot
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the rect to fit the title
    plt.subplots_adjust(top=0.9)
    plt.savefig(f'graphs/{SOL}/graph_results_{min(instance_range)}_to_{max(instance_range)}')  # Save before showing
    plt.show()
    return plt

if __name__ == "__main__":
    # Define the range of instances you want to plot
    instance_range = range(16, 22)  # Example: plotting instances 1 to 21
    SOL = "CP"  # Adjust with the type of solution to visualize
    visualize(instance_range, SOL)
