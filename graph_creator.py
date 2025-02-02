import matplotlib.pyplot as plt
import numpy as np
import json
import os
from sklearn.manifold import MDS
import argparse
import sys

def ensure_symmetric(matrix):
    matrix = np.array(matrix)
    symmetric_matrix = (matrix + matrix.T) / 2
    return symmetric_matrix

def plot_couriers_routes(instance_file, ax, SOL, model_name):
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
            
            # Ensure the model_name exists in the JSON file
            if model_name not in solution_data:
                print(f"Model '{model_name}' not found in the JSON file.")
                return
            
            solution = solution_data[model_name]  # Get the specified model solution
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
                label=f'Courier {courier_id + 1}')

    # Create a more detailed subtitle for each instance plot
    subtitle = (
        f'Instance {instance_number}\n'
        f'Implementation: {SOL}, Model: {model_name}\n'
        f'Objective (Max Distance): {solution["obj"]}'
    )
    ax.set_title(subtitle, fontsize=10)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')

def visualize(instance_range, SOL, model_name):
    # Determine the grid size for subplots
    n_instances = len(instance_range)
    # Change to 4 columns instead of 3 for a more horizontal layout
    n_cols = 3
    n_rows = (n_instances + n_cols - 1) // n_cols

    # Increase width relative to height for a more horizontal layout
    # Increased width from 15 to 20, and adjusted height ratio
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 20))
    
    # Create a more detailed title with implementation, model, and instance range
    title = (
        f'Multiple Couriers Planning (MCP) Problem\n'
        f'Implementation: {SOL}\n'
        f'Model: {model_name}\n'
        f'Instances: {instance_range.start} to {instance_range.stop - 1}'
    )
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.99)

    # Flatten the array of axes for easy indexing
    axs = axs.flatten()

    # Loop through the instances and plot each one
    for i, instance_number in enumerate(instance_range):
        instance_file = f'instances/inst{instance_number:02d}.dat'
        plot_couriers_routes(instance_file, axs[i], SOL, model_name)

    # Hide any empty subplots if there are any
    for j in range(i + 1, len(axs)):
        axs[j].set_visible(False)

    # Adjust layout and save the plot
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the rect to fit the title
    plt.subplots_adjust(top=0.90, wspace=0.4, hspace=0.4)  # Increased spacing between plots
    
    # Create output directory if it doesn't exist
    os.makedirs(f'graphs/{SOL}', exist_ok=True)
    
    # Save with more descriptive filename
    output_filename = (
        f'graphs/{SOL}/graph_{SOL.lower()}_{model_name}_'
        f'inst{min(instance_range):02d}_to_{max(instance_range):02d}.png'
    )
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    plt.show()
    return plt

def validate_args(implementation, start_instance, end_instance):
    """Validate command line arguments."""
    valid_implementations = ['cp', 'sat', 'smt', 'mip']
    if implementation.lower() not in valid_implementations:
        print(f"Error: Implementation must be one of {valid_implementations}")
        sys.exit(1)
    
    if not (1 <= start_instance <= 21 and 1 <= end_instance <= 21):
        print("Error: Instance numbers must be between 1 and 21")
        sys.exit(1)
        
    if start_instance > end_instance:
        print("Error: Start instance must be less than or equal to end instance")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate graphs for Multiple Couriers Planning solutions')
    parser.add_argument('implementation', type=str, help='Implementation type (cp/sat/smt/mip)')
    parser.add_argument('start_instance', type=int, help='Starting instance number (1-21)')
    parser.add_argument('end_instance', type=int, help='Ending instance number (1-21)')
    parser.add_argument('model', type=str, help='Model name (e.g., LNS_SYB, gurobipy)')
    
    args = parser.parse_args()
    
    # Validate arguments
    validate_args(args.implementation, args.start_instance, args.end_instance)
    
    # Create instance range
    instance_range = range(args.start_instance, args.end_instance + 1)
    
    # Convert implementation to uppercase for directory matching
    implementation = args.implementation.upper()
    
    # Create visualization
    visualize(instance_range, implementation, args.model)
