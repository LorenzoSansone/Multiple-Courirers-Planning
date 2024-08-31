import matplotlib.pyplot as plt
import numpy as np
import json
import os
from sklearn.manifold import MDS

def ensure_symmetric(matrix):
    matrix = np.array(matrix)
    symmetric_matrix = (matrix + matrix.T) / 2
    return symmetric_matrix

def plot_couriers_routes(instance_file):
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
    json_file_path = f'res/MIP/{json_file_name}'
    
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
            solution = json.load(file)["gurobi"]
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

    # Step 4: Create the plot
    plt.figure(figsize=(12, 12))

    # Plot the depot
    plt.scatter(*origin_coord, color='red', label='Depot (Origin)', s=100, edgecolor='black')
    plt.text(*origin_coord, 'Origin', fontsize=12, ha='right')

    # Plot the locations of items
    for i in range(n):
        plt.scatter(*coordinates[i], color='blue', s=100, edgecolor='black')
        plt.text(coordinates[i][0], coordinates[i][1], f'{i + 1}', fontsize=12, ha='right')

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
            plt.annotate(
                '', 
                xy=end, 
                xytext=start, 
                arrowprops=dict(facecolor=color, edgecolor=color, arrowstyle='->', lw=2),
                fontsize=10
            )
        
        # Plot the route line
        plt.plot(route[:, 0], route[:, 1], color=color, linestyle='-', linewidth=2, marker='o',
                 label=f'Courier {courier_id}')
        
    # Step 5: Add labels and legends
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'Couriers Routes Visualization\nObjective (Max Distance): {solution["obj"]}')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage:
instance_file = 'instances/inst22.dat'
plot_couriers_routes(instance_file)
