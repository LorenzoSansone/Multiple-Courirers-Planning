#
#
# Generate new .dat files to test the model
#
#

import random
import os
def generate_instance(file_path):
    # read couriers and items
    num_couriers = random.randint(2, 10)  
    num_items = random.randint(num_couriers, num_couriers + 5)  # At least as many items as couriers
    #  l_i
    capacities = [random.randint(5, 20) for i in range(num_couriers)]
    #  sizes
    sizes = [random.randint(1, 10) for i in range(num_items)]
    # distance matrix 
    distance_matrix = []
    for i in range(num_items + 1):  # n+1 rows (including the origin)
        row = []
        for j in range(num_items + 1):  # n+1 columns
            if i == j:
                row.append(0)
            else:
                row.append(random.randint(1, 10))
        distance_matrix.append(row)

    # write to file
    with open(file_path, "w") as file:
        file.write(f"{num_couriers}\n")
        file.write(f"{num_items}\n")
        file.write(" ".join(map(str, capacities)) + "\n")
        file.write(" ".join(map(str, sizes)) + "\n")
        for row in distance_matrix:
            file.write(" ".join(map(str, row)) + "\n")
# generate new instances
def generate_multiple_instances(num_instances):
    for i in range(22, 22+num_instances):
        file_name = f"instance_{i}.dat"
        file_path = os.path.join("instances", file_name)
        generate_instance(file_path)
        print(f"Generated {file_path}")
num_instances = int(input("Enter the number of instances to generate: "))
generate_multiple_instances(num_instances)
