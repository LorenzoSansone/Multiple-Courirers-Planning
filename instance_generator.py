#
#
# Generate new .dat files to test the model
#
#

import random
import os
def generate_instance(file_path):
    # read couriers and items
    num_couriers = random.randint(1, 1)  
    num_items = 20  # At least as many items as couriers
    #  l_i
    capacities = [random.randint(5, 200) for i in range(num_couriers)]
    #  sizes
    sizes = [random.randint(1, 30) for i in range(num_items)]
    # distance matrix (between 0 and 20)
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
        file_name = f"inst{i}.dat"
        file_path = os.path.join("test_instances", file_name)
        generate_instance(file_path)
        print(f"Generated {file_path}")
num_instances = int(input("Enter the number of instances to generate: "))
generate_multiple_instances(num_instances)
