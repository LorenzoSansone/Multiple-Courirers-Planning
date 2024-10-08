def read_input(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    m = int(lines[0].strip())  # number of couriers
    n = int(lines[1].strip())  # number of items
    l = list(map(int, lines[2].strip().split()))  # max load size for each courier
    s = list(map(int, lines[3].strip().split()))  # sizes of the items
    D = [list(map(int, line.strip().split())) for line in lines[4:]]  # distances
    locations = n + 1
    return m, n, l, s, D, locations

def check_load_sizes(x, s, m, n, l):
    load_sizes = [0] * m
    for courier in range(m):
        load_size = sum(x[courier][j] * s[j] for j in range(n))
        load_sizes[courier] = load_size
        print(f"Courier {courier} loaded: {load_sizes[courier]}, maximum capacity is {l[courier]}: {load_sizes[courier] <= l[courier]}")
    return load_sizes

def check_if_every_courier_starts_at_origin(y, n, m):
    origin = n+1
    for courier in range(m):
        starts_at_origin = any(y[courier][n][j] == 1 for j in range(n+1) if j != origin)
        if not starts_at_origin:
            print(f"Courier {courier} does not start at the origin.")
            return False

    return True

def check_if_every_courier_ends_at_origin(y, n, m):
    origin = n+1
    for courier in range(m):
        ends_at_origin = any(y[courier][j][n] == 1 for j in range(n+1) if j != origin-1   )
        if not ends_at_origin:
            print(f"Courier {courier} does not end at the origin.")
            return False
    return True

def picked_up_objects(x, m, n):
    for courier in range(m):
        print(f"Courier {courier} picked up objects: ")
        loaded_objects = []
        for item in range(n):
             if x[courier][item] == 1:
                loaded_objects.append(item)
        print(loaded_objects)

def distances_check(D, y_matrix):
    for courier_index in range(len(y_matrix)):
        route_distances = []
        total_distance = 0
        current_location = len(D) - 1  # Start from origin, assuming it's the last index

        while True:
            next_location = None
            for j2 in range(len(y_matrix[courier_index][current_location])):
                if y_matrix[courier_index][current_location][j2] == 1:
                    next_location = j2
                    break
            
            if next_location is None or next_location == len(D) - 1:
                break  # No further movement or return to origin

            # Calculate the distance between the current location and the next location
            distance = D[current_location][next_location]
            route_distances.append(distance)
            total_distance += distance

            # Move to the next location
            current_location = next_location

        # Add the distance back to the origin
        if current_location != len(D) - 1:
            distance_back_to_origin = D[current_location][len(D) - 1]
            route_distances.append(distance_back_to_origin)
            total_distance += distance_back_to_origin

        # Prepare the distance string
        distance_str = " + ".join(map(str, route_distances)) + f" = {total_distance}"
        output_str = f"Courier {courier_index}: {distance_str}"
        
        # Print to the console
        print(output_str)

def print_routes_from_solution(solution):
    y = solution['y']
    n = len(y[0]) - 1
    for courier_index in range(len(y)):
        route = []
        current_location = n  # Start from origin (assuming the origin is at index 'n')
        
        while True:
            route.append(current_location + 1) #correct indexing
            next_location = None
            for j2 in range(len(y[courier_index][current_location])):
                if y[courier_index][current_location][j2] == 1:
                    next_location = j2
                    break
            
            if next_location == n or next_location is None:
                break  
            else:
                current_location = next_location
        
        route.append(n + 1)
        
        print(f"Courier {courier_index}: {' -> '.join(map(str, route))}")

def get_instance_number(file_path):
    return file_path.split('/')[-1].split('.')[0].replace('inst', '')

def check_if_items_are_taken_by_couriers(x, m, n):
    for item in range(n):
        count = 0
        for courier in range(m):
            if x[courier][item] == 1:
                count += 1
        if count != 1:
            print(f"Item {item} is delivered by {count} couriers instead of exactly one.")
            return False
    return True