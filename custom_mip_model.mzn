int: m;  % Number of couriers
int: n;  % Number of items

array[1..m] of int: l;  % Load capacity of each courier
array[1..n] of int: s;  % Size of each item

array[1..n+1, 1..n+1] of int: D;  % Distance matrix

set of int: Locations = 1..n;  % Locations of items
set of int: Couriers = 1..m;    % Couriers

% Decision variables
% x[i,j] = 1 if courier i delivers item j, 0 otherwise
array[Couriers, Locations] of var 0..1: x;  

% Variable for the maximum distance traveled by any courier
var int: max_dist;

% Tour distance for each courier
array[Couriers] of var int: tour_distance;

% Distance matrix between locations and origin 
% (index of the origin point, since there are n items there are n locations, so the origin is the location n+1)
var int: origin = n + 1;

% Constraint: Each item must be delivered by exactly one courier
constraint forall(j in Locations) (
    sum(i in Couriers)(x[i, j]) = 1
);

% Constraint: Load capacity for each courier
% "location" and "item" are essentially different terms for the same concept 
constraint forall(i in Couriers) (
    sum(j in Locations)(s[j] * x[i, j]) <= l[i]
);

% Tour distance calculation for each courier
constraint forall(i in Couriers) ( % for each courier
    let {   
        % Distance from origin to the first item
        var int: dist_from_origin = sum(j in Locations where x[i, j] == 1)(D[origin, j]),
        % Distance from the last item to the origin
        var int: dist_to_origin = sum(j in Locations where x[i, j] == 1)(D[j, origin]),
        % Sum of distances between items
        var int: item_to_item_dist = sum(j1, j2 in Locations where j1 < j2)(
            D[j1, j2] * x[i, j1] * x[i, j2]
        )
    } in
    tour_distance[i] = dist_from_origin + dist_to_origin + item_to_item_dist
);

% Constraint: Max distance is the maximum of the tour distances
constraint max_dist = max(tour_distance);

% Objective: Minimize the maximum distance
solve minimize max_dist;
