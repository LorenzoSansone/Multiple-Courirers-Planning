% Constants
% - Couriers m
% - Items n
% - Load capacity: l_i
% - Size of each item: s_j
% - origin is n+1

% Decision Variables
% - x[i,j] = 1 if courier i delivers item j
% - tour_distance[i] = the total distance traveled by courier i
% - max_dist = maximum tour distance among all couriers
% - route[i] = path followed by courier i, consisting of a sequence of items, meaning that the courier follows that path
% - y[i,j1,j2] = 1 if courier i travels directly from item j1 to item j2, 0 otherwise

% Constraints
% - Each item is delivered by exactly one courier
% - Load capacity for each courier
% - Path continuity and flow conservation
% - Tour distance calculation
% - max_dist is the maximum of the tour distances

% Objective
% - Minimize the total path distance
% Constants
int: m;  % Number of couriers
int: n;  % Number of items

array[1..m] of int: l;  % Load capacity of each courier
array[1..n] of int: s;  % Size of each item

array[1..n+1, 1..n+1] of int: D;  % Distance matrix D which is of dimensions (n+1) x (n+1)

set of int: Locations = 1..n;  % Locations of items
set of int: Couriers = 1..m;    % Couriers

% Decision variables
array[Couriers, Locations] of var 0..1: x;  %  x[i,j] = 1 if courier i delivers item j

array[Couriers, 1..n+1, 1..n+1] of var 0..1: y;  % y[i,j1,j2] = 1 if courier i travels directly from item j1 to j2

array[Couriers] of var int: tour_distance;  % Total distance traveled by each courier

var int: max_dist;  % Maximum tour distance among all couriers

% Index of the origin point (location n+1)
var int: origin = n + 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Constraints

% Each item must be delivered by exactly one courier
constraint forall(j in Locations) (
    sum(i in Couriers)(x[i, j]) = 1
);
    
% Load capacity for each courier
constraint forall(i in Couriers) (
    sum(j in Locations)(s[j] * x[i, j]) <= l[i]
);

% Tour distance calculation for each courier
constraint forall(i in Couriers) (
    tour_distance[i] = sum(j1 in 1..n, j2 in 1..n)(D[j1, j2] * y[i, j1, j2])
);

% Flow conservation: Each courier starts and ends at the origin
constraint forall(i in Couriers) (
    % Start at the origin
    sum(j in Locations)(y[i, origin, j]) = 1 /\
    % Return to the origin
    sum(j in Locations)(y[i, j, origin]) = 1
);

% Ensure that each courier's route is continuous and valid
constraint forall(i in Couriers, j in Locations) (
    sum(j1 in 1..n+1)(y[i, j1, j]) = x[i, j] /\
    sum(j2 in 1..n+1)(y[i, j, j2]) = x[i, j]
);

% Constraint: Max distance is the maximum of the tour distances
constraint max_dist = max(tour_distance);

% Objective: Minimize the maximum tour distance
solve minimize max_dist;
