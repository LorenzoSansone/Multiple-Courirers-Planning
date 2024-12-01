% Cluster-first, route-second approach with backtracking
int: m;                                          % Number of couriers
int: n;                                          % Number of items
int: locations;                                  % Number of locations (n + 1 for origin)
array[1..m] of int: l;                           % Load capacities of each courier
array[1..n] of int: s;                           % Sizes of each item
array[1..locations, 1..locations] of int: D;     % Distance matrix between locations
int: LB;     % Lower bound for max_distance
int: UB;     % Upper bound for max_distance

% Variables
array[1..m, 1..n] of var 0..1: x;               % Assignment of items to couriers
array[1..m, 1..locations, 1..locations] of var 0..1: y;  % Routes
array[1..m] of var 0..UB: Distance;             % Total distance for each courier
var LB..UB: max_distance;                       % Maximum distance among all couriers
array[1..m, 1..locations] of var 0..locations: u;  % For subtour elimination

% Previous solution tracking
array[1..m, 1..n] of 0..1: prev_x;
var bool: new_solution;

% Clustering Constraints
constraint
    forall(j in 1..n)(
        sum(i in 1..m)(x[i, j]) = 1             % Each item assigned to exactly one courier
    );

constraint
    forall(i in 1..m)(
        sum(j in 1..n)(x[i, j] * s[j]) <= l[i]  % Load capacity constraints
    );

% Symmetry breaking for clustering
constraint
    forall(i in 1..m-1)(
        sum(j in 1..n)(x[i, j]) >= sum(j in 1..n)(x[i+1, j])
    );

% Routing Constraints
% Each courier must leave the depot if they have any items
constraint
    forall(i in 1..m)(
        sum(j in 1..n)(x[i,j]) > 0 -> 
        sum(j in 1..n)(y[i, locations, j]) = 1
    );

% Each courier that leaves must return to depot
constraint
    forall(i in 1..m)(
        sum(j in 1..n)(y[i, locations, j]) = 
        sum(j in 1..n)(y[i, j, locations])
    );

% No self-loops
constraint
    forall(i in 1..m, j in 1..locations)(
        y[i, j, j] = 0
    );

% Link clustering and routing
constraint
    forall(i in 1..m, j in 1..n)(
        x[i, j] = sum(k in 1..locations where k != j)(y[i, k, j])
    );

% Flow conservation
constraint
    forall(i in 1..m, j in 1..n)(
        sum(k in 1..locations)(y[i, k, j]) = 
        sum(k in 1..locations)(y[i, j, k])
    );

% Subtour elimination
constraint
    forall(i in 1..m, j in 1..n, k in 1..n where j != k)(
        y[i, j, k] = 1 -> u[i,k] = u[i,j] + 1
    );

constraint
    forall(i in 1..m, j in 1..locations)(
        u[i, j] >= 0 /\ u[i, j] <= n
    );

% Distance calculation
constraint
    forall(i in 1..m)(
        Distance[i] = sum(j1, j2 in 1..locations)(y[i, j1, j2] * D[j1, j2])
    );

constraint
    forall(i in 1..m)(
        max_distance >= Distance[i]
    );

% New solution constraint
constraint
    new_solution -> (
        sum(i in 1..m, j in 1..n)(bool2int(x[i,j] != prev_x[i,j])) >= 1
    );

% Search strategy
solve :: seq_search([
    int_search([x[i,j] | i in 1..m, j in 1..n], input_order, indomain_min),
    int_search([y[i,j1,j2] | i in 1..m, j1,j2 in 1..locations], input_order, indomain_min),
    int_search([max_distance], input_order, indomain_min)
]) minimize max_distance;
