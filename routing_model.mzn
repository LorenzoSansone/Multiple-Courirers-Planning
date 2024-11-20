% Routing Model
int: m;                     % Number of couriers
int: n;                     % Number of items
int: locations;             % Number of locations
array[1..locations, 1..locations] of int: D;  % Distance matrix between locations
array[1..m] of int: l;      % Load capacities of each courier
array[1..n] of int: s;      % Sizes of each item

array[1..m, 1..n] of var 0..1: x;  % Assignment variables from Phase 1

array[1..m, 1..locations, 1..locations] of var 0..1: y;  % Routing variable
array[1..m] of var int: Distance;  % Total distance for each courier

% Subtour elimination variables
array[1..m, 1..n] of var 1..n: u;

% Ensure courier only visits assigned item locations
constraint forall(i in 1..m, j in 1..n)(
    x[i, j] == 1 -> exists(k in 1..locations)(
        y[i, locations, k] == 1 /\ y[i, k, locations] == 1
    )
);

% Courier starts and ends at the depot (location)
constraint forall(i in 1..m)(
    sum(j in 1..locations)(y[i, locations, j]) == 1 /\
    sum(j in 1..locations)(y[i, j, locations]) == 1
);

% Prevent self-loops
constraint forall(i in 1..m, j in 1..locations)(
    y[i, j, j] == 0
);

% Flow conservation for routes
constraint forall(i in 1..m, j in 1..locations where j != locations)(
    sum(k in 1..locations)(y[i, k, j]) == sum(k in 1..locations)(y[i, j, k])
);

% Subtour elimination constraints
constraint forall(i in 1..m, j in 1..n, k in 1..n where j != k)(
    u[i, j] - u[i, k] + (n - 1) * y[i, j, k] <= n - 2
);

% Distance calculation
constraint forall(i in 1..m)(
    Distance[i] = sum(j1, j2 in 1..locations)(y[i, j1, j2] * D[j1, j2])
);

% Maximum distance objective
var int: max_distance;
constraint forall(i in 1..m)(
    max_distance >= Distance[i]
);

solve minimize max_distance;

% Output specification
output [
    "Solution = {\"max_distance\": " ++ show(max_distance) ++ "}"
];