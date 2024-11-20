int: m;                                         % Number of couriers
int: n;                                         % Number of items
int: locations;                                 % Number of locations (n + 1 for origin)
array[1..m] of int: l;                          % Load capacities of each courier
array[1..n] of int: s;                          % Sizes of each item
array[1..locations, 1..locations] of int: D;    % Distance matrix between locations

% Variables
array[1..m, 1..n] of var 0..1: x;                       % x[i, j] = 1 if courier i delivers item j
array[1..m, 1..locations, 1..locations] of var 0..1: y; % y[i, j1, j2] = 1 if courier i travels from j1 to j2
array[1..m] of var int: Distance;                       % Total distance for each courier
var int: max_distance;                                  % Maximum distance among couriers

% Subtour elimination variables
array[1..m, 1..n] of var 1..n: u;

% Constraints
% Load constraint for each courier
constraint
    forall(i in 1..m)(
        sum(j in 1..n)(x[i, j] * s[j]) <= l[i]
    );

% Each item must be delivered by exactly one courier
constraint
    forall(j in 1..n)(
        sum(i in 1..m)(x[i, j]) = 1
    );

% Each courier starts at the origin and ends at the origin
constraint
    forall(i in 1..m)(
        sum(j in 1..n)(y[i, locations, j]) = 1 /\
        sum(j in 1..n)(y[i, j, locations]) = 1
    );

% Prevent couriers from traveling from a point to itself
constraint
    forall(i in 1..m, j in 1..locations)(
        y[i, j, j] = 0
    );

% If a courier picks up an item, they must visit and leave its location
constraint
    forall(i in 1..m, j in 1..n)(
        sum(k in 1..locations)(y[i, j, k]) = x[i, j] /\
        sum(k in 1..locations)(y[i, k, j]) = x[i, j]
    );

% Flow conservation constraints
constraint
    forall(i in 1..m, j in 2..locations)(
        sum(k in 1..locations)(y[i, k, j]) = sum(k in 1..locations)(y[i, j, k])
    );

% Subtour elimination constraints
constraint
    forall(i in 1..m, j in 1..n, k in 1..n where j != k)(
        u[i, j] - u[i, k] + (n - 1) * y[i, j, k] <= n - 2
    );

% Distance calculation
constraint
    forall(i in 1..m)(
        Distance[i] = sum(j1, j2 in 1..locations)(y[i, j1, j2] * D[j1, j2])
    );

% Objective: Minimize maximum distance
constraint
    forall(i in 1..m)(
        max_distance >= Distance[i]
    );


% Search strategy for y variables (individual assignment of values)
solve :: int_search([y[i, j1, j2] | i in 1..m, j1 in 1..locations, j2 in 1..locations], first_fail, indomain_min) minimize max_distance;
