% Use this editor as a MiniZinc scratch book
%Define input data

include "bin_packing_capa.mzn";
include "globals.mzn";
%include "../instances_dzn/inst15.dzn";


int: m; %num couriers
int: n; %num items
array[1..m] of int: l; % maximum load size for each courier
array[1..n] of int: s; % size of each item
array[1..n+1, 1..n+1] of int: D; % distance matrix

int: LB; %lower bound
int: UB; %upper bound
int: min_dist;
int: max_dist;

% Pre-processing
int: depot = n+1;
set of int: COURIERS = 1..m;
set of int: ITEMS = 1..n;
set of int: POSITIONS = 1..n+2;
set of int: NODES = 1..n+1;

% Decision Variables
array[ITEMS] of var COURIERS: x;  % courier assignments
array[COURIERS, POSITIONS] of var NODES: path;  % routing
array[COURIERS] of var min_dist..max_dist: y;  % courier distances
var LB..UB: cost;

% Implied Variables to strengthen the model
array[COURIERS] of var 0..sum(s): load;  % load per courier
array[COURIERS] of var bool: used;  % whether courier is used

% Load Calculation and Capacity Constraints
constraint forall(i in COURIERS)(
    load[i] = sum(j in ITEMS)(bool2int(x[j] = i) * s[j]) /\
    load[i] <= l[i]
);

% Courier Usage Definition
constraint forall(i in COURIERS)(
    used[i] = exists(j in ITEMS)(x[j] = i)
);

% Path Constraints
constraint forall(i in COURIERS)(
    path[i,1] = depot /\
    path[i,n+2] = depot
);

% Link Assignment and Routing
constraint forall(i in COURIERS, j in 2..n+1)(
    path[i,j] != depot -> x[path[i,j]] = i
);

% Each item delivered exactly once
constraint alldifferent_except(array1d(path), {depot});

% All assigned items must be delivered
constraint forall(j in ITEMS)(
    exists(i in COURIERS, k in 2..n+1)(
        path[i,k] = j /\ x[j] = i
    )
);

% Implied Constraints
constraint forall(i in COURIERS)(
    used[i] -> path[i,2] != depot
);

% Strong Symmetry Breaking
constraint symmetry_breaking_constraint(
    forall(i in 1..m-1)(
        load[i] >= load[i+1]
    )
);

constraint symmetry_breaking_constraint(
    forall(i in COURIERS)(
        forall(j in 2..n+1)(
            (path[i,j] = depot) -> 
            forall(k in j+1..n+1)(path[i,k] = depot)
        )
    )
);

% Distance Calculation
constraint forall(i in COURIERS)(
    y[i] = sum(j in 1..n+1)(
        D[path[i,j], path[i,j+1]]
    )
);

constraint cost = max(y);

% Objective Bounds
constraint forall(i in COURIERS)(
    used[i] -> y[i] >= 2 * min([ D[depot,j] | j in ITEMS where x[j]=i ])
);

% Search Annotation
solve 
    :: int_search([x[i] | i in ITEMS], first_fail, indomain_min)
    :: int_search([path[i,j] | i in COURIERS, j in POSITIONS], dom_w_deg, indomain_min)
    :: restart_luby(100)
    :: relax_and_reconstruct(array1d(path), 85)
minimize cost;
