% Courier Assignment Model
int: m;                     % Number of couriers
int: n;                     % Number of items
array[1..m] of int: l;      % Load capacities of each courier
array[1..n] of int: s;      % Sizes of each item

% Decision variables
array[1..m, 1..n] of var 0..1: x;  % Assignment variable

% Constraint: Each item is assigned to a courier
constraint forall(j in 1..n) (
    sum(i in 1..m)(x[i, j]) == 1
);

% Constraint: Load capacity of each courier
constraint forall(i in 1..m) (
    sum(j in 1..n)(x[i, j] * s[j]) <= l[i]
);

% Ensure different items are assigned to different couriers
constraint forall(i1, i2 in 1..m where i1 < i2)(
    sum(j in 1..n)(x[i1, j] * x[i2, j]) == 0
);

% Objective: Minimize maximum load and encourage diversity
var int: max_load;
constraint max_load == max(i in 1..m)(
    sum(j in 1..n)(x[i, j] * s[j])
);

solve 
    minimize max_load;

% Output specification
output [
    "Solution = {\"assignments\": ["] ++
    [show(sum(j in 1..n)(j * x[i, j])) | i in 1..m, sep = ", "] ++
    ["], \"max_load\": " ++ show(max_load) ++ "}"
];