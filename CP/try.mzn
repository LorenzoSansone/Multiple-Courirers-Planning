include "globals.mzn";
include "../instances_dnz/inst11.dzn";

%%%%%%%%% DATA %%%%%%%%%%
int: m; 
int: n;   
int: time;
int : min_dist;
int : max_dist;
int: min_solution;
int : at_least_one;
array[1..m] of int: l; 
array[1..n] of int: s; 
int : min_load;
int : max_load;
array[1..n+1, 1..n+1] of int: distances;

%%%%%%%%% VARIABLES %%%%%%%%%%

% effective_max_dist is the maximum distance traveled by a courier
var min_solution..max_dist: effective_max_dist;

% x[i, j1] = j2 means that the ith courier will go from the j1th node to the j2th node
% if x[i, k] = k then the courier i is not going to visit the node k
array[1..m, 1..n+1] of var 1..n+1: x;

% y[i] is the total distance travelled by courier i
array[1..m] of var min_dist..max_dist: y;

% load[i] is the total load of courier i
array[1..m] of var min_load..max_load: load;

% numpacks[i] is the number of packages delivered by courier i
array[1..m] of var at_least_one..time: numpacks;

% bins[j] is the courier that will deliver the jth package - Tunneling approach using bin packing
array[1..n] of var 1..m : bins;

%%%%%%%% PREDICATES %%%%%%%%%%
predicate at_most_one(array[int] of var bool:x) =
          forall(i,j in index_set(x) where i < j)(
            (not x[i] \/ not x[j]));

predicate exactly_one(array[int] of var bool:x) =
          at_most_one(x) /\ exists(x);

%%%%%%%% CONSTRAINTS %%%%%%%%%%
% courier stars and ends at the depot
constraint
    forall(i in 1..m)(
        if x[i, n+1] != n+1 then count([x[i, j1] | j1 in 1..n], n+1, 1) 
        else forall(k in 1..n) (x[i, k] = k) endif
    );

% couriers do not exceed their max load
constraint
    forall(i in 1..m)(
        load[i] == sum([(if x[i, j1] == j1 then 0 else s[j1] endif) | j1 in 1..n]) /\
        load[i] <= l[i] 
    );

% there exists one path for each courier
constraint
    forall(i in 1..m) (
        subcircuit([x[i, j] | j in 1..n+1])
    );

% each package is delivered exactly once
constraint
    forall(j in 1..n) (
        exactly_one(
            [x[i, j] != j | i in 1..m]
        )
    );

% constraint
%     forall(j in 1..n) (
%         sum([x[i,j]!=j | i in 1..m]) == 1
%     );

% bins creation
constraint bin_packing_load(load, bins, s);

% link bins and x
constraint 
    forall(j in 1..n)(
        x[bins[j], j] != j
    );

% constraint on bins
constraint
    forall(i in 1..m)(
        count_eq(bins, i, numpacks[i])
    );

%%%%%%%% OBJECTIVE %%%%%%%%%%
constraint
    forall(i in 1..m) (
        y[i] = sum([ distances[j1, x[i, j1]] | j1 in 1..n+1])
    );

constraint
    effective_max_dist = max(y);

%%%%%%%% SEARCH STRATEGY %%%%%%%%%% (Gecode on inst12)
solve :: int_search(x, first_fail, indomain_min) minimize effective_max_dist; 

output [
    "\(x)"
];