% Use this editor as a MiniZinc scratch book
%Define input data

include "bin_packing_capa.mzn";
include "globals.mzn";
include "../instances_dnz/inst11.dzn";


int: m; %num couriers
int: n; %num items
array[1..m] of int: l; % maximum load size for each courier
array[1..n] of int: s; % size of each item
array[1..n+1, 1..n+1] of int: D; % distance matrix

int: LB; %lower bound
int: UB; %upper bound
int: min_dist;
int: max_dist;


var LB..UB: cost;

% DECISION VARIABLES
% y[i] is the total distance travelled by courier i
array[1..m] of var min_dist..max_dist: y;

% x[i] contains the index j of the courier assigned to item i. 
array[1..n] of var 1..m: x; 

% x[i, j1] = j2 means that the ith courier will go from the j1th node to the j2th node
array[1..m, 1..n+1] of var 1..n+1: path;


%%%%%%%% PREDICATES %%%%%%%%%%
predicate at_most_one(array[int] of var bool:x) =
          forall(i,j in index_set(x) where i < j)(
            (not x[i] \/ not x[j]));
predicate exactly_one(array[int] of var bool:x) =
          at_most_one(x) /\ exists(x);

%CONSTRAINTS

%C1: Courier don't exceed their max load
constraint bin_packing_capa(l, x, s);

%C2: Courier deliver only the own items but it doesn't guarantee that all the packages that are assigned to a courier are delivered by him
constraint forall(i in 1..m)(
    forall (j in 1..n+1 where path[i,j] != j /\  path[i,j] != n+1)(
      x[path[i,j]] == i
    )
);

%C3: A package (is carried by only one courier only one time) and is delivered one time
constraint
    forall(j in 1..n) (
        exactly_one(
            [path[i, j] != j | i in 1..m]
        )
    );

%C4 - Couriers Start and End at the Deposit
constraint
    forall(i in 1..m)(
        if path[i, n+1] != n+1 then count([path[i, j1] | j1 in 1..n], n+1, 1) 
        else forall(k in 1..n) (path[i, k] = k) endif
    );

%C5 - Checks if all the packages that are assigned to a courier are delivered by him
constraint forall(j in 1..n)(
        exists(i in 1..n+1)(path[x[j], i] == j) /\ path[x[j], j] != j
    );
    
%C6
constraint
    forall(i in 1..m) (
        subcircuit([path[i, j] | j in 1..n+1])
    );

%SYMMETRY
% couriers do not exceed their max load

%S1 - Couriers can't go back to the base to take other packages and starts immediatly


%S2
/*
constraint symmetry_breaking_constraint(
    forall(c1 in 1..m, c2 in 1..m where c1 < c2)
     ((l[c1] == l[c2]) -> (lex_less(row(path, c1), row(path, c2))))
);
*/
%S3
/*
constraint symmetry_breaking_constraint(
    forall(c1 in 1..m, c2 in 1..m where c1 < c2)
    (
      (max(load[c1], load[c2]) <= min(l[c1], l[c2])) ->
      (lex_less(row(path, c1), row(path,c2))))
);*/

%%%%%%%% Objective function

constraint
    forall(i in 1..m) (
        y[i] = sum([ D[j1, path[i, j1]] | j1 in 1..n+1])
    );


constraint cost = max(y);
  
%%%%%%%% SEARCH STRATEGY %%%%%%%%%% (Gecode on inst12)
solve :: int_search(path, dom_w_deg, indomain_min)  
      :: restart_luby(100)
      :: relax_and_reconstruct(array1d(path), 85)
      minimize cost;
%solve :: int_search(path, first_fail, indomain_min) minimize cost; 
