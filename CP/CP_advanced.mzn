% Use this editor as a MiniZinc scratch book
%Define input data

include "bin_packing_capa.mzn";
include "globals.mzn";
include "../instances_dnz/inst01.dzn";

int: m; %num couriers
int: n; %num items
array[1..m] of int: l; % maximum load size for each courier
array[1..n] of int: s; % size of each item
array[1..n+1, 1..n+1] of int: D; % distance matrix

int: LB; %lower bound of the objective function (cost)
int: UB; %upper bound of the objective function (cost)
int: min_dist; %minimum distance that a courier can cover 
int: max_dist; %maximum distance that a courier can cover
var LB..UB: cost;

% DECISION VARIABLES
% y[i] is the total distance travelled by courier i
array[1..m] of var min_dist..max_dist: y;

% x[i] contains the index j of the courier assigned to item i. 
array[1..n] of var 1..m: x; 

%path[i,k] = j means courier i goes to point j as the k-th point in the tour
%array[1..m, 1..n+2] of var 1..n+1: path = array2d(1..m, 1..n+2, [ n+1 | i in 1..m, j in 1..n+2 ]);
%array[1..m, 1..n+2] of var 1..n+1: path;
array[1..m, 1..n+2] of var 1..n+1: path;

%CONSTRAINTS

%C1: Courier don't exceed their max load
constraint bin_packing_capa(l, x, s);

%C2: Courier deliver only the own items but it doesn't guarantee that all the packages that are assigned to a courier are delivered by him
constraint forall(i in 1..m)(
    forall (j in 2..n+1 where path[i,j] != n+1)(
      x[path[i,j]] == i
    )
);

%C3: A package (is carried by only one courier only one time) and is delivered one time
%constraint all_different(i in 1..m, j in 2..n+1 where path[i,j] != n+1)(path[i,j]); 
constraint alldifferent_except(path, {n+1});

%C4 - Couriers Start and End at the Deposit
constraint forall(i in 1..m) (
  path[i,1] = n+1 /\ path[i,n+2] = n+1
);

%C5 - Checks if all the packages that are assigned to a courier are delivered by him
constraint forall(j in 1..n)(
        exists(i in 1..n+1)(
            path[x[j], i] == j
        )
    );
/*
%C6 - If a courier has some packages to deliver he starts immediatly
constraint forall(i in 1..m)(
        exists(j in 1..n)(x[j] == i) -> path[i,2] != n+1
    );
*/
/*  
%C7 - If a courier hasn't any package to deliver every cell is set to n+1
constraint forall(i in 1..m)(
        not exists(j in 1..n)(x[j] == i) -> not exists(k in 1..n+1)(path[i,k] != n+1)
    );
*/
 
%C8 - Couriers can't go back to the base to take other packages
%constraint forall(i in 1..m)(
%    forall(j in 2..n+1)((path[i,j] == n+1) -> not exists(k in j..n+1)(path[i,k] != n+1))
%);

%%%%%%%% Objective function

constraint
    forall(i in 1..m) (
        y[i] = sum([ D[path[i, j1],path[i, j1+1] ] | j1 in 1..n+1])
    );

constraint cost = max(y);
  
%%%%%%%% SEARCH STRATEGY %%%%%%%%%%

solve :: int_search(path, dom_w_deg, indomain_min)  %dom_w_deg first_fail indomain_random indomain_min
      :: restart_luby(100)
      :: relax_and_reconstruct(array1d(path), 85)
      %:: relax_and_reconstruct(array1d(path), 85, array1d(1..(m)*(n+2), [ n+1 | i in 1..((m) * (n+2))]))
      minimize cost;
output  [show(cost) ++ "\n"] ++ [show(path) ++ "\n"];




/*
output  [show(obj) ++ "\n"] ++
        [show(obj_lowerbound)  ++ "\n"]  ++[ show(obj_upperbound) ++ "\n"];
*/