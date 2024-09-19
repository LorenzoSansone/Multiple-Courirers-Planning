% Use this editor as a MiniZinc scratch book
%Define input data

include "bin_packing_capa.mzn";
include "globals.mzn";

int: m; %num couriers
int: n; %num items
array[1..m] of int: l; % maximum load size for each courier
array[1..n] of int: s; % size of each item
array[1..n+1, 1..n+1] of int: D; % distance matrix
set of int: ITEMS = 1..n;
set of int: COURIERS = 1..m;
set of int: D_SIZE = 1..n+1;
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

%path[i,k] = j means courier i goes to point j as the k-th point in the tour
array[1..m, 1..n+2] of var 1..n+1: path;


% load[i] is the total load of courier i
%array[1..m] of var 0..80: load;

% numpacks[i] is the number of packages delivered by courier i
%array[1..m] of var 0..10: numpacks;

%CONSTRAINTS

% constraint on bins
%constraint
%    forall(i in 1..m)(
%        count_eq(x, i, numpacks[i])
%    );

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
    

%C6 - Couriers can't go back to the base to take other packages
%constraint forall(i in 1..m)(
%    forall(j in 2..n+1)((path[i,j] == n+1) -> not exists(k in j..n+1)(path[i,k] != n+1))
%);
%%%

/*
int: obj_lowerbound = max(i in ITEMS)(D[n+1,i] + D[i,n+1]);
array[ITEMS] of int: max_dists = [max(j in ITEMS)(D[i,j]) | i in ITEMS];

%array[ITEMS] of int: max_dists = sort([max(j in ITEMS)(D[i,j]) | i in ITEMS]);
int: obj_upperbound = sum(i in m+1..n)(max_dists[i]) + max(j in ITEMS)(D[n+1,j]) + max(j in ITEMS)(D[j,n+1]);

var obj_lowerbound..obj_upperbound: obj = max(i in COURIERS)(sum([ D[path[i, j1],path[i, j1+1] ] | j1 in 1..n+1]));
*/

%%%
%%%%%%%% Objective function

constraint
    forall(i in 1..m) (
        y[i] = sum([ D[path[i, j1],path[i, j1+1] ] | j1 in 1..n+1])
    );

constraint cost = max(y);
  
%%%%%%%% SEARCH STRATEGY %%%%%%%%%% (Gecode on inst12)
solve :: int_search(path, first_fail, indomain_min) minimize cost; %dom_w_deg first_fail
output  [show(cost) ++ "\n"];

/*
output  [show(obj) ++ "\n"] ++
        [show(obj_lowerbound)  ++ "\n"]  ++[ show(obj_upperbound) ++ "\n"];
*/