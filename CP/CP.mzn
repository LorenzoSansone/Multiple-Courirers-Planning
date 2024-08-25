% Use this editor as a MiniZinc scratch book
%Define input data

include "bin_packing_capa.mzn";
include "globals.mzn";

int: m; %num couriers
int: n; %num items
array[1..m] of int: l; % maximum load size for each courier
array[1..n] of int: s; % size of each item
array[1..n+1, 1..n+1] of int: D; % distance matrix

int: LB; %lower bound
int: UB; %upper bound


var LB..UB: cost;


%%%%%%%% PREDICATES %%%%%%%%%%
predicate at_most_one(array[int] of var bool:x) =
          forall(i,j in index_set(x) where i < j)(
            (not x[i] \/ not x[j]));

predicate exactly_one(array[int] of var bool:x) =
          at_most_one(x) /\ exists(x);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% DECISION VARIABLES
% y[i] is the total distance travelled by courier i
array[1..m] of var int: y;

% x[i] contains the index j of the courier assigned to item i. 
array[1..n] of var 1..m: x; 
%array[1..m, 1..n] of var 0..1: x; % x[i,j] = 1 if courier i delivers item j, 0 otherwise


array[1..m, 1..n+2] of var 0..n+1: path; % p[i,k] = j means courier i goes to point j as the k-th point in the tour


%CONSTRAINTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Courier don't exceed their max load
constraint bin_packing_capa(l, x, s);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Courier deliver only the own items
constraint forall(k in 1..m)(
    forall(i in 1..n)(
        if x[i] = k then
            exists(j in 2..n+1)(path[k, j] = i)
        else
            true
        endif
    )
);

% Each courier's total load must not exceed their capacity
constraint forall(i in 1..m)(
sum(j in 1..n+2 where path[i,j] != n+1)(s[path[i,j]]) <=  l[i]
);

% A package is carried by only one courier only one time and is delivered one time
constraint all_different(i in 1..m, j in 1..n+2 where path[i,j] != n+1)(path[i,j]); %all_different(i in 1..m, j in 2..n+1 where path[i,j] != n+1)(path[i,j]);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  C1 - Couriers Start and End at the Depot
constraint forall(i in 1..m) (
  path[i,1] = n+1 /\ path[i,n+2] = n+1
);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%constraint forall(i in 1..m)(
%        if path[i, n+1] != n+1 then count([path[i, j1] | j1 in 1..n], n+1, 1) 
%        else forall(k in 1..n) (path[i, k] = k) endif
%    );



% Comment if couriers can go back to base before taking other packages
constraint forall(i in 1..m)(
    forall(j in 2..n+1)((c[i,j] == n+1) -> not exists(k in j..n+1)(c[i,k] != n+1))
);
%%%%%%%% Objective function
constraint
    forall(i in 1..m) (
        y[i] = sum([ D[path[i, j1],path[i, j1+1] ] | j1 in 1..n+1])
    );

constraint cost = max(y);

%%%%%%%% SEARCH STRATEGY %%%%%%%%%% (Gecode on inst12)
solve :: int_search(path, first_fail, indomain_min) minimize cost; 
