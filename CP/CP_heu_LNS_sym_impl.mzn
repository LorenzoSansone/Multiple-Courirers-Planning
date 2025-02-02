include "bin_packing_capa.mzn";
include "globals.mzn";

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

%path[i,k] = j means courier i goes to point j as the k-th point in the tour
array[1..m, 1..n+2] of var 1..n+1: path;

%load[i] 
int: max_load;
max_load = sum([ l[i] | i in 1..m]);
array[1..m] of var 0..max_load: load;


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
constraint alldifferent_except(path, {n+1});

%C4 - Couriers Start and End at the Deposit
constraint forall(i in 1..m) (
  path[i,1] = n+1 /\ path[i,n+2] = n+1
);

%C5 - Checks if all the packages that are assigned to a courier are delivered by him
constraint forall(j in 1..n)(
        exists(i in 2..n+1)(
            path[x[j], i] == j
        )
    );

%C6 - Couriers can't go back to the base to take other packages and starts immediatly
constraint forall(i in 1..m)(
    forall(j in 2..n+1)((path[i,j] == n+1) -> not exists(k in j..n+1)(path[i,k] != n+1))
 );
 
%C7 - Compute the load of each courier
constraint forall(i in 1..m)(
  load[i] = sum([(if x[j] == i then s[j] else 0 endif) | j in 1..n])
);
 
%C8: All the couriers starts
constraint forall(j in 1..n)(exists(i in 1..m)(x[j] == i));
   
%IMPLIED CONSTRAINTS
constraint forall(i in 1..m)(
        path[i,2] != n+1
    );

%SYMMETRY BREAKING CONSTRAINTS  

%S1: If two couriers have the same capacity then they are symmetric,
% to break the symmetry we impose an order (for the package they pick up) betweem them.
constraint symmetry_breaking_constraint(
    forall(courier1 in 1..m, courier2 in 1..m where courier1 < courier2)
     ((l[courier1] == l[courier2]) -> (lex_less(row(path, courier1), row(path, courier2))))
);

%S2: Two couriers path are exchangeable if the maximum weight of the two is less than the minimum loading capacity
% in that case we impose an ordering between them
constraint symmetry_breaking_constraint(
    forall(courier1 in 1..m, courier2 in 1..m where courier1 != courier2)
    (
      (l[courier1] > l[courier2]) ->
      (load[courier1] >= load[courier2])
));


%%%%%%%% Objective function

constraint
    forall(i in 1..m) (
        y[i] = sum([ D[path[i, j1],path[i, j1+1] ] | j1 in 1..n+1])
    );

constraint cost = max(y);
  
%%%%%%%% SEARCH STRATEGY %%%%%%%%%%
solve :: int_search(path, dom_w_deg, indomain_min)  
      :: restart_luby(100)
      :: relax_and_reconstruct(array1d(path), 85)
      minimize cost;
