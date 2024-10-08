% Use this editor as a MiniZinc scratch book
%Define input data

include "bin_packing_capa.mzn";
include "globals.mzn";
%include "../instances_dzn/inst11.dzn";


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

%I1 Implied constraint
constraint forall(i in 1..m)(
        if not exists(j in 1..n)(x[j] == i) then not exists(k in 2..n+1)(path[i,k] != n+1)
        else path[i,2] != n+1
        endif
    );

%SYMMETRY

%S1 - Couriers can't go back to the base to take other packages and starts immediatly
constraint symmetry_breaking_constraint (forall(i in 1..m)(
    forall(j in 2..n+1)((path[i,j] == n+1) -> not exists(k in j..n+1)(path[i,k] != n+1)))
);


%S2
constraint symmetry_breaking_constraint(
    forall(c1 in 1..m, c2 in 1..m where c1 < c2)
    (
    (max(sum([s[path[c1, j1]] | j1 in 2..n+1 where path[c1, j1] != n+1]), sum([s[path[c2, j1]] | j1 in 2..n+1 where path[c2, j1] != n+1])) <=   min(l[c1], l[c2])) -> (lex_less(row(path, c1), row(path,c2))))
);


%%%%%%%% Objective function
constraint
    forall(i in 1..m) (
        y[i] = sum([ D[path[i, j1],path[i, j1+1] ] | j1 in 1..n+1])
    );

constraint cost = max(y);
  
%%%%%%%% SEARCH STRATEGY %%%%%%%%%% (Gecode on inst12)
solve :: int_search(path, dom_w_deg, indomain_min)  
      :: restart_luby(100)
      :: relax_and_reconstruct(array1d(path), 85)
      minimize cost;
