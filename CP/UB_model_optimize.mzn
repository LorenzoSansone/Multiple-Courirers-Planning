% Use this editor as a MiniZinc scratch book
%Define input data

include "bin_packing_capa.mzn";
include "globals.mzn";
include "../instances_dnz/inst21.dzn";


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
var min_dist..max_dist: y;


%path[i,k] = j means courier i goes to point j as the k-th point in the tour
array[1..n] of var 1..n: path;

% load[i] is the total load of courier i
%array[1..m] of var 0..80: load;

%CONSTRAINTS

/*
constraint forall(j in 1..n)(
        exists(i in 1..n)(
            path[i] == j
        )
    );*/
%C3: A package (is carried by only one courier only one time) and is delivered one time
%constraint all_different(i in 1..m, j in 2..n+1 where path[i,j] != n+1)(path[i,j]); 
constraint alldifferent(path);

%C4 - Couriers Start and End at the Deposit
%constraint(path[1] = n+1 /\ path[n+2] = n+1);

    
%C6 - Couriers can't go back to the base to take other packages
%constraint forall(i in 1..m)(
%    forall(j in 2..n+1)((path[i,j] == n+1) -> not exists(k in j..n+1)(path[i,k] != n+1))
%);

%%%%%%%% Objective function

constraint (
        %y = sum([ D[path[j1],path[j1+1] ] | j1 in 1..n+1])
        y =  D[n+1,path[1]] +  sum([ D[path[j1],path[j1+1] ] | j1 in 1..n-1]) + D[path[n],n+1] 

    );

constraint cost = y;

%%%%%%%% SEARCH STRATEGY %%%%%%%%%% 
%indomain_min -> Assign the smallest value in the domain
%first_fail -> Choose the variable with the smallest domain

%solve :: int_search(path, first_fail, indomain_min) minimize cost;
solve :: int_search(path, dom_w_deg, indomain_min)
      :: restart_luby(100)
      :: relax_and_reconstruct(array1d(path), 85) minimize cost; 

/*
output  [show(obj) ++ "\n"] ++
        [show(obj_lowerbound)  ++ "\n"]  ++[ show(obj_upperbound) ++ "\n"];
*/