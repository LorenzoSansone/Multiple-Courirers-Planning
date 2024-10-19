% Use this editor as a MiniZinc scratch book
%Define input data

include "globals.mzn";
%include "../instances_dnz/inst00.dzn";

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
% y is the total distance travelled by courier 
var min_dist..max_dist: y;

%path[i,k] = j means courier i goes to point j as the k-th point in the tour
array[1..n+2] of var 1..n+1: path;

%CONSTRAINTS

%C1
constraint forall(j in 1..n)(
        exists(i in 2..n+1)(
            path[i] == j
        )
    );
%C2: A package (is carried by only one courier only one time) and is delivered one time
constraint alldifferent_except(path, {n+1});

%C3 - Couriers Start and End at the Deposit
constraint(path[1] = n+1 /\ path[n+2] = n+1); 

%%%%%%%% Objective function

constraint (
        y = sum([ D[path[j1],path[j1+1] ] | j1 in 1..n+1])

    );

constraint cost = y;

%%%%%%%% SEARCH STRATEGY %%%%%%%%%% 
%indomain_min -> Assign the smallest value in the domain
%first_fail -> Choose the variable with the smallest domain

%solve :: int_search(path, first_fail, indomain_min) minimize cost;
solve :: int_search(path, dom_w_deg, indomain_min)
      :: restart_luby(100)
      :: relax_and_reconstruct(array1d(path), 85) minimize cost; 