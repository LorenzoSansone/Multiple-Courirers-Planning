int: m; % number of couriers
int: n; % number of items
array[1..m]: l; % load capacity for each courier
array[1..n]: s; % item size
array[1..n+1, 1..n+1]: D; % distance matrix
set of int: Locations = 1..n; %  Locations of items
set of int: Couriers = 1..m; % Couriers
% DECISION VARIABLES
array[Couriers, Locations] of var 0..1: x; % x[i,j]=1 means courier i delivers to location j
array[Couriers, 1..n+1, 1..n+1] of var 0..1: y;  % y[i,j1,j2] = 1 if courier i travels directly from item j1 to j2
array[Couriers] of var int: tour_distance; %distance traveled by each courier
var int: max_dist; % max between tour distances
var int: origin = n+1; % origin is location n+1 by design