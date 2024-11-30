from dataclasses import dataclass
from enum import Enum

def seconds_to_milliseconds(seconds: int):
    return seconds * 1000

def minutes_to_milliseconds(minutes: int):
    return minutes * 60 * 1000
def milliseconds_to_seconds(milliseconds: int):
    return int(milliseconds / 1000)

class Status(Enum):
    OPTIMAL_SOLUTION = "OPTIMAL_SOLUTION"
    FEASIBLE_SOLUTION = "FEASIBLE_SOLUTION"
    INFEASIBLE = "INFEASIBLE"

@dataclass
class Solution:
    x: list
    y: list

@dataclass
class Result:
    solution: Solution = None
    status: Status = None
    objective: int = None
    statistics: dict = None
