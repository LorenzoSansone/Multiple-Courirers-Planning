from itertools import combinations
from z3 import *

import math
import re
import multiprocessing
import signal
import multiprocessing
import time

def compute_solution(list_shared):
    """
    A function that continuously computes some solution until terminated.
    """
    iteration = 0
    try:
        while True:
            iteration += 1
            print(f"Iteration {iteration}: Computing...")
            time.sleep(0.5)  # Simulate computation time
            s = Solver()
            x = Bool("x")
            s.add(x == True)
            list_shared["it"] = [s.to_smt2(),iteration, [[[[[1]]]]]]
    except KeyboardInterrupt:
        print("Computation interrupted.")
    finally:
        print(f"Computation terminated after {iteration} iterations.")

def clock_function(duration, compute_process,p2_pid):
    print(f"Clock started: Allowing computation for {duration} seconds.")
    start = time.time()
    sat = True
    while sat:
        if time.time()-start > duration:
            sat= False
    #time.sleep(duration)
    #os.kill(p2_pid, signal.SIGTERM)
    compute_process.terminate()  # Terminate the computation process
    print("Clock expired: Terminated computation process.")

if __name__ == "__main__":
    # Duration for which the computation should run
    for i in range(1):
        with multiprocessing.Manager() as manager:#manager = multiprocessing.Manager()
            print(multiprocessing.get_start_method())
            list_shared = manager.dict()
            # Shared dictionary to stor
            list_shared["it"] = "ciao"
            computation_duration = 2  # seconds
        
            # Create a process for the computation
            compute_process = multiprocessing.Process(target=compute_solution, args = (list_shared,))
            compute_process.start()

            time.sleep(computation_duration)
            compute_process.terminate()
            print(list_shared)
            serialized_solver = list_shared["it"][0]
            solver = Solver()
            solver.from_string(serialized_solver)






    #clock = multiprocessing.Process(target=clock_function, args=(5,compute_process,p2_pid))

    # Start the computation process
    #clock.start()



    # Start the clock function in the main proces

    # Wait for the computation process to fully terminate
    #compute_process.join()
    #clock.join()

    print("Main program completed.")














