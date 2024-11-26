import gurobipy as gp
from gurobipy import GRB
import time
import math
import numpy as np

def read_knapsack_instance(filename):
    #Read the instance from file.
    with open(filename, 'r') as f:
        # number of items and knapsack capacity
        n, capacity = map(int, f.readline().split())
        # item values
        values = list(map(int, f.readline().split()))
        # item weights
        weights = list(map(int, f.readline().split()))
    return n, capacity, values, weights

def solve_knapsack_BLP(n, capacity, values, weights):
    start_time = time.time()
    
    # Initialize Gurobi model with parameters for exact solution
    model = gp.Model("knapsack")
    model.setParam('TimeLimit', 900)  # 15 minutes timeout
    model.setParam('MIPGap', 0.0)     # Require optimal solution
    model.setParam('OutputFlag', 0) 
    
    # Create binary variables: x[i] = 1 if item i is selected, 0 otherwise
    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    
    # Objective: maximize total value
    model.setObjective(gp.quicksum(values[i] * x[i] for i in range(n)), GRB.MAXIMIZE)
    
    # Constraint: knapsack weight capacity
    model.addConstr(gp.quicksum(weights[i] * x[i] for i in range(n)) <= capacity)
    
    # Solve the model
    model.optimize()
    
    # return objective value
    solution = [int(x[i].x) for i in range(n)]
    return model.objVal, time.time() - start_time, solution

def solve_knapsack_dp(n, capacity, values, weights):
    start_time = time.time()
    
    # dp[i][w] stores maximum value achievable with weight w using items 0..i
    # Only reachable weights are stored
    dp = [{}, {}]
    
    # Track decisions for solution reconstruction
    # decisions[(i,w)] = 1 means item i was used to achieve weight w
    decisions = {}
    
    # Base case: zero weight has zero value
    dp[0][0] = 0
    
    # Handle first item separately
    if weights[0] <= capacity:
        dp[0][weights[0]] = values[0]
        decisions[(0, weights[0])] = 1
    
    # Main DP loop: process each item
    for i in range(1, n):
        curr, prev = i % 2, (i-1) % 2  # Use two arrays to save memory
        dp[curr] = dp[prev].copy()      # Start with previous state
        
        # For each achievable weight, try adding current item
        for w, v in list(dp[prev].items()):
            new_w = w + weights[i]
            if new_w <= capacity:  # Check if new weight is feasible
                new_v = v + values[i]
                # Update if new value is better than existing
                if new_w not in dp[curr] or new_v > dp[curr][new_w]:
                    dp[curr][new_w] = new_v
                    decisions[(i, new_w)] = 1
                    
    # Find optimal value and corresponding weight
    final_dict = dp[(n-1) % 2]
    max_value = max(final_dict.values()) if final_dict else 0
    max_weight = max((w for w, v in final_dict.items() if v == max_value), default=0)
    
    # Reconstruct solution by backtracking through decisions
    solution = [0] * n
    curr_weight = max_weight
    for i in range(n-1, -1, -1):
        if (i, curr_weight) in decisions:
            solution[i] = 1
            curr_weight -= weights[i]
    
    return max_value, time.time() - start_time, solution

def solve_knapsack_fptas(n, capacity, values, weights, epsilon):
    start_time = time.time()
    
    # Calculate scaling factor K = (ε * max_value) / n
    # This ensures approximation ratio of (1-ε) while keeping scaled values small
    c_max = max(values)
    K = (epsilon * c_max) / n
    
    # Scale down values and round down to ensure feasibility
    # Smaller ε means less scaling (better precision) but larger state space
    scaled_values = [math.floor(v/K) for v in values]
    
    # Use dictionaries to store only reachable profits
    # dp[i][p] = minimum weight needed to achieve profit p using items 0..i
    dp = [{}, {}]
    dp[0][0] = 0  # Base case: zero profit needs zero weight
    
    # Initialize with first item
    if scaled_values[0] >= 0:
        dp[0][scaled_values[0]] = weights[0]
    
    # Main FPTAS loop: process each item
    for j in range(1, n):
        curr, prev = j % 2, (j-1) % 2
        dp[curr] = dp[prev].copy()  # Start with previous profits
        
        # For each achievable profit, try adding current item
        for p, w in list(dp[prev].items()):
            new_p = p + scaled_values[j]
            new_w = w + weights[j]
            
            if new_w <= capacity:  # Check if weight is feasible
                # Update if new weight is better for this profit
                if new_p not in dp[curr] or new_w < dp[curr][new_p]:
                    dp[curr][new_p] = new_w
    
    # Find maximum profit that respects capacity constraint
    max_profit = max(dp[(n-1) % 2].keys())
    
    # Scale back to original values
    actual_value = max_profit * K
    
    return actual_value, time.time() - start_time, None

def evaluate_instance(filename):
    # Read and parse instance
    n, capacity, values, weights = read_knapsack_instance(filename)
    results = {}
    
    # Solve using Gurobi (exact solution)
    print("Solving the Binary LP...")
    opt_value, gurobi_time, gurobi_sol = solve_knapsack_BLP(n, capacity, values, weights)
    results['BinaryLP'] = {'value': opt_value, 'time': gurobi_time, 'solution': gurobi_sol}
    
    # Solve using Dynamic Programming (exact solution)
    print("Solving with Dynamic Programming...")
    try:
        dp_value, dp_time, dp_sol = solve_knapsack_dp(n, capacity, values, weights)
        results['DP'] = {'value': dp_value, 'time': dp_time, 'solution': dp_sol}
    except (ValueError, MemoryError) as e:
        print(f"Error in DP solution: {e}")
        results['DP'] = {'value': None, 'time': None, 'solution': None}
    
    # Solve using FPTAS (approximate solution)
    epsilons = [10, 1, 0.1, 0.01]  # Different approximation ratios
    results['FPTAS'] = {}
    
    for eps in epsilons:
        print(f"Solving with FPTAS (ε={eps})")
        try:
            fptas_value, fptas_time, _ = solve_knapsack_fptas(n, capacity, values, weights, eps)
            if fptas_value is not None:
                # Calculate gap from optimal solution
                opt_gap = (opt_value - fptas_value) / opt_value * 100 if opt_value else 0
                results['FPTAS'][eps] = {
                    'value': fptas_value,
                    'time': fptas_time,
                    'gap': opt_gap
                }
            else:
                print("FPTAS skipped due to problem size")
                results['FPTAS'][eps] = {
                    'value': None,
                    'time': fptas_time,
                    'gap': None
                }
        except Exception as e:
            print(f"Error in FPTAS: {e}")
            results['FPTAS'][eps] = {
                'value': None,
                'time': None,
                'gap': None
            }
    
    return results

if __name__ == "__main__":
    # Process all test instances
    for i in range(1, 11):
        filename = f"instances/instance{i}.txt"
        print(f"\nProcessing {filename}")
        results = evaluate_instance(filename)
        
        # Print results for all methods
        print("\nResults:")
        print("BLP Solution:")
        print(f"Value: {results['BinaryLP']['value']}")
        print(f"Time: {results['BinaryLP']['time']:.3f} seconds")
        
        print("\nDynamic Programming Solution:")
        if results['DP']['value'] is not None:
            print(f"Value: {results['DP']['value']}")
            print(f"Time: {results['DP']['time']:.3f} seconds")
        else:
            print("DP solution failed.")
        
        print("\nFPTAS Solutions:")
        for eps in [10, 1, 0.1, 0.01]:
            print(f"\nε = {eps}:")
            if results['FPTAS'][eps]['value'] is not None:
                print(f"Value: {results['FPTAS'][eps]['value']}")
                print(f"Time: {results['FPTAS'][eps]['time']:.3f} seconds")
                print(f"Gap: {results['FPTAS'][eps]['gap']:.2f}%")
            else:
                print("FPTAS solution failed.")