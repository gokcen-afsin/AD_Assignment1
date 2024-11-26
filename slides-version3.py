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
    
    # Initialize 1D DP table
    dp = [0] * (capacity + 1)
    
    # Handle first item separately
    for w in range(weights[0], capacity + 1):
        dp[w] = values[0]
    
    # Main DP loop
    for i in range(1, n):
        # Forward iteration
        for w in range(weights[i], capacity + 1):  # Changed to forward iteration
            new_val = dp[w - weights[i]] + values[i]
            if new_val > dp[w]:
                dp[w] = new_val
    
    # Build solution
    solution = [0] * n
    w = capacity
    for i in range(n-1, -1, -1):
        if w >= weights[i] and dp[w] == dp[w-weights[i]] + values[i]:
            solution[i] = 1
            w -= weights[i]
    
    return dp[capacity], time.time() - start_time, solution

def solve_knapsack_fptas(n, capacity, values, weights, epsilon):
    start_time = time.time()
    
    # Find c_max = max{c_i}
    c_max = max(values)
    
    # Define K = εc_max/n as per theorem
    K = (epsilon * c_max) / n
    
    # Step 1: Scale values by K: ⌊c_j/K⌋
    scaled_values = [math.floor(v/K) for v in values]
    
    # Initialize F_1(p) as per slides using dictionary for sparse representation
    F = {0: 0}  # Only store reachable profits to save memory
    
    # Initialize for first item
    if scaled_values[0] > 0:
        F[scaled_values[0]] = weights[0]
    
    # Step 2: Compute F_j+1(p) = min{F_j(p), a_j+1 + F_j(p - c_j+1)}
    for j in range(1, n):
        # Create new dictionary for current iteration
        new_F = F.copy()
        
        # Process existing profits
        for p in F:
            new_p = p + scaled_values[j]
            new_w = F[p] + weights[j]
            
            if new_w <= capacity:
                new_F[new_p] = min(new_F.get(new_p, float('inf')), new_w)
        
        F = new_F
    
    # Find z* = max{p|F_n(p) ≤ b}
    opt_scaled_value = max((p for p in F.keys() if F[p] <= capacity), default=0)
    
    # Convert back to original scale
    actual_value = opt_scaled_value * K
    
    return actual_value, time.time() - start_time, None

def evaluate_instance(filename):
    n, capacity, values, weights = read_knapsack_instance(filename)
    results = {}
    
    # Solve using BLP
    print("Solving Binary LP...")
    opt_value, gurobi_time, gurobi_sol = solve_knapsack_BLP(n, capacity, values, weights)
    results['BinaryLP'] = {'value': opt_value, 'time': gurobi_time, 'solution': gurobi_sol}
    
    # Solve using DP
    print("Solving with Dynamic Programming...")
    try:
        dp_value, dp_time, dp_sol = solve_knapsack_dp(n, capacity, values, weights)
        results['DP'] = {'value': dp_value, 'time': dp_time, 'solution': dp_sol}
    except Exception as e:
        print(f"DP failed: {e}")
        results['DP'] = {'value': None, 'time': None, 'solution': None}
    
    # Solve using FPTAS for different ε values
    epsilons = [10, 1, 0.1, 0.01]
    results['FPTAS'] = {}
    
    for eps in epsilons:
        print(f"Solving with FPTAS (ε={eps})")
        try:
            fptas_value, fptas_time, _ = solve_knapsack_fptas(n, capacity, values, weights, eps/100)
            opt_gap = (opt_value - fptas_value) / opt_value * 100 if opt_value else 0
            results['FPTAS'][eps] = {
                'value': fptas_value,
                'time': fptas_time,
                'gap': opt_gap
            }
        except Exception as e:
            print(f"FPTAS failed for ε={eps}: {e}")
            results['FPTAS'][eps] = {'value': None, 'time': None, 'gap': None}
    
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
