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
    
    # Use 1D array instead of 2D: we only need to keep track of the current state
    # This array represents v_k(θ) for the current k we're processing
    dp = [0] * (capacity + 1)
    
    # We still need to track decisions for backtracking
    # Keep this 2D since we need all decisions for the solution reconstruction
    decisions = [[False for _ in range(capacity + 1)] for _ in range(n + 1)]
    
    # Initialize v_1(θ) as shown in slides
    # Process first item separately
    for theta in range(weights[0], capacity + 1):
        dp[theta] = values[0]
        decisions[1][theta] = True
    
    # Main DP loop implementing v_k+1(θ) = max{v_k(θ), v_k(θ - a_k+1) + c_k+1}
    # Process each item after the first one
    for k in range(1, n):
        # Process capacities in reverse to avoid overwriting values we still need
        for theta in range(capacity, weights[k]-1, -1):
            # Check if taking the current item gives better value
            value_with_item = dp[theta - weights[k]] + values[k]
            if value_with_item > dp[theta]:
                dp[theta] = value_with_item
                decisions[k+1][theta] = True
            # If we don't take the item, dp[theta] keeps its current value
    
    # Reconstruct solution using the decisions table
    solution = [0] * n
    current_capacity = capacity
    
    # Backtrack through decisions to build solution
    for k in range(n, 0, -1):
        if decisions[k][current_capacity]:
            solution[k-1] = 1
            current_capacity -= weights[k-1]
    
    return dp[capacity], time.time() - start_time, solution

def solve_knapsack_fptas(n, capacity, values, weights, epsilon):
    start_time = time.time()
    
    # Find c_max = max{c_i}
    c_max = max(values)
    
    # Define K = εc_max/n as per theorem
    K = (epsilon * c_max) / n
    
    # Step 1: Scale values by K: ⌊c_j/K⌋
    scaled_values = [math.floor(v/K) for v in values]
    max_scaled_profit = sum(scaled_values)  # Maximum possible profit after scaling
    
    # Use 1D array for F_j(p) with rolling updates
    # Initialize F_1(p) as per slides
    F = [float('inf')] * (max_scaled_profit + 1)
    F[0] = 0  # Base case
    
    # Initialize for first item
    if scaled_values[0] >= 0:
        F[scaled_values[0]] = weights[0]
    
    # Step 2: Compute F_j+1(p) = min{F_j(p), a_j+1 + F_j(p - c_j+1)}
    for j in range(1, n):
        # Process profits in reverse to avoid overwriting values we need
        for p in range(max_scaled_profit, scaled_values[j]-1, -1):
            # Try to include current item if possible
            prev_profit = p - scaled_values[j]
            if F[prev_profit] != float('inf'):
                F[p] = min(F[p], weights[j] + F[prev_profit])
    
    # Find z* = max{p|F_n(p) ≤ b}
    opt_scaled_value = 0
    for p in range(max_scaled_profit, -1, -1):
        if F[p] <= capacity:
            opt_scaled_value = p
            break
    
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
