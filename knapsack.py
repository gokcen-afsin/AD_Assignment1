import gurobipy as gp
from gurobipy import GRB
import time
import math
import numpy as np

def read_knapsack_instance(filename):
    # Read instance with n items, capacity b, values c_j, and weights a_j
    with open(filename, 'r') as f:
        n, capacity = map(int, f.readline().split())
        values = list(map(int, f.readline().split()))   
        weights = list(map(int, f.readline().split())) 
    return n, capacity, values, weights

def solve_knapsack_BLP(n, capacity, values, weights):
    # Binary LP formulation
    start_time = time.time()
    model = gp.Model("knapsack")
    model.setParam('TimeLimit', 900)  # time limit
    model.setParam('MIPGap', 0.0)
    model.setParam('OutputFlag', 0)
    
    # Decision variables x_j ∈ {0,1}
    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    
    # Objective: maximize Σ c_j·x_j
    model.setObjective(gp.quicksum(values[i] * x[i] for i in range(n)), GRB.MAXIMIZE)
    
    # capacity constraint: Σ a_j·x_j ≤ b
    model.addConstr(gp.quicksum(weights[i] * x[i] for i in range(n)) <= capacity)
    
    model.optimize()
    solution = [int(x[i].x) for i in range(n)]
    return model.objVal, time.time() - start_time, solution

def solve_knapsack_dp(n, capacity, values, weights):
    # v_k(θ) dynamic programming approach
    start_time = time.time()
    
    # Initialize v_k(θ) table as array [k][θ]
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
    
    # Step 1: Initialize v_1(θ) for first item
    # v_1(θ) = 0 if θ < a_1, c_1 if θ ≥ a_1
    for w in range(capacity + 1):
        if weights[0] <= w:
            dp[1][w] = values[0]
    
    # Step 2: Compute v_(k+1)(θ) using below recurrence
    # v_(k+1)(θ) = max{v_k(θ), v_k(θ - a_(k+1)) + c_(k+1)}
    for k in range(1, n):
        for theta in range(capacity + 1):
            exclude_item = dp[k][theta]  # v_k(θ)
            include_item = 0
            
            if weights[k] <= theta:
                # include the item if its weight ≤ remaining capacity
                include_item = dp[k][theta - weights[k]] + values[k]  # v_k(θ - a_(k+1)) + c_(k+1)
            
            dp[k+1][theta] = max(exclude_item, include_item)
    
    # Step 3: Reconstruct solution with backtracking
    # p_k(θ) decisions - TRUE if item k is chosen
    solution = [0] * n  # p_k(θ) values
    remaining_capacity = capacity
    
    for k in range(n, 0, -1):
        # Check if item k was included in optimal solution
        if dp[k][remaining_capacity] != dp[k-1][remaining_capacity]:
            solution[k-1] = 1  # p_k(θ) = TRUE
            remaining_capacity -= weights[k-1]
    
    return dp[n][capacity], time.time() - start_time, solution

def solve_knapsack_fptas(n, capacity, values, weights, epsilon):
    start_time = time.time()
    
    # Step 1: Find c_max = max{c_i}
    c_max = max(values)
    
    # Define set K = εc_max/n 
    K = (epsilon * c_max) / n
    
    # Step 1: Scale values by ⌊c_j/K⌋
    scaled_values = [math.floor(v/K) for v in values]
    
    # Use dictionary for sparse F_j(p) representation
    F = {0: 0}  # F_1(p) initialization
    
    # Initialize F_1(p) as per slides
    if scaled_values[0] > 0:
        F[scaled_values[0]] = weights[0]
    
    # Step 2: Compute F_j+1(p) = min{F_j(p), a_j+1 + F_j(p - c_j+1)}
    for j in range(1, n):
        new_F = F.copy()
        for p in F:
            new_p = p + scaled_values[j]
            new_w = F[p] + weights[j]
            if new_w <= capacity:
                new_F[new_p] = min(new_F.get(new_p, float('inf')), new_w)
        F = new_F
    
    # Step 3: Find z* = max{p|F_n(p) ≤ b}
    opt_scaled_value = max((p for p in F.keys() if F[p] <= capacity), default=0)
    
    # Convert scaled solution back to original values
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