import gurobipy as gp
from gurobipy import GRB
import time
import math
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
import sys

def read_knapsack_instance(filename):
    with open(filename, 'r') as f:
        n, capacity = map(int, f.readline().split())
        values = list(map(int, f.readline().split()))
        weights = list(map(int, f.readline().split()))
    return n, capacity, values, weights

def solve_knapsack_BLP(n, capacity, values, weights):
    start_time = time.time()
    model = gp.Model("knapsack")
    model.setParam('TimeLimit', 900)
    model.setParam('MIPGap', 0.0)
    model.setParam('OutputFlag', 0)
    
    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    model.setObjective(gp.quicksum(values[i] * x[i] for i in range(n)), GRB.MAXIMIZE)
    model.addConstr(gp.quicksum(weights[i] * x[i] for i in range(n)) <= capacity)
    
    model.optimize()
    
    solution = [int(x[i].x) for i in range(n)]
    return model.objVal, time.time() - start_time, solution

def solve_knapsack_dp(n, capacity, values, weights):
    start_time = time.time()
    
    # Use numpy for memory efficiency
    dp = np.zeros((2, capacity + 1), dtype=np.int64)
    decisions = np.zeros((n, capacity + 1), dtype=np.int8)
    
    for i in range(n):
        curr = i % 2
        prev = 1 - curr
        
        for w in range(capacity + 1):
            if i == 0:
                if weights[i] <= w:
                    dp[curr, w] = values[i]
                    decisions[i, w] = 1
            else:
                dp[curr, w] = dp[prev, w]
                if weights[i] <= w:
                    val_with_item = values[i] + dp[prev, w - weights[i]]
                    if val_with_item > dp[curr, w]:
                        dp[curr, w] = val_with_item
                        decisions[i, w] = 1
    
    # Reconstruct solution
    solution = np.zeros(n, dtype=np.int8)
    remaining_capacity = capacity
    for i in range(n-1, -1, -1):
        if decisions[i, remaining_capacity]:
            solution[i] = 1
            remaining_capacity -= weights[i]
    
    total_value = int(dp[(n-1) % 2, capacity])
    return total_value, time.time() - start_time, solution.tolist()

def solve_knapsack_fptas(n, capacity, values, weights, epsilon):
    start_time = time.time()
    
    # Scale values
    c_max = max(values)
    K = (epsilon * c_max) / n
    scaled_values = [math.floor(v/K) for v in values]
    
    # Instead of arrays, use dictionaries to store only reachable profits
    dp = [{}, {}]  # dp[i][p] = minimum weight to achieve profit p using items 0..i
    dp[0][0] = 0
    
    # Initial item
    if scaled_values[0] >= 0:
        dp[0][scaled_values[0]] = weights[0]
    
    # Fill DP table
    for j in range(1, n):
        curr, prev = j % 2, (j-1) % 2
        dp[curr] = dp[prev].copy()  # Start with previous profits
        
        # Try adding current item to each known profit
        for p, w in list(dp[prev].items()):
            new_p = p + scaled_values[j]
            new_w = w + weights[j]
            
            if new_w <= capacity:  # Only add if within capacity
                if new_p not in dp[curr] or new_w < dp[curr][new_p]:
                    dp[curr][new_p] = new_w
    
    # Find maximum profit achievable within capacity
    max_profit = max(dp[(n-1) % 2].keys())
    actual_value = max_profit * K
    
    return actual_value, time.time() - start_time, None

def evaluate_instance(filename):
    n, capacity, values, weights = read_knapsack_instance(filename)
    results = {}
    
    # Solve BLP
    print("Solving the Binary LP...")
    opt_value, gurobi_time, gurobi_sol = solve_knapsack_BLP(n, capacity, values, weights)
    results['BinaryLP'] = {'value': opt_value, 'time': gurobi_time, 'solution': gurobi_sol}
    
    # Solve DP
    print("Solving with Dynamic Programming...")
    try:
        dp_value, dp_time, dp_sol = solve_knapsack_dp(n, capacity, values, weights)
        results['DP'] = {'value': dp_value, 'time': dp_time, 'solution': dp_sol}
    except (ValueError, MemoryError) as e:
        print(f"Error in DP solution: {e}")
        results['DP'] = {'value': None, 'time': None, 'solution': None}
    
    # Solve FPTAS
    epsilons = [10, 1, 0.1, 0.01]
    results['FPTAS'] = {}
    
    for eps in epsilons:
        print(f"Solving with FPTAS (ε={eps})")
        try:
            fptas_value, fptas_time, _ = solve_knapsack_fptas(n, capacity, values, weights, eps)
            if fptas_value is not None:
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
        filename = f"instances/instance1.txt"
        print(f"\nProcessing {filename}")
        results = evaluate_instance(filename)
        
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