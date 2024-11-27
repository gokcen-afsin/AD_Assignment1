import gurobipy as gp
from gurobipy import GRB
import time
import math
import numpy as np

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
    
    # Use numpy array for faster operations
    dp = np.zeros(capacity + 1, dtype=np.int64)  
    
    # Initialize for first item using slice operation instead of loop
    dp[weights[0]:] = values[0]
    
    # Main DP loop - vectorized operations where possible
    for k in range(1, n):
        # Process only weights that can fit
        valid_weights = range(capacity, weights[k]-1, -1)
        dp_view = dp[valid_weights]
        dp_prev_view = dp[np.array(valid_weights) - weights[k]]
        updates = dp_prev_view + values[k]
        mask = updates > dp_view
        dp[valid_weights] = np.maximum(dp_view, updates)
    
    # Build solution - this part is already efficient
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
    
    # Scale values by K: ⌊c_j/K⌋
    scaled_values = [math.floor(v/K) for v in values]
    
    # Initialize F_j(p) using sparse representation
    F = {0: 0}  # Base case: F_1(0) = 0
    if scaled_values[0] > 0:
        F[scaled_values[0]] = weights[0]  # F_1(c_1) = a_1
    
    # Compute F_j+1(p) = min{F_j(p), a_j+1 + F_j(p - c_j+1)}
    for j in range(1, n):
        new_F = F.copy()
        for p, w in F.items():
            new_p = p + scaled_values[j]
            new_w = w + weights[j]
            if new_w <= capacity:
                new_F[new_p] = min(new_F.get(new_p, float('inf')), new_w)
        F = new_F
    
    # Find z* = max{p|F_n(p) ≤ b}
    opt_scaled_value = max(F.keys())
    
    # Convert back to original scale
    actual_value = opt_scaled_value * K
    
    return actual_value, time.time() - start_time, None

def evaluate_instance(filename):
    n, capacity, values, weights = read_knapsack_instance(filename)
    results = {}
    
    print("Solving Binary LP...")
    opt_value, gurobi_time, gurobi_sol = solve_knapsack_BLP(n, capacity, values, weights)
    results['BinaryLP'] = {'value': opt_value, 'time': gurobi_time, 'solution': gurobi_sol}
    
    print("Solving with Dynamic Programming...")
    try:
        dp_value, dp_time, dp_sol = solve_knapsack_dp(n, capacity, values, weights)
        results['DP'] = {'value': dp_value, 'time': dp_time, 'solution': dp_sol}
    except Exception as e:
        print(f"DP failed: {e}")
        results['DP'] = {'value': None, 'time': None, 'solution': None}
    
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
    for i in range(1, 11):
        filename = f"instances/instance{i}.txt"
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