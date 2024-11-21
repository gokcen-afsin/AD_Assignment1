import gurobipy as gp
from gurobipy import GRB
import time
import math

def read_knapsack_instance(filename):
    with open(filename, 'r') as f:
        n, capacity = map(int, f.readline().split())
        values = list(map(int, f.readline().split()))
        weights = list(map(int, f.readline().split()))
    return n, capacity, values, weights

def solve_knapsack_BLP(n, capacity, values, weights):
    start_time = time.time()
    
    # Create a new model
    model = gp.Model("knapsack")
    model.setParam('TimeLimit', 900)  # 15 minutes = 900 seconds
    model.setParam('MIPGap', 0.0)     # Set MIP gap to 0
    
    # Create variables
    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    
    # Set objective
    model.setObjective(gp.quicksum(values[i] * x[i] for i in range(n)), GRB.MAXIMIZE)
    
    # Add constraint
    model.addConstr(gp.quicksum(weights[i] * x[i] for i in range(n)) <= capacity)
    
    # Optimize model
    model.optimize()
    
    end_time = time.time()
    
    if model.status == GRB.OPTIMAL:
        solution = [int(x[i].x) for i in range(n)]
        return model.objVal, end_time - start_time, solution
    else:
        return None, end_time - start_time, None

# def solve_knapsack_dp(n, capacity, values, weights):
#     start_time = time.time()
    
#     # Initialize DP table
#     dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
    
#     # Fill DP table
#     for i in range(1, n + 1):
#         for w in range(capacity + 1):
#             if weights[i-1] <= w:
#                 dp[i][w] = max(values[i-1] + dp[i-1][w-weights[i-1]], dp[i-1][w])
#             else:
#                 dp[i][w] = dp[i-1][w]
    
#     # Reconstruct solution
#     solution = [0] * n
#     w = capacity
#     for i in range(n, 0, -1):
#         if dp[i][w] != dp[i-1][w]:
#             solution[i-1] = 1
#             w -= weights[i-1]
    
#     end_time = time.time()
#     return dp[n][capacity], end_time - start_time, solution

# def solve_knapsack_fptas(n, capacity, values, weights, epsilon):
#     start_time = time.time()
    
#     # Find maximum value
#     max_value = max(values)
    
#     # Calculate scaling factor
#     k = epsilon * max_value / n
    
#     # Scale values
#     scaled_values = [int(v / k) for v in values]
    
#     # Maximum scaled value possible
#     max_scaled_value = sum(scaled_values)
    
#     # DP table with scaled values
#     dp = [[float('inf')] * (max_scaled_value + 1) for _ in range(n + 1)]
#     dp[0][0] = 0
    
#     # Fill DP table
#     for i in range(1, n + 1):
#         dp[i][0] = 0
#         for v in range(max_scaled_value + 1):
#             if scaled_values[i-1] <= v:
#                 dp[i][v] = min(dp[i-1][v], 
#                              dp[i-1][v-scaled_values[i-1]] + weights[i-1])
#             else:
#                 dp[i][v] = dp[i-1][v]
    
#     # Find maximum value that fits in capacity
#     opt_scaled_value = 0
#     for v in range(max_scaled_value, -1, -1):
#         if dp[n][v] <= capacity:
#             opt_scaled_value = v
#             break
    
#     # Reconstruct solution
#     solution = [0] * n
#     remaining_value = opt_scaled_value
#     for i in range(n, 0, -1):
#         if remaining_value >= scaled_values[i-1] and \
#            dp[i][remaining_value] == dp[i-1][remaining_value-scaled_values[i-1]] + weights[i-1]:
#             solution[i-1] = 1
#             remaining_value -= scaled_values[i-1]
    
#     # Calculate actual value
#     actual_value = sum(values[i] for i in range(n) if solution[i] == 1)
    
#     end_time = time.time()
#     return actual_value, end_time - start_time, solution

def evaluate_instance(filename):
    # Read instance
    n, capacity, values, weights = read_knapsack_instance(filename)
    
    results = {}
    
    # Solve the Binary LP
    print("Solving the Binary LP...")
    opt_value, gurobi_time, gurobi_sol = solve_knapsack_BLP(n, capacity, values, weights)
    results['BinaryLP'] = {'value': opt_value, 'time': gurobi_time}
    
    # # Solve using Dynamic Programming
    # print("Solving with Dynamic Programming...")
    # dp_value, dp_time, dp_sol = solve_knapsack_dp(n, capacity, values, weights)
    # results['dp'] = {'value': dp_value, 'time': dp_time}
    
    # # Solve using FPTAS with different epsilon values
    # epsilons = [10, 1, 0.1, 0.01]
    # results['fptas'] = {}
    
    # for eps in epsilons:
    #     print(f"Solving with FPTAS (ε={eps})...")
    #     fptas_value, fptas_time, fptas_sol = solve_knapsack_fptas(n, capacity, values, weights, eps/100)  # Convert to decimal
    #     opt_gap = (opt_value - fptas_value) / opt_value * 100 if opt_value else 0
    #     results['fptas'][eps] = {
    #         'value': fptas_value,
    #         'time': fptas_time,
    #         'gap': opt_gap
    #     }
    
    return results

# Example usage
if __name__ == "__main__":
    results = evaluate_instance("instances/instance10.txt")
    
    # Print results
    print("\nResults:")
    print("BLP Solution:")
    print(f"Value: {results['BinaryLP']['value']}")
    print(f"Time: {results['BinaryLP']['time']:.3f} seconds")
    
    # print("\nDynamic Programming Solution:")
    # print(f"Value: {results['dp']['value']}")
    # print(f"Time: {results['dp']['time']:.3f} seconds")
    
    # print("\nFPTAS Solutions:")
    # for eps in [10, 1, 0.1, 0.01]:
    #     print(f"\nε = {eps}:")
    #     print(f"Value: {results['fptas'][eps]['value']}")
    #     print(f"Time: {results['fptas'][eps]['time']:.3f} seconds")
    #     print(f"Gap: {results['fptas'][eps]['gap']:.2f}%")