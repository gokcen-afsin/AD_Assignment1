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
    model.setParam('OutputFlag', 0)   # Less verbose
    model.setParam('Threads', 4)      # Use 4 threads

    # Create variables
    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    
    # Set objective
    model.setObjective(gp.quicksum(values[i] * x[i] for i in range(n)), GRB.MAXIMIZE)
    
    # Add constraint
    model.addConstr(gp.quicksum(weights[i] * x[i] for i in range(n)) <= capacity)
    
    # Optimize model
    model.optimize()
    
    end_time = time.time()
    
    # if model.status == GRB.OPTIMAL:
    solution = [int(x[i].x) for i in range(n)]
    return model.objVal, end_time - start_time, solution
    # else:
    #     return None, end_time - start_time, None

def solve_knapsack_dp(n, capacity, values, weights):
    start_time = time.time()
    
    # Initialize DP table
    dp = [[0 for _ in range(capacity + 1)] for _ in range(2)]

    # Keep track of decisions for solution reconstruction
    decisions = [[0 for _ in range(capacity + 1)] for _ in range(n)]
    
    # Fill DP table
    for i in range(n):
        
        curr = i % 2
        prev = (i-1) % 2
        
        for w in range(capacity + 1):
            if i == 0:
                # Handle first item separately
                if weights[i] <= w:
                    dp[curr][w] = values[i]
                    decisions[i][w] = 1
            else:
               # Don't take item i
                dp[curr][w] = dp[prev][w]

                # Check if we can take item i
                if weights[i] <= w:
                    # Value with current item
                    val_with_item = values[i] + dp[prev][w - weights[i]]
                    
                    # Take item if it gives better value
                    if val_with_item > dp[curr][w]:
                        dp[curr][w] = val_with_item
                        decisions[i][w] = 1

    # Reconstruct solution
    solution = [0] * n
    remaining_capacity = capacity

    # Start from last item
    for i in range(n-1, -1, -1):
        if decisions[i][remaining_capacity]:
            solution[i] = 1
            remaining_capacity -= weights[i]

    # Verify solution feasibility
    total_weight = sum(weights[i] for i in range(n) if solution[i])
    if total_weight > capacity:
        raise ValueError("Solution exceeds capacity")
    
    total_value = sum(values[i] for i in range(n) if solution[i])
    
    end_time = time.time()
    return total_value, end_time - start_time, solution

def solve_knapsack_fptas(n, capacity, values, weights, epsilon):
    
    start_time = time.time()

    try:
        # Find maximum value
        max_value = max(values)
        
        # Calculate scaling factor
        k = (epsilon * max_value) / n
    
        # Scale values
        scaled_values = [int(v / k) for v in values]
        # Maximum scaled value possible
        max_scaled_value = n*max(scaled_values)
        
        # Step 1: Creating F(p)
        f = [[float('inf')] *(max_scaled_value + 1) for _ in range(n)]
        decisions = [[0] *(max_scaled_value + 1) for _ in range(n)]

        for p in range(max_scaled_value + 1):
            if p == 0:
                f[0][p] = 0
            if p == scaled_values[0]:
                f[0][p] = weights[0]
                decisions[0][p] = 1  

        # Step 2: 
        for j in range(1,n):
            for p in range(max_scaled_value + 1):
                if p-scaled_values[j] >= 0:
                    added_value = weights[j] + f[j][p-scaled_values[j]]
                    if added_value < f[j-1][p]:
                        f[j][p] = added_value
                        decisions[j][p] = 1
        
        feasible_values = [0 for _ in range(max_scaled_value + 1)]
        solution = [0 for _ in range(n)]
        # Step 3
        for j in range(n):
            for p in range(max_scaled_value + 1):
                if f[j][p] <= capacity:
                    feasible_values[p] = p
        total_value = max(values)
        for j in range(n):
            solution[j] = decisions[total_value][j]
        end_time = time.time()
        return total_value, end_time - start_time, solution        
    
    except Exception as e:
        print(f"Error in FPTAS solution: {str(e)}")
        return None, time.time() - start_time, None

def evaluate_instance(filename):
    # Read instance
    n, capacity, values, weights = read_knapsack_instance(filename)
    
    results = {}
    
    # # Solve the Binary LP
    # print("Solving the Binary LP...")
    # opt_value, gurobi_time, gurobi_sol = solve_knapsack_BLP(n, capacity, values, weights)
    # results['BinaryLP'] = {'value': opt_value, 'time': gurobi_time, 'solution': gurobi_sol}
    
    # # Solve using Dynamic Programming
    # print("Solving with Dynamic Programming...")
    # try: 
    #     dp_value, dp_time, dp_sol = solve_knapsack_dp(n, capacity, values, weights)
    #     results['DP'] = {'value': dp_value, 'time': dp_time, 'solution': dp_sol}
    
    # except ValueError as e:
    #     print(f"Error in DP solution: {e}")
    #     results['DP'] = {'value': None, 'time': None, 'solution': None}

    # except MemoryError:
    #     print("DP solution exceeded available memory")
    #     results['DP'] = {'value': None, 'time': None, 'solution': None}

    # Solve using FPTAS with different epsilon values
    epsilons = [10, 1, 0.1, 0.01]
    results['FPTAS'] = {}
    
    for eps in epsilons:
        print(f"Solving with FPTAS (ε={eps})...")
        fptas_value, fptas_time, fptas_sol = solve_knapsack_fptas(n, capacity, values, weights, eps)  # Convert to decimal

        if fptas_value is not None:
            # Calculate gap from optimal (BLP) solution
            opt_gap = (results['BinaryLP']['value'] - fptas_value) / results['BinaryLP']['value'] * 100
            results['FPTAS'][eps] = {'value': fptas_value, 'time': fptas_time, 'solution': fptas_sol, 'gap': opt_gap}
        else:
            results['FPTAS'][eps] = {'value': None, 'time': fptas_time, 'solution': None, 'gap': None}
    
    return results

# Example usage
if __name__ == "__main__":
    results = evaluate_instance("instances/instance1.txt")
    
    # # Print results
    # print("\nResults:")
    # print("BLP Solution:")
    # print(f"Value: {results['BinaryLP']['value']}")
    # print(f"Time: {results['BinaryLP']['time']:.3f} seconds")
    
    # print("\nDynamic Programming Solution:")
    # if results['DP']['value'] is not None:
    #     print(f"Value: {results['DP']['value']}")
    #     print(f"Time: {results['DP']['time']:.3f} seconds")
    # else:
    #     print("DP solution failed.")
    
    print("\nFPTAS Solutions:")
    for eps in [10, 1, 0.1, 0.01]:
        print(f"\nε = {eps}:")
        if results['FPTAS'][eps]['time'] == None and results['FPTAS'][eps]['gap'] == None:
            print(f"Value: {results['FPTAS'][eps]['value']}")
        else:
            print(f"Value: {results['FPTAS'][eps]['value']}")
            print(f"Time: {results['FPTAS'][eps]['time']} seconds")
            print(f"Gap: {results['FPTAS'][eps]['gap']}%")