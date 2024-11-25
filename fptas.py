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

   # Find maximum value
    max_value = max(values)
        
    # Calculate scaling factor
    k = (epsilon * max_value) / n

    # Scale values
    scaled_values = [int(v / k) for v in values]

    # Maximum scaled value possible
    max_scaled_value = n*max(scaled_values)
    # Initialize DP table
    dp = [[0 for _ in range(capacity + 1)] for _ in range(2)]

    # Keep track of decisions for solution reconstruction
    decisions = [[0 for _ in range(capacity + 1)] for _ in range(n)]
    
    # Fill DP table
    for i in range(n):
        
        curr = i % 2
        prev = (i-1) % 2
        
        for p in range(max_scaled_value + 1):
            if i == 0:
                # Handle first item separately
                if scaled_values[i] <= p:
                    dp[curr][p] = weights[i]
                    decisions[i][p] = 1
            else:
               # Don't take item i
                dp[curr][p] = dp[prev][p]

                # Check if we can take item i
                if scaled_values[i] <= p:
                    # Value with current item
                    val_with_item = weights[i] + dp[prev][p - scaled_values[i]]
                    
                    # Take item if it gives better value
                    if val_with_item > dp[curr][p]:
                        dp[curr][p] = val_with_item
                        decisions[i][p] = 1

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
    
    # start_time = time.time()

    
    # # Find maximum value
    # max_value = max(values)
        
    # # Calculate scaling factor
    # k = (epsilon * max_value) / n
     
    # # Add debug prints
    # print(f"\nDebug Info for ε = {epsilon}:")
    # print(f"max_value: {max_value}")
    # print(f"k: {k}")

    # # Scale values
    # scaled_values = [int(v / k) for v in values]
    # # Maximum scaled value possible
    # max_scaled_value = n*max(scaled_values)
    # # Feasibility check
    # weights_per_value = [0 for _ in range(max_scaled_value + 1)]
    # feasibility = [0 for _ in range(max_scaled_value + 1)]
    # solution = [0 for _ in range(n)]
    # print(f"Scaled values range: {min(scaled_values)} to {max(scaled_values)}")
    # print(f"n*C_max: {max_scaled_value}")
        
    # # DP table with scaled values
    # dp = [[float('inf')] for _ in range(max_scaled_value + 1) for _ in range(n + 1)]
    # dp[0][0] = 0  # Base case: empty subset has zero weight

    # # Track decisions for solution reconstruction
    # decisions = [[0 for _ in range(max_scaled_value + 1)] for _ in range(n)]
 
    # # Fill DP table - O(n * max_scaled_value) = O(n³/ε)
    # for i in range(n):
    #     curr = i % 2
    #     prev = (i - 1) % 2
                        
    #     for v in range(max_scaled_value + 1):
    #         # Don't take item i
    #         dp[curr][v] = dp[prev][v]

    #     # Handle first item separately
    #     if i == 0 and weights[i] <= capacity:
    #         dp[curr][scaled_values[i]] = weights[i]
    #         decisions[i][scaled_values[i]] = 1                
            

    #             # Try to take item i if possible
    #     for v in range(max_scaled_value + 1):
    #         if v-scaled_values[i] >= 0:    
    #             additem_value = weights[i] + dp[curr][v-scaled_values[i]]
    #             if additem_value < dp[curr][v]:
    #                 dp[curr][v] = additem_value
    #                 decisions[i][v] = 1
                
    #     for v in range(max_scaled_value + 1):
    #         # weights_per_value[v] = decisions[i][v] * weights[i] (maybe not necassary? included in dp)
    #         if dp[n][v] <= capacity:
    #             feasibility[v] = v
    #     total_value = max(feasibility)
    #     solution[i] = decisions[i][total_value]  

            # THE COMMENTED OUT SECTION IS THE CHAT GPT PART            
                # if scaled_values[i] <= v:
                #     prev_v = v - scaled_values[i]
                #     # Feasibility check
                #     if dp[prev][prev_v] != float('inf'):
                #         new_weight = dp[prev][prev_v] + weights[i]
                #         if new_weight <= capacity and new_weight < dp[curr][v]:
                #             dp[curr][v] = new_weight
                #             decisions[i][v] = 1
        
        # # Find maximum scaled value achievable within capacity
        # final_row = (n - 1) % 2
        # feasible_values = [v for v in range(max_scaled_value + 1) if dp[final_row][v] != float('inf')]
        # print(f"Number of feasible scaled values: {len(feasible_values)}")
        # if feasible_values:
        #     print(f"Range of feasible values: {min(feasible_values)} to {max(feasible_values)}")
        
        # if not feasible_values:
        #     raise ValueError('No feasible solution found.')

        # # Find maximum scaled value achievable within capacity
        # opt_scaled_value = max_scaled_value
               
        # # Reconstruct solution
        # solution = [0] * n
        # remaining_value = opt_scaled_value
        
        # for i in range(n-1, -1, -1):
        #     if decisions[i][remaining_value]:
        #         solution[i] = 1
        #         remaining_value -= scaled_values[i]
        
        # # Calculate actual (unscaled) value and verify solution
        # total_value = sum(values[i] for i in range(n) if solution[i])
        # total_weight = sum(weights[i] for i in range(n) if solution[i])
        
        
        # # Verify capacity constraint
        # if total_weight > capacity:
        #     raise ValueError("Solution exceeds capacity")
        
    return total_value, end_time - start_time, solution        
    
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