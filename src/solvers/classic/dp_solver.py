# src/solvers/classic/dp_solver.py
import time
import math
from typing import Dict, Any, List
from src.solvers.interface import SolverInterface
from src.utils.generator import load_instance_from_file

class DPSolver2D(SolverInterface):
    """
    A solver for the 0-1 Knapsack Problem using a 2D Dynamic Programming table.
    This corresponds to the original 'knapsack_01_2d' function.
    """
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.name = "2D DP"

    def solve(self, instance_path: str) -> Dict[str, Any]:
        weights, values, capacity = load_instance_from_file(instance_path)
        n = len(weights)
        start_time = time.time()

        dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]

        for i in range(1, n + 1):
            for w in range(1, capacity + 1):
                if weights[i - 1] <= w:
                    dp[i][w] = max(values[i - 1] + dp[i - 1][w - weights[i - 1]], dp[i - 1][w])
                else:
                    dp[i][w] = dp[i - 1][w]
        
        end_time = time.time()
        
        return {
            "value": dp[n][capacity],
            "time": end_time - start_time,
            "solution": [] # Note: Backtracking for the item set is not implemented here.
        }

class DPSolver1D(SolverInterface):
    """
    A solver for the 0-1 Knapsack Problem using a space-optimized 1D DP table.
    This corresponds to the original 'knapsack_01_1d' function.
    """
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.name = "1D DP (Optimized)"

    def solve(self, instance_path: str) -> Dict[str, Any]:
        weights, values, capacity = load_instance_from_file(instance_path)
        n = len(weights)
        start_time = time.time()
        
        dp = [0] * (capacity + 1)

        for i in range(n):
            current_weight = weights[i]
            current_value = values[i]
            for j in range(capacity, current_weight - 1, -1):
                dp[j] = max(dp[j], current_value + dp[j - current_weight])
        
        end_time = time.time()
        
        return {
            "value": dp[capacity],
            "time": end_time - start_time,
            "solution": []
        }

class DPValueSolver(SolverInterface):
    """
    A solver for the 0-1 Knapsack Problem using DP based on value.
    Efficient when total value is smaller than capacity.
    This corresponds to the original 'knapsack_01_1d_value' function.
    """
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.name = "1D DP (on value)"

    def solve(self, instance_path: str) -> Dict[str, Any]:
        weights, values, capacity = load_instance_from_file(instance_path)
        start_time = time.time()
        
        if not weights or capacity < 0:
            return {"value": 0, "time": time.time() - start_time, "solution": []}

        total_value = sum(values)
        dp = [math.inf] * (total_value + 1)
        dp[0] = 0

        for i in range(len(weights)):
            item_weight = weights[i]
            item_value = values[i]
            for v in range(total_value, item_value - 1, -1):
                dp[v] = min(dp[v], dp[v - item_value] + item_weight)

        final_value = 0
        for v in range(total_value, -1, -1):
            if dp[v] <= capacity:
                final_value = v
                break
        
        end_time = time.time()
        
        return {
            "value": final_value,
            "time": end_time - start_time,
            "solution": []
        }