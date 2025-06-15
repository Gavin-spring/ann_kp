def knapsack_01_2d(*, weights: list, values: list, capacity: int) -> int:
    """
    Solves the 0/1 knapsack problem using a 2D DP array.

    Args:
        weights (list): A list of weights for each item.
        values (list): A list of values for each item.
        capacity (int): The maximum capacity of the knapsack.

    Returns:
        int: The maximum total value that can be obtained.
    """
    n = len(weights)
    # dp[i][j] stores the maximum value using the first 'i' items
    # with a knapsack capacity of 'j'.
    # Initialize with zeros.
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    # Iterate through each item
    for i in range(1, n + 1):
        # Current item's weight and value (adjusting for 0-indexed lists)
        current_weight = weights[i - 1]
        current_value = values[i - 1]

        # Iterate through each possible capacity
        for j in range(capacity + 1):
            # Case 1: Don't include the current item
            dp[i][j] = dp[i - 1][j]

            # Case 2: Include the current item (if capacity allows)
            if j >= current_weight:
                dp[i][j] = max(dp[i][j], current_value + dp[i - 1][j - current_weight])

    # The result is in the bottom-right corner of the DP table
    return dp[n][capacity]


def knapsack_01_1d(*, weights: list, values: list, capacity: int) -> int:
    """
    Solves the 0/1 knapsack problem using a 1D space-optimized DP array.

    Args:
        weights (list): A list of weights for each item.
        values (list): A list of values for each item.
        capacity (int): The maximum capacity of the knapsack.

    Returns:
        int: The maximum total value that can be obtained.
    """
    n = len(weights)
    # dp[j] stores the maximum value for a knapsack with capacity 'j'.
    # Initialize with zeros.
    dp = [0] * (capacity + 1)

    # Iterate through each item
    for i in range(n):
        current_weight = weights[i]
        current_value = values[i]

        # Iterate through capacities in reverse order
        # This ensures that when calculating dp[j], dp[j - current_weight]
        # refers to the value from the previous item's consideration,
        # not the current item's.
        for j in range(capacity, current_weight - 1, -1):
            # Either don't include the current item (dp[j] already holds this from previous iteration)
            # Or include the current item (value + dp[j - current_weight])
            dp[j] = max(dp[j], current_value + dp[j - current_weight])

    # The result is the maximum value for the full capacity
    return dp[capacity]


def knapsack_01_2d_with_trace(*, weights: list, values: list, capacity: int) -> int:
    """
    Solves the 0/1 knapsack problem using a 2D DP array,
    and prints the DP table's filling process step by step.

    Args:
        weights (list): A list of weights for each item.
        values (list): A list of values for each item.
        capacity (int): The maximum capacity of the knapsack.

    Returns:
        int: The maximum total value that can be obtained.
    """
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    print("----- Starting 0/1 Knapsack DP Table Filling Process -----")
    print(f"Items: {list(zip(weights, values))} (weight, value)")
    print(f"Knapsack Capacity: {capacity}")
    print("-" * 60)

    # Print initial DP table
    print("\nInitial DP Table (before filling):")
    # Header for capacities
    header = "      " + " ".join([f"{j:4d}" for j in range(capacity + 1)])
    print(header)
    print("      " + "----" * (capacity + 1))
    for r_idx, row in enumerate(dp):
        print(f"Item {r_idx:<2d}| {' '.join([f'{val:4d}' for val in row])}")
    print("-" * 60)


    # Iterate through each item
    for i in range(1, n + 1):
        current_weight = weights[i - 1]
        current_value = values[i - 1]
        print(f"\n--- Considering Item {i} (Weight: {current_weight}, Value: {current_value}) ---")

        # Iterate through each possible capacity
        for j in range(capacity + 1):
            # Case 1: Don't include the current item
            # The value from the previous row for the same capacity
            value_if_not_included = dp[i - 1][j]
            dp[i][j] = value_if_not_included

            # Case 2: Include the current item (if capacity allows)
            value_if_included = -1 # Sentinel value, indicates not possible
            if j >= current_weight:
                value_if_included = current_value + dp[i - 1][j - current_weight]
                dp[i][j] = max(dp[i][j], value_if_included)

            print(f"  Capacity j={j}:")
            print(f"    - Value if NOT included (dp[{i-1}][{j}]): {value_if_not_included}")
            if j >= current_weight:
                print(f"    - Value if INCLUDED ({current_value} + dp[{i-1}][{j - current_weight}]): {value_if_included}")
            else:
                print(f"    - Cannot include (capacity {j} < weight {current_weight})")
            print(f"    -> dp[{i}][{j}] set to: {dp[i][j]}")
            
            # Print current state of DP table after updating dp[i][j]
            print("\n  Current DP Table State:")
            # Header for capacities
            header = "      " + " ".join([f"{k:4d}" for k in range(capacity + 1)])
            print(header)
            print("      " + "----" * (capacity + 1))
            for r_idx, row in enumerate(dp):
                print(f"Item {r_idx:<2d}| {' '.join([f'{val:4d}' for val in row])}")
            print("-" * (60 + (capacity * 5))) # Adjust line length dynamically

    print("\n----- Final DP Table -----")
    # Header for capacities
    header = "      " + " ".join([f"{j:4d}" for j in range(capacity + 1)])
    print(header)
    print("      " + "----" * (capacity + 1))
    for r_idx, row in enumerate(dp):
        print(f"Item {r_idx:<2d}| {' '.join([f'{val:4d}' for val in row])}")
    print("-" * 60)

    final_max_value = dp[n][capacity]
    print(f"\nMax Value (dp[{n}][{capacity}]): {final_max_value}")
    print("----- End of Knapsack DP Process -----")

    return final_max_value



