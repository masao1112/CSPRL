import json
import numpy as np

# Data
CHARGING_POWER = np.array([3, 7, 11, 20, 22, 30, 60, 80, 120, 150, 180, 250])
INSTALL_FEE = np.array([5, 11, 12, 100, 12, 143, 278, 397, 416, 676, 956, 3272])
K = 10


def prepare_config_optimized():
    # best_configs[power] = [cost, configuration_list]
    # We start with 0 power costing 0 VND
    best_configs = {0: [0, [0] * len(CHARGING_POWER)]}

    # We iterate K times (once for each "slot" in the charging station)
    for _ in range(K):
        new_configs = best_configs.copy()

        for current_power, (current_cost, current_counts) in best_configs.items():
            for i in range(len(CHARGING_POWER)):
                new_power = int(current_power + CHARGING_POWER[i])
                new_cost = current_cost + INSTALL_FEE[i]

                # If we found a power level we've never seen,
                # or found a cheaper way to reach an existing power level:
                if new_power not in new_configs or new_cost < new_configs[new_power][0]:
                    new_counts = list(current_counts)
                    new_counts[i] += 1
                    new_configs[new_power] = [new_cost, new_counts]

        best_configs = new_configs

    # --- Monotonicity Logic ---
    # If a higher capacity is cheaper than a lower one, use the higher one's config
    sorted_powers = sorted(best_configs.keys())
    for i in range(len(sorted_powers)):
        power = sorted_powers[i]
        # Look at all options providing EQUAL or MORE power
        # and find the absolute minimum cost
        future_powers = sorted_powers[i:]
        cheapest_power = min(future_powers, key=lambda p: best_configs[p][0])

        # Update the current power level to use that better configuration
        best_configs[power] = best_configs[cheapest_power]

    # Clean up: remove the 0 entry and convert to standard dictionary for JSON
    best_configs.pop(0, None)
    return {k: v[1] for k, v in sorted(best_configs.items())}


# Run and Save
lookup_table = prepare_config_optimized()

save_path = "data/config_lookup.json"
with open(save_path, 'w') as json_file:
    json.dump(lookup_table, json_file, indent=4)

print(f"Success! Saved {len(lookup_table)} optimized configurations to {save_path}.")