import json
import itertools
import numpy as np

# Data extracted from your image
# Power in kW (image says kWh, but usually power is kW)
CHARGING_POWER = np.array([7, 11, 20, 22, 30, 60, 80, 120, 150, 180, 250])
# Price in Million VND (Triệu Đồng)
INSTALL_FEE = np.array([11, 12, 100, 12, 143, 278, 397, 416, 676, 956, 3272])

# Constraint: Max number of chargers allowed in a single station
K = 10

def prepare_config():
    """
    Creates a lookup table to find the cheapest charger combinations 
    for any given total power capacity.
    """
    N = len(CHARGING_POWER)
    # The 'urn' creates a pool of possible quantities for each charger type
    urn = list(range(0, K + 1)) * N
    config_list = []
    
    # Generate all possible combinations of charger counts
    for combination in itertools.combinations(urn, N):
        config_list.append(list(combination))

    my_config_dict = {}
    
    for config in config_list:
        config = np.array(config)
        # We only care about configurations that don't exceed our max charger limit K
        if np.sum(config) > K:
            continue
        
        total_capacity = np.sum(CHARGING_POWER * config)
        total_cost = np.sum(INSTALL_FEE * config)
        
        if total_capacity in my_config_dict:
            # Check if this new setup is a better deal (cheaper) for the same power
            current_best_cost = np.sum(INSTALL_FEE * np.array(my_config_dict[total_capacity]))
            if total_cost < current_best_cost:
                my_config_dict[total_capacity] = config.tolist()
        else:
            my_config_dict[total_capacity] = config.tolist()

    # Monotonicity Logic: If a higher capacity is actually cheaper, we use that instead.
    # Why pay more for less?
    key_list = sorted(list(my_config_dict.keys()))
    for index, key in enumerate(key_list):
        # Look at the costs of all configurations that provide AT LEAST this much power
        costs_for_higher_capacities = [
            np.sum(INSTALL_FEE * np.array(my_config_dict[my_key])) 
            for my_key in key_list[index:]
        ]
        
        # Find the absolute cheapest option among equal or higher power levels
        best_cost_index = costs_for_higher_capacities.index(min(costs_for_higher_capacities)) + index
        best_config = my_config_dict[key_list[best_cost_index]]
        my_config_dict[key] = best_config
        
    return my_config_dict

# Run the function
lookup_table = prepare_config()

# Displaying a few examples from the result
print(f"{'Total Capacity (kW)':<20} | {'Best Configuration (Counts)':<30} | {'Total Cost (Million VND)'}")
print("-" * 85)
for cap in sorted(lookup_table.keys())[1:10]: # Showing first few results
    conf = lookup_table[cap]
    cost = np.sum(INSTALL_FEE * np.array(conf))
    print(f"{cap:<20} | {str(conf):<30} | {cost}")
    
save_path = "../data/config_lookup.json"
print(f"Saving to {save_path}...")
with open(save_path, 'w') as json_file:
    json.dump(lookup_table, json_file, indent=4)