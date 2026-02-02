import pandapower as pp
import pandapower.topology as top


def debug_grid():
    grid_path = "CSPRL/data/hanoi_citywide"
    print(f"Loading grid from {grid_path}...")
    
    # Load network (using code similar to grid_loader but simplified for debug)
    # Actually simpler to use grid_loader directly if possible, but let's replicate minimal load to be sure
    try:
        from CSPRL.power_grid import GridLoader
        loader = GridLoader(grid_path)
        net = loader.create_network()
    except Exception as e:
        print(f"Failed to load grid: {e}")
        return

    print("\n--- Diagnostic Report ---")
    print(f"Buses: {len(net.bus)}")
    print(f"Lines: {len(net.line)}")
    print(f"Trafos: {len(net.trafo)}")
    print(f"Loads: {len(net.load)}")
    print(f"Ext Grids: {len(net.ext_grid)}")

    # 1. Check Connectivity
    print("\n[1] Connectivity Check:")
    if len(net.ext_grid) == 0:
        print("CRITICAL: No External Grid (Slack Bus) defined!")
    else:
        # Check unsupplied buses
        n_completed = top.unsupplied_buses(net)
        print(f"Unsupplied buses: {len(n_completed)}")
        if len(n_completed) > 0:
            print(f"IDs: {list(n_completed)[:10]}...")

    # 2. Check Load vs Capacity
    print("\n[2] Load vs Capacity Analysis:")
    total_load_mw = net.load.p_mw.sum()
    print(f"Total Load: {total_load_mw:.2f} MW")
    
    # Check max capacity of transformers connected to HV (assuming ext_grid is at HV)
    # Ext grid usually at 500kV or 220kV.
    # Check 500/220kV trafos
    hv_trafos = net.trafo[net.trafo.vn_hv_kv >= 220]
    total_trafo_cap = hv_trafos.sn_mva.sum()
    print(f"Total HV Transformer Capacity: {total_trafo_cap:.2f} MVA")
    
    if total_load_mw > total_trafo_cap:
        print("WARNING: Total load exceeds HV transformer capacity!")

    # 3. Validation
    print("\n[3] Running Diagnostics (pandapower diagnostic function)...")
    try:
        from pandapower.diagnostic import diagnostic
        diagnostic(net)
    except ImportError:
        print("pandapower.diagnostic not available.")

    # 4. Try Power Flow with different options
    print("\n[4] Attempting Power Flow...")
    
    # Try DC first (linear)
    try:
        pp.rundcpp(net)
        print("DC Power Flow: Converged")
    except Exception as e:
        print(f"DC Power Flow: Failed ({e})")

    # Try AC with different algorithms
    algorithms = ['nr', 'iwamoto', 'bfsw', 'gs']
    for alg in algorithms:
        try:
            pp.runpp(net, algorithm=alg, max_iteration=20)
            print(f"AC Power Flow ({alg}): Converged")
            break
        except Exception as e:
            print(f"AC Power Flow ({alg}): Failed")

    # If converged, check results
    if len(net.res_bus) > 0:
        min_vm = net.res_bus.vm_pu.min()
        max_line_loading = net.res_line.loading_percent.max()
        max_trafo_loading = net.res_trafo.loading_percent.max()
        
        print("\n--- Results ---")
        print(f"Min Voltage: {min_vm:.3f} pu")
        print(f"Max Line Loading: {max_line_loading:.2f}%")
        print(f"Max Trafo Loading: {max_trafo_loading:.2f}%")
        
        if min_vm < 0.9:
            print("CRITICAL: Voltage collapse detected!")
        if max_line_loading > 100:
            print("CRITICAL: Line overload detected!")

if __name__ == "__main__":
    debug_grid()
