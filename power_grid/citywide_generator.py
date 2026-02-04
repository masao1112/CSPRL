"""
Hanoi City-wide Grid Generator

Tạo dữ liệu lưới điện cho toàn bộ Hà Nội sử dụng dữ liệu TBA thực tế.
"""

import pandas as pd
import numpy as np
import os
from .hanoi_substations import SUBSTATIONS_500KV, SUBSTATIONS_220KV, SUBSTATIONS_110KV


def generate_hanoi_citywide_grid(
    output_folder: str = None,
    seed: int = 42,
) -> str:
    """
    Tao du lieu luoi dien cho toan bo Ha Noi.
    
    Power grid nay chi tao co so ha tang dien, KHONG bao gom tram sac EV.
    Tram sac EV se duoc RL agent dat o bat ky dau tren map dua tren reward function.
    
    Su dung du lieu TBA thuc te tu hanoi_substations.py de tao:
    - Buses: Tat ca TBA 500kV, 220kV, 110kV, 22kV
    - Lines: Ket noi giua cac TBA theo cap dien ap
    - Trafos: MBA tai moi TBA
    - Loads: Phu tai phan bo theo khu vuc (input cho reward function)
    
    Args:
        output_folder: Thu muc luu ket qua
        seed: Random seed
        
    Returns:
        Duong dan thu muc da tao
    """
    np.random.seed(seed)
    
    if output_folder is None:
        # Default to CSPRL/data/hanoi_citywide
        output_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "hanoi_citywide")
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # ========== 1. TAO BUSES ==========
    buses = []
    bus_idx = 0
    bus_id_map = {}  # Map substation name -> bus index
    
    # 500kV buses
    for sub in SUBSTATIONS_500KV:
        buses.append({
            "name": sub["name"],
            "vn_kv": 500.0,
            "type": "b",  # busbar
            "x": sub["lon"],
            "y": sub["lat"],
            "voltage_level": "500kV",
            "capacity_mva": sub.get("capacity_mva", 900),
        })
        bus_id_map[sub["name"]] = bus_idx
        bus_idx += 1
    
    # 220kV buses
    for sub in SUBSTATIONS_220KV:
        buses.append({
            "name": sub["name"],
            "vn_kv": 220.0,
            "type": "b",
            "x": sub["lon"],
            "y": sub["lat"],
            "voltage_level": "220kV",
            "capacity_mva": sub.get("capacity_mva", 500),
        })
        bus_id_map[sub["name"]] = bus_idx
        bus_idx += 1
    
    # 110kV buses
    for sub in SUBSTATIONS_110KV:
        buses.append({
            "name": sub["name"],
            "vn_kv": 110.0,
            "type": "n",  # node
            "x": sub["lon"],
            "y": sub["lat"],
            "voltage_level": "110kV",
            "district": sub.get("district", "unknown"),
            "capacity_mva": sub.get("capacity_mva", 63),
        })
        bus_id_map[sub["name"]] = bus_idx
        bus_idx += 1
    
    # 22kV buses (secondary side of 110kV transformers)
    n_110kv = len(SUBSTATIONS_110KV)
    for i, sub in enumerate(SUBSTATIONS_110KV):
        buses.append({
            "name": f"{sub['name']}_22kV",
            "vn_kv": 22.0,
            "type": "n",
            "x": sub["lon"] + 0.001,  # Slight offset
            "y": sub["lat"] + 0.001,
            "voltage_level": "22kV",
            "district": sub.get("district", "unknown"),
        })
        bus_id_map[f"{sub['name']}_22kV"] = bus_idx
        bus_idx += 1
    
    buses_df = pd.DataFrame(buses)
    buses_df.index.name = "index"
    
    # ========== 2. TAO EXTERNAL GRID ==========
    # Connect to 500kV buses
    ext_grids = []
    for i, sub in enumerate(SUBSTATIONS_500KV):
        ext_grids.append({
            "name": f"EVN_GRID_{i+1}",
            "bus": bus_id_map[sub["name"]],
            "vm_pu": 1.0,
            "va_degree": 0.0,
        })
    
    ext_grid_df = pd.DataFrame(ext_grids)
    ext_grid_df.index.name = "index"
    
    # ========== 3. TAO TRANSFORMERS ==========
    trafos = []
    trafo_idx = 0
    
    # 500/220kV transformers
    # Strategy: Connect every 500kV to its 2 nearest 220kV substations to feed into the 220kV grid
    for sub_500 in SUBSTATIONS_500KV:
        # Find 2 nearest 220kV substations
        distances = []
        for sub_220 in SUBSTATIONS_220KV:
            dist = np.sqrt((sub_500["lat"] - sub_220["lat"])**2 + 
                          (sub_500["lon"] - sub_220["lon"])**2)
            distances.append((sub_220["name"], dist))
        distances.sort(key=lambda x: x[1])
        
        # Connect to 2 nearest 220kV (Redundant feed)
        for nearest_name, _ in distances[:2]:
            trafos.append({
                "name": f"T_{sub_500['name'][:10]}_{nearest_name[:10]}",
                "hv_bus": bus_id_map[sub_500["name"]],
                "lv_bus": bus_id_map[nearest_name],
                "sn_mva": 450.0,
                "vn_hv_kv": 500.0,
                "vn_lv_kv": 220.0,
                "std_type": "450 MVA 500/220 kV",
                "pfe_kw": 150.0,
                "i0_percent": 0.04,
            })
            trafo_idx += 1
    
    # 220/110kV transformers
    # Strategy: GUARANTEE every 110kV substation is connected to its nearest 220kV source
    for sub_110 in SUBSTATIONS_110KV:
        # Find nearest 220kV substation
        min_dist = float('inf')
        nearest_220_name = None
        
        for sub_220 in SUBSTATIONS_220KV:
            dist = np.sqrt((sub_110["lat"] - sub_220["lat"])**2 + 
                          (sub_110["lon"] - sub_220["lon"])**2)
            if dist < min_dist:
                min_dist = dist
                nearest_220_name = sub_220["name"]
        
        # Connect to nearest 220kV (No distance limit, must connect)
        if nearest_220_name:
            trafos.append({
                "name": f"T_{nearest_220_name[:8]}_{sub_110['name'][:8]}",
                "hv_bus": bus_id_map[nearest_220_name],
                "lv_bus": bus_id_map[sub_110["name"]],
                "sn_mva": 125.0,
                "vn_hv_kv": 220.0,
                "vn_lv_kv": 115.0,  # Slightly higher tap for voltage support
                "std_type": "125 MVA 220/110 kV",
                "pfe_kw": 80.0,
                "i0_percent": 0.05,
            })
            trafo_idx += 1
    
    # 110/22kV transformers at each 110kV substation
    n_500_220 = len(SUBSTATIONS_500KV) + len(SUBSTATIONS_220KV)
    for i, sub in enumerate(SUBSTATIONS_110KV):
        hv_bus = bus_id_map[sub["name"]]
        lv_bus = bus_id_map[f"{sub['name']}_22kV"]
        
        trafos.append({
            "name": f"T_{sub['name'][:15]}_22kV",
            "hv_bus": hv_bus,
            "lv_bus": lv_bus,
            "sn_mva": sub.get("capacity_mva", 63),
            "vn_hv_kv": 110.0,
            "vn_lv_kv": 23.0,
            "std_type": f"{sub.get('capacity_mva', 63)} MVA 110/22 kV",
            "pfe_kw": 35.0,
            "i0_percent": 0.04,
        })
        trafo_idx += 1
    
    trafos_df = pd.DataFrame(trafos)
    trafos_df.index.name = "index"
    
    # ========== 4. TAO LINES ==========
    lines = []
    line_idx = 0
    
    # 220kV lines between 220kV substations
    # 220kV lines between 220kV substations (Mesh)
    for i, sub1 in enumerate(SUBSTATIONS_220KV):
        # Connect to at least 2 nearest neighbors to form a mesh
        distances = []
        for j, sub2 in enumerate(SUBSTATIONS_220KV):
            if i != j:
                dist = np.sqrt((sub1["lat"] - sub2["lat"])**2 + 
                              (sub1["lon"] - sub2["lon"])**2) * 111
                distances.append((sub2, dist))
        distances.sort(key=lambda x: x[1])
        
        # Connect to 2 nearest neighbors
        for sub2, dist in distances[:2]:
            # Simple check: only add if name1 < name2 to avoid double lines
            if sub1["name"] < sub2["name"]:
                lines.append({
                    "name": f"L220_{sub1['name'][:6]}_{sub2['name'][:6]}",
                    "from_bus": bus_id_map[sub1["name"]],
                    "to_bus": bus_id_map[sub2["name"]],
                    "length_km": round(dist, 2),
                    "std_type": "220kV OHL",
                    "max_i_ka": 1.5,
                })
                line_idx += 1
    
    # 110kV lines between 110kV substations in same district
    districts = {}
    for sub in SUBSTATIONS_110KV:
        dist = sub.get("district", "unknown")
        if dist not in districts:
            districts[dist] = []
        districts[dist].append(sub)
    
    for district, subs in districts.items():
        for i, sub1 in enumerate(subs):
            for j, sub2 in enumerate(subs):
                if i < j:
                    dist = np.sqrt((sub1["lat"] - sub2["lat"])**2 + 
                                  (sub1["lon"] - sub2["lon"])**2) * 111
                    if dist < 15:
                        lines.append({
                            "name": f"L110_{sub1['name'][:6]}_{sub2['name'][:6]}",
                            "from_bus": bus_id_map[sub1["name"]],
                            "to_bus": bus_id_map[sub2["name"]],
                            "length_km": round(dist, 2),
                            "std_type": "110kV OHL",
                            "max_i_ka": 0.8,
                        })
                        line_idx += 1
    
    # 22kV feeders following road network
    road_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "QGIS-Related", "POIs")
    
    try:
        from .feeder_generator import generate_feeders_from_roads
        
        print("\n[INFO] Generating 22kV feeders from road network...")
        feeder_buses, feeder_lines = generate_feeders_from_roads(
            substations=SUBSTATIONS_110KV,
            bus_id_map=bus_id_map,
            road_folder=road_folder,
            start_bus_idx=bus_idx,
        )
        
        # Add feeder buses to main list
        for fb in feeder_buses:
            buses.append(fb)
            bus_idx += 1
        
        # Add feeder lines
        lines.extend(feeder_lines)
        line_idx += len(feeder_lines)
        
        # Update buses_df with new feeder buses
        buses_df = pd.DataFrame(buses)
        buses_df.index.name = "index"
        
    except Exception as e:
        print(f"[WARN] Road-based feeder generation failed: {e}")
        print("[INFO] Creating simple radial feeders as fallback...")
        
        # Fallback: simple radial feeders (still avoid self-loops)
        for sub in SUBSTATIONS_110KV:
            bus_22kv = bus_id_map[f"{sub['name']}_22kV"]
            for f in range(3):
                angle = f * 120
                offset = 0.02  # ~2km
                
                feeder_lat = sub["lat"] + offset * np.cos(np.radians(angle))
                feeder_lon = sub["lon"] + offset * np.sin(np.radians(angle))
                feeder_name = f"{sub['name'][:12]}_F{f+1}"
                
                buses.append({
                    "name": feeder_name,
                    "vn_kv": 22.0,
                    "type": "n",
                    "x": feeder_lon,
                    "y": feeder_lat,
                    "voltage_level": "22kV",
                })
                bus_id_map[feeder_name] = bus_idx
                
                lines.append({
                    "name": f"L22_{sub['name'][:8]}_F{f+1}",
                    "from_bus": bus_22kv,
                    "to_bus": bus_idx,
                    "length_km": 2.0,
                    "std_type": "NAYY 4x240 SE",
                    "max_i_ka": 0.42,
                })
                bus_idx += 1
                line_idx += 1
        
        buses_df = pd.DataFrame(buses)
        buses_df.index.name = "index"
    
    lines_df = pd.DataFrame(lines)
    lines_df.index.name = "index"
    
    # ========== 5. TAO LOADS (POI-BASED) ==========
    # Try to use POI-based load generation, fallback to district profiles
    poi_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "QGIS-Related", "POIs")
    use_poi_loads = os.path.exists(poi_folder)
    
    if use_poi_loads:
        try:
            from .poi_load_generator import POILoadGenerator
            
            print("\n[INFO] Using POI-based load generation...")
            poi_gen = POILoadGenerator(poi_folder)
            
            # Prepare bus coordinates for aggregation
            bus_coords = []
            for idx, row in buses_df.iterrows():
                bus_coords.append({
                    "idx": idx,
                    "lat": row.get("y", 21.0),
                    "lon": row.get("x", 105.8),
                    "name": row.get("name", f"Bus_{idx}"),
                    "vn_kv": row.get("vn_kv", 22.0),
                })
            
            loads_df = poi_gen.aggregate_to_buses(bus_coords, voltage_kv=22.0)
            
            # Ensure required columns exist
            if "poi_count" in loads_df.columns:
                loads_df = loads_df.drop(columns=["poi_count"])
            
        except Exception as e:
            print(f"[WARN] POI load generation failed: {e}")
            print("[INFO] Falling back to district-based loads...")
            use_poi_loads = False
    
    if not use_poi_loads:
        # Fallback: District load profiles (MW per 110kV substation area)
        loads = []
        district_loads = {
            "cau_giay": {"base_mw": 35, "type": "commercial"},
            "dong_da": {"base_mw": 30, "type": "mixed"},
            "hoan_kiem": {"base_mw": 25, "type": "commercial"},
            "thanh_xuan": {"base_mw": 28, "type": "mixed"},
            "ha_dong": {"base_mw": 32, "type": "industrial"},
            "long_bien": {"base_mw": 25, "type": "industrial"},
            "tay_ho": {"base_mw": 20, "type": "residential"},
        }
        
        for sub in SUBSTATIONS_110KV:
            dist = sub.get("district", "unknown")
            profile = district_loads.get(dist, {"base_mw": 20, "type": "mixed"})
            
            # Base load at 22kV bus
            p_mw = profile["base_mw"] * np.random.uniform(0.8, 1.2)
            pf = 0.9 if profile["type"] == "industrial" else 0.92
            q_mvar = p_mw * np.tan(np.arccos(pf))
            
            bus_22kv = bus_id_map[f"{sub['name']}_22kV"]
            
            loads.append({
                "name": f"LOAD_{sub['name'][:12]}",
                "bus": bus_22kv,
                "p_mw": round(p_mw, 2),
                "q_mvar": round(q_mvar, 2),
                "scaling": 1.0,
                "type": profile["type"],
            })
        
        loads_df = pd.DataFrame(loads)
        loads_df.index.name = "index"
    
    # ========== 6.5 INJECT INDUSTRIAL/SHADOW LOADS ==========
    print("\n[INFO] Injecting industrial base loads...")
    industrial_keywords = ["KCN", "CNC", "Thăng Long", "Phú Nghĩa", "Quang Minh", "Nội Bài", "Vân Trì", "Công viên Thống Nhất"]
    industrial_base_load = 3.0 # MW - Reduced from 20.0 to prevent voltage collapse
    
    extra_loads = []
    
    for idx, row in buses_df.iterrows():
        bus_name = str(row["name"])
        vn_kv = row["vn_kv"]
        
        # Target 22kV buses
        if vn_kv == 22.0:
            is_industrial = any(k in bus_name for k in industrial_keywords)
            
            if is_industrial:
                 extra_loads.append({
                    "name": f"LOAD_IND_{bus_name[:15]}",
                    "bus": idx,
                    "p_mw": industrial_base_load,
                    "q_mvar": industrial_base_load * 0.3,
                    "scaling": 1.0,
                    "type": "industrial",
                })
    
    if extra_loads:
        extra_df = pd.DataFrame(extra_loads)
        loads_df = pd.concat([loads_df, extra_df], ignore_index=True)
        print(f"[OK] Injected {len(extra_loads)} industrial loads (Total: {sum(l['p_mw'] for l in extra_loads):.2f} MW)")

    # ========== 7. SAVE TO CSV ==========
    buses_df.to_csv(os.path.join(output_folder, "bus.csv"))
    lines_df.to_csv(os.path.join(output_folder, "line.csv"))
    trafos_df.to_csv(os.path.join(output_folder, "trafo.csv"))
    loads_df.to_csv(os.path.join(output_folder, "load.csv"))
    ext_grid_df.to_csv(os.path.join(output_folder, "ext_grid.csv"))
    
    # Metadata
    total_load = loads_df["p_mw"].sum()
    
    metadata = {
        "name": "Hanoi Citywide Power Grid",
        "n_buses": len(buses_df),
        "n_lines": len(lines_df),
        "n_trafos": len(trafos_df),
        "n_loads": len(loads_df),
        "total_load_mw": round(total_load, 2),
        "n_500kv": len(SUBSTATIONS_500KV),
        "n_220kv": len(SUBSTATIONS_220KV),
        "n_110kv": len(SUBSTATIONS_110KV),
    }
    
    pd.DataFrame([metadata]).to_csv(
        os.path.join(output_folder, "metadata.csv"), index=False
    )
    
    print(f"\n[OK] Generated Hanoi citywide grid at: {output_folder}")
    print(f"   Buses: {metadata['n_buses']} ({metadata['n_500kv']}x500kV, {metadata['n_220kv']}x220kV, {metadata['n_110kv']}x110kV)")
    print(f"   Lines: {metadata['n_lines']}")
    print(f"   Transformers: {metadata['n_trafos']}")
    print(f"   Loads: {metadata['n_loads']} (background loads only, no EV)")
    print(f"   Total Load: {metadata['total_load_mw']:.2f} MW")
    print(f"   Note: EV stations will be placed by RL agent based on reward function.")
    
    return output_folder


if __name__ == "__main__":
    import argparse
    
    default_output = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "hanoi_citywide")
    
    parser = argparse.ArgumentParser(description="Generate Hanoi citywide grid data (no EV stations)")
    parser.add_argument("--output", "-o", default=default_output)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    generate_hanoi_citywide_grid(
        output_folder=args.output,
        seed=args.seed,
    )
