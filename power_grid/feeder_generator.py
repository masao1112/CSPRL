"""
Feeder Generator - Graph-Based & Population-Driven
==================================================

Generates 22kV feeders that:
1. Start from 110kV substations.
2. Follow actual road network topology (using NetworkX).
3. Target high-population-density areas (using Population Density Data).

Replaces the old "longest radial road" logic with meaningful, physical path finding.
"""

import os
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

try:
    import geopandas as gpd
    import networkx as nx
    from shapely.geometry import Point, LineString
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False


# Feeder parameters
FEEDER_CONFIG = {
    "max_radius_km": 2.5,
    "n_feeders_per_tba": 3,
    "points_per_feeder": 4,
    "node_spacing_km": 0.6,
    "max_i_ka": 0.20,
}

POPULATION_FILE_REL_PATH = "QGIS-Related/population/hanoi_pop_density_100m.gpkg"


def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance in km between two GPS points."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def _build_graph_from_gdf(roads_gdf: gpd.GeoDataFrame) -> nx.MultiDiGraph:
    """Convert a GeoDataFrame of LineStrings into a NetworkX Graph."""
    G = nx.MultiGraph()
    
    for idx, row in roads_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
            
        if geom.geom_type == 'LineString':
            geoms = [geom]
        elif geom.geom_type == 'MultiLineString':
            geoms = list(geom.geoms)
        else:
            continue
            
        for line in geoms:
            coords = list(line.coords)
            if len(coords) < 2:
                continue
                
            u = coords[0] # (lon, lat)
            v = coords[-1]
            
            length_val = _haversine_distance(u[1], u[0], v[1], v[0])
            
            G.add_edge(u, v, weight=length_val, geometry=line)
            
    return G


def _get_nearest_node(G: nx.Graph, point: Tuple[float, float]):
    """Find nearest node in Graph G to a point (lon, lat)."""
    nodes = list(G.nodes)
    if not nodes:
        return None
        
    best_node = None
    min_dist = float('inf')
    pt_lon, pt_lat = point
    
    # Naive search check
    for node in nodes:
        dist = (node[0] - pt_lon)**2 + (node[1] - pt_lat)**2
        if dist < min_dist:
            min_dist = dist
            best_node = node
            
    return best_node


def _load_population_data(data_folder: str) -> Optional[gpd.GeoDataFrame]:
    """Load population density polygons."""
    potential_paths = [
        os.path.join(POPULATION_FILE_REL_PATH),
        os.path.join("..", "data", POPULATION_FILE_REL_PATH),
        os.path.join("data", POPULATION_FILE_REL_PATH),
        os.path.abspath(os.path.join("CSPRL/data", POPULATION_FILE_REL_PATH))
    ]
    
    for p in potential_paths:
        if os.path.exists(p):
            try:
                gdf = gpd.read_file(p)
                if gdf.crs and gdf.crs.to_epsg() != 4326:
                    gdf = gdf.to_crs(epsg=4326)
                print(f"  [OK] Loaded population data from {p}")
                return gdf
            except Exception as e:
                print(f"  [WARN] Failed to load population data: {e}")
                
    print("  [WARN] Population data not found.")
    return None


def _get_angle(lat1, lon1, lat2, lon2):
    """Get angle from point 1 to point 2 (0-360 degrees)."""
    dy = lat2 - lat1
    dx = lon2 - lon1
    rads = math.atan2(dy, dx)
    deg = math.degrees(rads)
    return (deg + 360) % 360


def _get_population_targets(
    sub_lat: float, 
    sub_lon: float, 
    pop_gdf: gpd.GeoDataFrame,
    road_graph: nx.Graph,
    n_targets: int, 
    radius_km: float
) -> List[Tuple[float, float]]:
    """
    Select targets distributed across sectors (angular directions).
    Prioritize high population in each sector.
    """
    targets = []
    
    # 1. Define Sectors
    sector_size = 360.0 / n_targets
    
    # 2. Prepare Population Candidates
    candidates = []
    has_pop = pop_gdf is not None and len(pop_gdf) > 0
    
    if has_pop:
        center = Point(sub_lon, sub_lat)
        radius_deg = radius_km / 111.0
        nearby = pop_gdf[pop_gdf.intersects(center.buffer(radius_deg))].copy()
        
        # Identify value column
        val_col = None
        for col in ['VALUE', 'population', 'pop', 'density']:
            if col in nearby.columns:
                val_col = col
                break
        
        if val_col and len(nearby) > 0:
            for idx, row in nearby.iterrows():
                cent = row.geometry.centroid
                candidates.append({
                    'lat': cent.y, 'lon': cent.x, 
                    'val': row[val_col], 'type': 'pop'
                })
    
    # 3. Prepare Graph Node Candidates (Fallback)
    # Pick nodes at the edge of the radius
    graph_candidates = []
    nodes = list(road_graph.nodes)
    # Sampling to avoid iterating all
    sample_nodes = nodes if len(nodes) < 2000 else nodes[::5] 
    
    for n in sample_nodes:
        # n is (lon, lat)
        dist = _haversine_distance(sub_lat, sub_lon, n[1], n[0])
        if 0.5 < dist <= radius_km:
            graph_candidates.append({
                'lat': n[1], 'lon': n[0], 
                'val': 0, 'type': 'road' # low priority
            })
            
    all_candidates = candidates + graph_candidates
    
    # 4. Select Best per Sector
    for i in range(n_targets):
        angle_start = i * sector_size
        angle_end = (i + 1) * sector_size
        
        sector_cands = []
        for cand in all_candidates:
            angle = _get_angle(sub_lat, sub_lon, cand['lat'], cand['lon'])
            # Handle wrap around 360 if needed, but simplified here
            # Normalize to match sectors roughly
            if angle_start <= angle < angle_end:
                 sector_cands.append(cand)
                 
        if sector_cands:
            # Sort by priority: Pop value high -> Road val 0
            # If candidates have mixed types, pop wins due to value
            sector_cands.sort(key=lambda x: x['val'], reverse=True)
            best = sector_cands[0]
            targets.append((best['lat'], best['lon']))
            
    return targets


def generate_feeders_from_roads(
    substations: List[Dict],
    bus_id_map: Dict[str, int],
    road_folder: str,
    start_bus_idx: int,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Generate feeders ensuring density and spread.
    """
    if not GEOPANDAS_AVAILABLE:
        print("[WARN] geopandas not available.")
        return [], []
    
    # Load Roads
    highway_path = os.path.join(road_folder, "highway.gpkg")
    road_path = os.path.join(road_folder, "road.gpkg")
    loaded_parts = []
    
    for p in [highway_path, road_path]:
        if os.path.exists(p):
            try:
                part = gpd.read_file(p)
                if part.crs and part.crs.to_epsg() != 4326:
                    part = part.to_crs(epsg=4326)
                loaded_parts.append(part)
            except: pass
            
    if not loaded_parts:
        return [], []
        
    roads_gdf = pd.concat(loaded_parts, ignore_index=True)
    print(f"  [OK] Road segments: {len(roads_gdf)}")

    G = _build_graph_from_gdf(roads_gdf)
    pop_gdf = _load_population_data(road_folder)
    
    new_buses = []
    new_lines = []
    bus_idx = start_bus_idx
    config = FEEDER_CONFIG
    
    for sub in substations:
        sub_name = sub["name"]
        sub_lat = sub["lat"]
        sub_lon = sub["lon"]
        
        bus_22kv_name = f"{sub_name}_22kV"
        if bus_22kv_name not in bus_id_map:
            continue
        bus_22kv_idx = bus_id_map[bus_22kv_name]
        
        start_node = _get_nearest_node(G, (sub_lon, sub_lat))
        if not start_node:
            continue
            
        targets = _get_population_targets(
            sub_lat, sub_lon, pop_gdf, G,
            config["n_feeders_per_tba"], 
            config["max_radius_km"]
        )
        
        for f_idx, target in enumerate(targets):
            target_node = _get_nearest_node(G, (target[1], target[0]))
            if not target_node or start_node == target_node:
                continue
                
            try:
                path_nodes = nx.shortest_path(G, source=start_node, target=target_node, weight='weight')
                
                # Placement Logic: Accumulate Distance
                nodes_to_place = []
                acc_dist = 0
                last_coord = path_nodes[0]
                
                for i in range(1, len(path_nodes)):
                    curr_coord = path_nodes[i]
                    seg_dist = _haversine_distance(last_coord[1], last_coord[0], 
                                                 curr_coord[1], curr_coord[0])
                    acc_dist += seg_dist
                    
                    if acc_dist >= config["node_spacing_km"]:
                        nodes_to_place.append(curr_coord)
                        acc_dist = 0 # reset or keep remainder? 
                        # Resetting creates cleaner discrete spacing
                        
                    last_coord = curr_coord
                    
                    if len(nodes_to_place) >= config["points_per_feeder"]:
                        break
                
                # If path is short but we have enough room for at least 1 node
                if not nodes_to_place and len(path_nodes) > 1:
                    nodes_to_place.append(path_nodes[-1])
                
                prev_bus_idx = bus_22kv_idx
                prev_pos = (sub_lon, sub_lat)
                
                for p_idx, curr_node in enumerate(nodes_to_place):
                    feeder_name = f"{sub_name}_22kV_F{f_idx+1}_{p_idx+1}"
                    # Tiny jitter
                    jitter = np.random.uniform(-0.00001, 0.00001, 2)
                    
                    new_buses.append({
                        "name": feeder_name,
                        "vn_kv": 22.0,
                        "type": "n",
                        "x": curr_node[0] + jitter[0],
                        "y": curr_node[1] + jitter[1],
                        "voltage_level": "22kV",
                        "district": sub.get("district", "unknown"),
                        "feeder_of": sub_name,
                    })
                    bus_id_map[feeder_name] = bus_idx
                    
                    dist = _haversine_distance(prev_pos[1], prev_pos[0], curr_node[1], curr_node[0])
                    new_lines.append({
                        "name": f"L22_{sub_name[:8]}_F{f_idx+1}_{p_idx+1}",
                        "from_bus": prev_bus_idx,
                        "to_bus": bus_idx,
                        "length_km": max(0.05, round(dist, 3)),
                        "std_type": config["std_type"],
                        "max_i_ka": config["max_i_ka"],
                    })
                    
                    prev_bus_idx = bus_idx
                    prev_pos = curr_node
                    bus_idx += 1
                    
            except nx.NetworkXNoPath:
                continue

    print(f"\n[OK] Generated {len(new_buses)} buses, {len(new_lines)} lines (Graph w/ Sectors)")
    return new_buses, new_lines
