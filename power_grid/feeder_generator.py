"""
Feeder Generator - Road-Based 22kV Feeders

Tạo 22kV feeders theo road network thực tế từ OSM.
Thay thế self-loop feeders trong citywide_generator.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

try:
    import geopandas as gpd
    from shapely.geometry import Point, LineString
    from shapely.ops import nearest_points
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False


# Feeder parameters - tuned to avoid voltage violations
FEEDER_CONFIG = {
    "max_radius_km": 3.0,       # Max distance từ TBA để tìm roads (reduced from 5)
    "n_feeders_per_tba": 3,     # Số feeders mỗi TBA
    "points_per_feeder": 2,     # Số điểm trên mỗi feeder (reduced from 3)
    "point_spacing_km": 1.0,    # Khoảng cách giữa các điểm (reduced from 1.5)
    "max_i_ka": 0.42,           # Dòng điện max (22kV cable)
    "std_type": "NAYY 4x240 SE",
}


def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance in km between two GPS points."""
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def _sample_points_along_line(line: LineString, n_points: int, start_offset: float = 0.1) -> List[Tuple[float, float]]:
    """Sample n points along a LineString, returning (lat, lon) tuples."""
    points = []
    total_length = line.length
    
    for i in range(n_points):
        # Sample from start_offset to 0.9 of line length
        fraction = start_offset + (i + 1) * (0.9 - start_offset) / (n_points + 1)
        point = line.interpolate(fraction, normalized=True)
        points.append((point.y, point.x))  # lat, lon
    
    return points


def _select_radial_roads(
    roads_gdf: gpd.GeoDataFrame,
    center_lat: float,
    center_lon: float,
    n_roads: int = 3,
) -> List[LineString]:
    """
    Select roads that radiate outward from center in different directions.
    
    Chia thành n_roads sectors và chọn road dài nhất trong mỗi sector.
    """
    if len(roads_gdf) == 0:
        return []
    
    selected = []
    sector_angle = 360.0 / n_roads
    
    for i in range(n_roads):
        angle_start = i * sector_angle
        angle_end = (i + 1) * sector_angle
        
        # Filter roads by direction from center
        sector_roads = []
        for idx, row in roads_gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            
            # Get road endpoint furthest from center
            coords = list(geom.coords)
            if len(coords) < 2:
                continue
            
            # Calculate angle from center to road midpoint
            mid_x = (coords[0][0] + coords[-1][0]) / 2
            mid_y = (coords[0][1] + coords[-1][1]) / 2
            
            dx = mid_x - center_lon
            dy = mid_y - center_lat
            angle = (np.degrees(np.arctan2(dx, dy)) + 360) % 360
            
            if angle_start <= angle < angle_end:
                sector_roads.append((row.geometry, geom.length))
        
        # Select longest road in this sector
        if sector_roads:
            sector_roads.sort(key=lambda x: -x[1])
            selected.append(sector_roads[0][0])
    
    return selected


def generate_feeders_from_roads(
    substations: List[Dict],
    bus_id_map: Dict[str, int],
    road_folder: str,
    start_bus_idx: int,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Generate 22kV feeder buses and lines following road network.
    
    Args:
        substations: List of 110kV substation dicts with name, lat, lon
        bus_id_map: Mapping from bus name to index
        road_folder: Path to folder containing highway.gpkg and road.gpkg
        start_bus_idx: Starting index for new buses
        
    Returns:
        Tuple of (new_buses, new_lines)
    """
    if not GEOPANDAS_AVAILABLE:
        print("[WARN] geopandas not available, using simple radial feeders")
        return _generate_simple_feeders(substations, bus_id_map, start_bus_idx)
    
    # Load road network
    roads_gdf = None
    highway_path = os.path.join(road_folder, "highway.gpkg")
    road_path = os.path.join(road_folder, "road.gpkg")
    
    if os.path.exists(highway_path):
        roads_gdf = gpd.read_file(highway_path)
        print(f"  [OK] Loaded {len(roads_gdf)} highway segments")
    
    if os.path.exists(road_path):
        roads2 = gpd.read_file(road_path)
        print(f"  [OK] Loaded {len(roads2)} road segments")
        if roads_gdf is not None:
            roads_gdf = pd.concat([roads_gdf, roads2], ignore_index=True)
        else:
            roads_gdf = roads2
    
    if roads_gdf is None or len(roads_gdf) == 0:
        print("[WARN] No road data found, using simple radial feeders")
        return _generate_simple_feeders(substations, bus_id_map, start_bus_idx)
    
    # Ensure CRS is WGS84
    if roads_gdf.crs and roads_gdf.crs.to_epsg() != 4326:
        roads_gdf = roads_gdf.to_crs(epsg=4326)
    
    new_buses = []
    new_lines = []
    bus_idx = start_bus_idx
    
    config = FEEDER_CONFIG
    
    for sub in substations:
        sub_name = sub["name"]
        sub_lat = sub["lat"]
        sub_lon = sub["lon"]
        
        # Get 22kV bus for this substation
        bus_22kv_name = f"{sub_name}_22kV"
        if bus_22kv_name not in bus_id_map:
            continue
        bus_22kv_idx = bus_id_map[bus_22kv_name]
        
        # Find roads within radius
        # Buffer in degrees (approx 5km = 0.045 degrees)
        buffer_deg = config["max_radius_km"] / 111.0
        
        center = Point(sub_lon, sub_lat)
        nearby_roads = roads_gdf[roads_gdf.geometry.intersects(center.buffer(buffer_deg))]
        
        if len(nearby_roads) == 0:
            # No roads nearby, create simple radial feeder
            _add_simple_feeder(
                new_buses, new_lines, bus_id_map,
                sub, bus_22kv_idx, bus_idx, config
            )
            bus_idx += config["points_per_feeder"] * config["n_feeders_per_tba"]
            continue
        
        # Select roads radiating in different directions
        selected_roads = _select_radial_roads(
            nearby_roads, sub_lat, sub_lon, 
            n_roads=config["n_feeders_per_tba"]
        )
        
        if len(selected_roads) == 0:
            selected_roads = list(nearby_roads.geometry.head(config["n_feeders_per_tba"]))
        
        # Create feeders along selected roads
        for f_idx, road_geom in enumerate(selected_roads):
            # Sample points along road
            points = _sample_points_along_line(
                road_geom, 
                config["points_per_feeder"]
            )
            
            prev_bus_idx = bus_22kv_idx
            prev_lat, prev_lon = sub_lat, sub_lon
            
            for p_idx, (pt_lat, pt_lon) in enumerate(points):
                # Create feeder bus
                feeder_name = f"{sub_name[:12]}_F{f_idx+1}_{p_idx+1}"
                
                new_buses.append({
                    "name": feeder_name,
                    "vn_kv": 22.0,
                    "type": "n",
                    "x": pt_lon,
                    "y": pt_lat,
                    "voltage_level": "22kV",
                    "district": sub.get("district", "unknown"),
                    "feeder_of": sub_name,
                })
                
                bus_id_map[feeder_name] = bus_idx
                
                # Create line from previous bus
                distance = _haversine_distance(prev_lat, prev_lon, pt_lat, pt_lon)
                
                new_lines.append({
                    "name": f"L22_{sub_name[:8]}_F{f_idx+1}_{p_idx+1}",
                    "from_bus": prev_bus_idx,
                    "to_bus": bus_idx,
                    "length_km": round(max(0.1, distance), 2),
                    "std_type": config["std_type"],
                    "max_i_ka": config["max_i_ka"],
                })
                
                prev_bus_idx = bus_idx
                prev_lat, prev_lon = pt_lat, pt_lon
                bus_idx += 1
    
    print(f"\n[OK] Generated {len(new_buses)} feeder buses and {len(new_lines)} feeder lines")
    
    return new_buses, new_lines


def _generate_simple_feeders(
    substations: List[Dict],
    bus_id_map: Dict[str, int],
    start_bus_idx: int,
) -> Tuple[List[Dict], List[Dict]]:
    """Fallback: Generate simple radial feeders without road data."""
    new_buses = []
    new_lines = []
    bus_idx = start_bus_idx
    config = FEEDER_CONFIG
    
    for sub in substations:
        _add_simple_feeder(
            new_buses, new_lines, bus_id_map,
            sub, bus_id_map.get(f"{sub['name']}_22kV", 0),
            bus_idx, config
        )
        bus_idx += config["points_per_feeder"] * config["n_feeders_per_tba"]
    
    return new_buses, new_lines


def _add_simple_feeder(
    buses: List, lines: List, bus_id_map: Dict,
    sub: Dict, bus_22kv_idx: int, start_idx: int, config: Dict
):
    """Add simple radial feeders for one substation."""
    bus_idx = start_idx
    
    for f in range(config["n_feeders_per_tba"]):
        angle = f * (360 / config["n_feeders_per_tba"])
        prev_bus_idx = bus_22kv_idx
        prev_lat, prev_lon = sub["lat"], sub["lon"]
        
        for p in range(config["points_per_feeder"]):
            # Calculate position
            distance_km = (p + 1) * config["point_spacing_km"]
            offset_deg = distance_km / 111.0
            
            pt_lat = sub["lat"] + offset_deg * np.cos(np.radians(angle))
            pt_lon = sub["lon"] + offset_deg * np.sin(np.radians(angle))
            
            feeder_name = f"{sub['name'][:12]}_F{f+1}_{p+1}"
            
            buses.append({
                "name": feeder_name,
                "vn_kv": 22.0,
                "type": "n",
                "x": pt_lon,
                "y": pt_lat,
                "voltage_level": "22kV",
                "district": sub.get("district", "unknown"),
                "feeder_of": sub["name"],
            })
            
            bus_id_map[feeder_name] = bus_idx
            
            lines.append({
                "name": f"L22_{sub['name'][:8]}_F{f+1}_{p+1}",
                "from_bus": prev_bus_idx,
                "to_bus": bus_idx,
                "length_km": config["point_spacing_km"],
                "std_type": config["std_type"],
                "max_i_ka": config["max_i_ka"],
            })
            
            prev_bus_idx = bus_idx
            bus_idx += 1



