"""
POI-Based Load Generator

Tạo phụ tải điện dựa trên dữ liệu POI từ GeoPackage files.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any

try:
    import geopandas as gpd
    from shapely.geometry import Point
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False


# Load parameters per POI type (MW)
LOAD_PARAMS = {
    "apartment": {
        "mw_per_floor": 0.015,  # ~15kW per floor (residential) - UPGRADED
        "flats_per_floor": 8,   # Default if building:flats not available
        "mw_per_flat": 0.002,   # ~2kW per flat average - UPGRADED
        "min_load": 0.05,       # Minimum load even for small buildings
    },
    "mall": {
        "base_mw": 0.5,         # Reduced from 5.0
        "mw_per_floor": 0.2,    # Reduced from 1.5
        "min_load": 0.5,        # Minimum for any mall
    },
    "entertainment": {
        "cinema": 0.8,
        "restaurant": 0.1,
        "bar": 0.05,
        "nightclub": 0.3,
        "theatre": 0.5,
        "default": 0.15,
    },
    "highway": {
        "mw_per_km": 0.02,      # Street lighting
    },
    "road": {
        "mw_per_km": 0.005,     # Minor road lighting
    },
}


class POILoadGenerator:
    """
    Generate electrical loads from POI GeoPackage data.
    
    Sử dụng:
        gen = POILoadGenerator("power_grid/data/QGIS-Related/POIs")
        loads_df = gen.aggregate_to_buses(buses_df, bus_coords)
    """
    
    def __init__(self, poi_folder: str):
        """
        Khởi tạo generator.
        
        Args:
            poi_folder: Thư mục chứa các file .gpkg
        """
        if not GEOPANDAS_AVAILABLE:
            raise ImportError("geopandas chưa được cài đặt. Chạy: pip install geopandas")
        
        self.poi_folder = poi_folder
        self.pois: Dict[str, gpd.GeoDataFrame] = {}
        self.poi_loads: List[Dict] = []  # List of {lat, lon, load_mw, poi_type, name}
        
        self._load_pois()
        self._calculate_all_loads()
    
    def _load_pois(self) -> None:
        """Load all POI GeoPackages."""
        poi_files = {
            "apartment": "apartment.gpkg",
            "mall": "mall.gpkg",
            "entertainment": "entertainment.gpkg",
            "highway": "highway.gpkg",
            "road": "road.gpkg",
        }
        
        for poi_type, filename in poi_files.items():
            filepath = os.path.join(self.poi_folder, filename)
            if os.path.exists(filepath):
                gdf = gpd.read_file(filepath)
                # Ensure CRS is WGS84 (lat/lon)
                if gdf.crs and gdf.crs.to_epsg() != 4326:
                    gdf = gdf.to_crs(epsg=4326)
                self.pois[poi_type] = gdf
                print(f"  [OK] Loaded {poi_type}: {len(gdf)} records")
            else:
                print(f"  [WARN] Not found: {filepath}")
    
    def _parse_building_levels(self, value) -> int:
        """Parse building:levels value to integer."""
        if pd.isna(value) or value is None:
            return 1
        try:
            # Handle cases like "5", "5.0", "5-7" (take first number)
            s = str(value).split("-")[0].split(".")[0].strip()
            return max(1, int(s))
        except (ValueError, TypeError):
            return 1
    
    def _get_centroid(self, geometry) -> Tuple[float, float]:
        """Get centroid of geometry as (lat, lon)."""
        try:
            centroid = geometry.centroid
            return (centroid.y, centroid.x)  # lat, lon
        except Exception:
            return (21.0, 105.8)  # Default Hanoi center
    
    def _calculate_apartment_load(self, row: pd.Series) -> float:
        """Calculate load for apartment building."""
        params = LOAD_PARAMS["apartment"]
        
        floors = self._parse_building_levels(row.get("building:levels"))
        
        # Try to get flats count
        flats = row.get("building:flats")
        if pd.notna(flats):
            try:
                flats = int(flats)
                load = flats * params["mw_per_flat"]
            except (ValueError, TypeError):
                load = floors * params["mw_per_floor"]
        else:
            load = floors * params["mw_per_floor"]
        
        return max(params["min_load"], load)
    
    def _calculate_mall_load(self, row: pd.Series) -> float:
        """Calculate load for mall/shopping center."""
        params = LOAD_PARAMS["mall"]
        
        floors = self._parse_building_levels(row.get("building:levels"))
        load = params["base_mw"] + floors * params["mw_per_floor"]
        
        return max(params["min_load"], load)
    
    def _calculate_entertainment_load(self, row: pd.Series) -> float:
        """Calculate load for entertainment venue."""
        params = LOAD_PARAMS["entertainment"]
        
        # Check amenity or leisure type
        amenity = str(row.get("amenity", "")).lower()
        leisure = str(row.get("leisure", "")).lower()
        
        for venue_type, load in params.items():
            if venue_type in amenity or venue_type in leisure:
                return load
        
        return params["default"]
    
    def _calculate_road_load(self, row: pd.Series, poi_type: str) -> float:
        """Calculate street lighting load based on road length."""
        params = LOAD_PARAMS[poi_type]
        
        try:
            # Calculate length in km from geometry
            geom = row.geometry
            if geom is not None:
                # Need to project to UTM for accurate length calculation
                # Approximate: 1 degree ≈ 111 km at equator
                length_deg = geom.length if hasattr(geom, 'length') else 0
                length_km = length_deg * 111  # Rough approximation
                return length_km * params["mw_per_km"]
        except Exception:
            pass
        
        return 0.01  # Default small load
    
    def _calculate_all_loads(self) -> None:
        """Calculate loads for all POIs."""
        self.poi_loads = []
        
        # Apartments
        if "apartment" in self.pois:
            for idx, row in self.pois["apartment"].iterrows():
                lat, lon = self._get_centroid(row.geometry)
                load = self._calculate_apartment_load(row)
                self.poi_loads.append({
                    "lat": lat,
                    "lon": lon,
                    "load_mw": load,
                    "poi_type": "apartment",
                    "name": row.get("name", f"Apartment_{idx}"),
                })
        
        # Malls
        if "mall" in self.pois:
            for idx, row in self.pois["mall"].iterrows():
                lat, lon = self._get_centroid(row.geometry)
                load = self._calculate_mall_load(row)
                self.poi_loads.append({
                    "lat": lat,
                    "lon": lon,
                    "load_mw": load,
                    "poi_type": "mall",
                    "name": row.get("name", f"Mall_{idx}"),
                })
        
        # Entertainment
        if "entertainment" in self.pois:
            for idx, row in self.pois["entertainment"].iterrows():
                lat, lon = self._get_centroid(row.geometry)
                load = self._calculate_entertainment_load(row)
                self.poi_loads.append({
                    "lat": lat,
                    "lon": lon,
                    "load_mw": load,
                    "poi_type": "entertainment",
                    "name": row.get("name", f"Entertainment_{idx}"),
                })
        
        # Highway lighting (aggregate per segment)
        if "highway" in self.pois:
            for idx, row in self.pois["highway"].iterrows():
                lat, lon = self._get_centroid(row.geometry)
                load = self._calculate_road_load(row, "highway")
                if load > 0.001:  # Only significant loads
                    self.poi_loads.append({
                        "lat": lat,
                        "lon": lon,
                        "load_mw": load,
                        "poi_type": "highway_lighting",
                        "name": row.get("name", f"Highway_{idx}"),
                    })
        
        print(f"\n[OK] Calculated loads for {len(self.poi_loads)} POIs")
        print(f"   Total POI load: {self.get_total_load():.2f} MW")
    
    def get_total_load(self) -> float:
        """Get total load from all POIs."""
        return sum(p["load_mw"] for p in self.poi_loads)
    
    def get_load_by_type(self) -> Dict[str, float]:
        """Get load breakdown by POI type."""
        breakdown = {}
        for p in self.poi_loads:
            ptype = p["poi_type"]
            breakdown[ptype] = breakdown.get(ptype, 0) + p["load_mw"]
        return breakdown
    
    def aggregate_to_buses(
        self,
        bus_coords: List[Dict[str, Any]],
        voltage_kv: float = 22.0,
    ) -> pd.DataFrame:
        """
        Aggregate POI loads to nearest buses.
        
        Args:
            bus_coords: List of dicts with 'idx', 'lat', 'lon', 'name'
            voltage_kv: Only consider buses of this voltage level
            
        Returns:
            DataFrame with load data (bus, p_mw, q_mvar, name, type)
        """
        from collections import defaultdict
        
        # Filter to PRIMARY 22kV buses only (those ending with _22kV, not feeder endpoints)
        # This ensures loads are connected to main transformer output, not feeder endpoints
        buses_22kv = [
            b for b in bus_coords 
            if b.get("vn_kv", 22) == voltage_kv and b.get("name", "").endswith("_22kV")
        ]
        
        if not buses_22kv:
            # Fallback: use all 22kV buses if no primary ones found
            buses_22kv = [b for b in bus_coords if b.get("vn_kv", 22) == voltage_kv]
        
        if not buses_22kv:
            print("[WARN] No 22kV buses found, using all buses")
            buses_22kv = bus_coords
        
        # Aggregate loads per bus
        bus_loads = defaultdict(lambda: {"p_mw": 0, "types": set(), "count": 0})
        
        for poi in self.poi_loads:
            # Find nearest bus
            min_dist = float('inf')
            nearest_bus = None
            
            for bus in buses_22kv:
                dist = np.sqrt(
                    (poi["lat"] - bus["lat"])**2 + 
                    (poi["lon"] - bus["lon"])**2
                )
                if dist < min_dist:
                    min_dist = dist
                    nearest_bus = bus
            
            if nearest_bus:
                bus_idx = nearest_bus["idx"]
                bus_loads[bus_idx]["p_mw"] += poi["load_mw"]
                bus_loads[bus_idx]["types"].add(poi["poi_type"])
                bus_loads[bus_idx]["count"] += 1
                bus_loads[bus_idx]["name"] = nearest_bus.get("name", f"Bus_{bus_idx}")
        
        # Create DataFrame
        loads = []
        for bus_idx, data in bus_loads.items():
            p_mw = round(data["p_mw"], 3)
            # Power factor based on dominant load type
            if "mall" in data["types"] or "entertainment" in data["types"]:
                pf = 0.90  # Commercial
            else:
                pf = 0.92  # Residential
            
            q_mvar = round(p_mw * np.tan(np.arccos(pf)), 3)
            
            loads.append({
                "name": f"LOAD_{data['name'][:15]}",
                "bus": bus_idx,
                "p_mw": p_mw,
                "q_mvar": q_mvar,
                "scaling": 1.0,
                "type": "+".join(sorted(data["types"])),
                "poi_count": data["count"],
            })
        
        loads_df = pd.DataFrame(loads)
        loads_df.index.name = "index"
        
        print(f"\n[OK] Aggregated loads to {len(loads_df)} buses")
        print(f"   Total aggregated load: {loads_df['p_mw'].sum():.2f} MW")
        
        return loads_df
    
    def get_poi_dataframe(self) -> pd.DataFrame:
        """Get all POI loads as DataFrame for analysis."""
        return pd.DataFrame(self.poi_loads)


def generate_loads_from_pois(
    poi_folder: str,
    buses_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convenience function to generate loads from POIs.
    
    Args:
        poi_folder: Path to POI GeoPackages
        buses_df: DataFrame with bus data (must have name, vn_kv, x, y columns)
        
    Returns:
        DataFrame with load data ready for pandapower
    """
    gen = POILoadGenerator(poi_folder)
    
    # Extract bus coordinates for 22kV buses
    bus_coords = []
    for idx, row in buses_df.iterrows():
        bus_coords.append({
            "idx": idx,
            "lat": row.get("y", row.get("lat", 21.0)),
            "lon": row.get("x", row.get("lon", 105.8)),
            "name": row.get("name", f"Bus_{idx}"),
            "vn_kv": row.get("vn_kv", 22.0),
        })
    
    return gen.aggregate_to_buses(bus_coords, voltage_kv=22.0)


if __name__ == "__main__":
    # Test loading
    POI_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "QGIS-Related", "POIs")
    
    print("Testing POI Load Generator...")
    print("=" * 50)
    
    gen = POILoadGenerator(POI_FOLDER)
    
    print("\nLoad breakdown by type:")
    for ptype, load in gen.get_load_by_type().items():
        print(f"  {ptype}: {load:.2f} MW")
    
    print(f"\nTotal: {gen.get_total_load():.2f} MW")
