"""
CSPRL Grid Adapter

Module adapter để tích hợp power_grid với hệ thống CSPRL.
Cho phép mở rộng node features với dữ liệu lưới điện và tính toán grid penalty.

Usage:
    from power_grid import CSPRLGridAdapter
    
    adapter = CSPRLGridAdapter("power_grid/data/hanoi_citywide")
    extended_nodes = adapter.extend_node_features(csprl_node_list)
"""

import os
from typing import List, Dict, Tuple, Optional, Any
from functools import lru_cache
import math

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from .grid_loader import GridLoader


# =============================================================================
# CONFIGURATION - Dựa trên nghiên cứu thực tế
# =============================================================================

# Công suất trạm sạc EV
# Nghiên cứu: DC fast charger 50-350 kW mỗi cổng
# Giả định: Trạm sạc tiêu biểu có 4 cổng DC fast (50-100 kW/cổng)
# => Tổng: 0.2 - 0.4 MW, chọn 0.35 MW làm mặc định
DEFAULT_EV_STATION_POWER_MW = 0.35

# Ngưỡng khoảng cách đến lưới điện (km)
# Nghiên cứu: Chi phí đấu nối tăng đáng kể khi >1km trong đô thị
# Penalty bắt đầu từ 1km, tối đa tại 3km
DISTANCE_THRESHOLD_MIN_KM = 1.0  # Bắt đầu penalty
DISTANCE_THRESHOLD_MAX_KM = 3.0  # Penalty tối đa

# Penalty weights
PENALTY_DISTANCE_WEIGHT = 0.3    # Weight cho distance penalty
PENALTY_CAPACITY_WEIGHT = 0.7   # Weight cho capacity penalty

# Capacity margin - dự phòng 20% cho growth
CAPACITY_SAFETY_MARGIN = 0.2


class CSPRLGridAdapter:
    """
    Adapter để tích hợp power_grid module với CSPRL node format.
    
    Cung cấp:
    1. Mở rộng node features với dữ liệu lưới điện
    2. Kiểm tra khả năng đặt trạm sạc
    3. Tính toán grid penalty cho reward function
    
    Attributes:
        loader: GridLoader instance
        ev_station_power_mw: Công suất yêu cầu cho mỗi trạm sạc (MW)
        _bus_cache: Cache mapping từ tọa độ đến bus info
    """
    
    def __init__(
        self, 
        grid_data_folder: str,
        ev_station_power_mw: float = DEFAULT_EV_STATION_POWER_MW,
        auto_run_power_flow: bool = True
    ):
        """
        Khởi tạo adapter.
        
        Args:
            grid_data_folder: Đường dẫn đến folder chứa CSV lưới điện
            ev_station_power_mw: Công suất yêu cầu cho trạm sạc EV (MW)
            auto_run_power_flow: Tự động chạy power flow khi khởi tạo
        """
        self.grid_data_folder = grid_data_folder
        self.ev_station_power_mw = ev_station_power_mw
        self._bus_cache: Dict[Tuple[float, float], Dict] = {}
        
        # Khởi tạo GridLoader
        self.loader = GridLoader(grid_data_folder)
        self.loader.create_network()
        
        if auto_run_power_flow:
            self.loader.run_power_flow()
    
    def _get_bus_info(self, lat: float, lon: float) -> Dict:
        """
        Lấy thông tin bus gần nhất với caching.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Dict với bus_idx, distance_km, available_mw, bus_name
        """
        # Round để tạo cache key (độ chính xác ~11m)
        cache_key = (round(lat, 4), round(lon, 4))
        
        if cache_key not in self._bus_cache:
            result = self.loader.find_nearest_bus(lat, lon, voltage_kv=22.0)
            self._bus_cache[cache_key] = result
            
        return self._bus_cache[cache_key]
    
    def extend_node_features(self, node_list: List) -> List:
        """
        Thêm các thuộc tính lưới điện vào danh sách node CSPRL.
        
        Args:
            node_list: Danh sách node theo format CSPRL
                       [(node_id, {'x': lon, 'y': lat, ...}), ...]
        
        Returns:
            node_list với thêm các key:
            - grid_bus_idx: Index của bus 22kV gần nhất
            - grid_distance_km: Khoảng cách đến bus (km)
            - grid_available_mw: Công suất còn lại tại bus (MW)
            - grid_feasible: True nếu đủ capacity cho trạm sạc
            
        Example:
            >>> adapter = CSPRLGridAdapter("power_grid/data/hanoi_citywide")
            >>> nodes = [(123, {'x': 105.82, 'y': 21.02, 'demand': 0.5})]
            >>> extended = adapter.extend_node_features(nodes)
            >>> print(extended[0][1]['grid_distance_km'])
            0.45
        """
        extended_nodes = []
        
        for node_id, attrs in node_list:
            # Lấy tọa độ từ node attributes
            lat = attrs.get('y', 0)
            lon = attrs.get('x', 0)
            
            # Lấy thông tin lưới điện
            bus_info = self._get_bus_info(lat, lon)
            
            # Xác định tính khả thi
            available_mw = bus_info.get('available_mw', 0)
            required_mw = self.ev_station_power_mw * (1 + CAPACITY_SAFETY_MARGIN)
            feasible = available_mw >= required_mw
            
            # Thêm các thuộc tính mới
            new_attrs = attrs.copy()
            new_attrs['grid_bus_idx'] = bus_info.get('bus_idx', -1)
            new_attrs['grid_distance_km'] = bus_info.get('distance_km', float('inf'))
            new_attrs['grid_available_mw'] = available_mw
            new_attrs['grid_feasible'] = feasible
            
            extended_nodes.append((node_id, new_attrs))
        
        return extended_nodes
    
    def check_feasibility(self, node: Tuple) -> Dict:
        """
        Kiểm tra khả năng đặt trạm sạc tại một node.
        
        Args:
            node: CSPRL node tuple (node_id, attrs_dict)
            
        Returns:
            Dict với:
            - feasible: bool
            - reason: str (mô tả lý do nếu không khả thi)
            - available_mw: float
            - required_mw: float
            - distance_km: float
            - penalty: float (penalty score nếu không khả thi)
        """
        node_id, attrs = node
        lat = attrs.get('y', 0)
        lon = attrs.get('x', 0)
        
        bus_info = self._get_bus_info(lat, lon)
        
        available_mw = bus_info.get('available_mw', 0)
        distance_km = bus_info.get('distance_km', float('inf'))
        required_mw = self.ev_station_power_mw * (1 + CAPACITY_SAFETY_MARGIN)
        
        # Kiểm tra các điều kiện
        reasons = []
        penalty = 0.0
        
        # Điều kiện 1: Công suất
        if available_mw < required_mw:
            shortage = required_mw - available_mw
            reasons.append(f"Thiếu công suất: cần {required_mw:.2f} MW, còn {available_mw:.2f} MW")
            # Soft penalty tỷ lệ với mức thiếu hụt
            penalty += PENALTY_CAPACITY_WEIGHT * min(1.0, shortage / required_mw)
        
        # Điều kiện 2: Khoảng cách
        if distance_km > DISTANCE_THRESHOLD_MAX_KM:
            reasons.append(f"Quá xa lưới điện: {distance_km:.2f} km (>= {DISTANCE_THRESHOLD_MAX_KM} km)")
            penalty += PENALTY_DISTANCE_WEIGHT * 1.0  # Max distance penalty
        elif distance_km > DISTANCE_THRESHOLD_MIN_KM:
            # Penalty tuyến tính từ min đến max threshold
            dist_ratio = (distance_km - DISTANCE_THRESHOLD_MIN_KM) / (DISTANCE_THRESHOLD_MAX_KM - DISTANCE_THRESHOLD_MIN_KM)
            penalty += PENALTY_DISTANCE_WEIGHT * dist_ratio
        
        feasible = len(reasons) == 0
        reason = "; ".join(reasons) if reasons else "OK"
        
        return {
            'feasible': feasible,
            'reason': reason,
            'available_mw': available_mw,
            'required_mw': required_mw,
            'distance_km': distance_km,
            'penalty': penalty,
            'bus_name': bus_info.get('bus_name', 'Unknown')
        }
    
    def calculate_grid_penalty(self, station_nodes: List) -> float:
        """
        Tính tổng penalty từ ràng buộc lưới điện cho một charging plan.
        
        Args:
            station_nodes: List các node đã đặt trạm sạc
                          [(node_id, attrs_dict), ...]
            
        Returns:
            Penalty score (số âm hoặc 0)
            - 0.0 = Tất cả trạm đều khả thi
            - Số âm = Có vi phạm ràng buộc
        """
        total_penalty = 0.0
        
        for node in station_nodes:
            result = self.check_feasibility(node)
            total_penalty -= result['penalty']
        
        return total_penalty
    
    def get_grid_summary_for_nodes(self, node_list: List) -> Any:
        """
        Trả về thông tin tổng hợp về lưới điện cho tất cả nodes.
        
        Args:
            node_list: Danh sách node CSPRL
            
        Returns:
            pandas.DataFrame nếu pandas có sẵn, hoặc List[Dict]
        """
        summary_data = []
        
        for node_id, attrs in node_list:
            lat = attrs.get('y', 0)
            lon = attrs.get('x', 0)
            
            bus_info = self._get_bus_info(lat, lon)
            feasibility = self.check_feasibility((node_id, attrs))
            
            summary_data.append({
                'node_id': node_id,
                'lat': lat,
                'lon': lon,
                'demand': attrs.get('demand', 0),
                'bus_idx': bus_info.get('bus_idx', -1),
                'bus_name': bus_info.get('bus_name', 'Unknown'),
                'distance_km': bus_info.get('distance_km', float('inf')),
                'available_mw': bus_info.get('available_mw', 0),
                'feasible': feasibility['feasible'],
                'reason': feasibility['reason'],
                'penalty': feasibility['penalty']
            })
        
        if PANDAS_AVAILABLE:
            return pd.DataFrame(summary_data)
        return summary_data
    
    def get_all_22kv_buses(self) -> List[Dict]:
        """
        Lấy danh sách tất cả bus 22kV với tọa độ.
        
        Returns:
            List[Dict] với mỗi phần tử có:
            - bus_idx: int
            - lat: float  
            - lon: float
            - name: str
            - available_mw: float
        """
        if self.loader.net is None:
            return []
        
        buses_22kv = []
        net = self.loader.net
        
        # Lọc các bus 22kV
        for idx, row in net.bus.iterrows():
            if abs(row['vn_kv'] - 22.0) < 0.5:  # Tolerance for floating point
                # Lấy available capacity
                capacity_info = self.loader.get_available_capacity(idx)
                
                buses_22kv.append({
                    'bus_idx': idx,
                    'lat': row.get('geodata', (0, 0))[0] if row.get('geodata') else 0,
                    'lon': row.get('geodata', (0, 0))[1] if row.get('geodata') else 0,
                    'name': row.get('name', f'Bus_{idx}'),
                    'available_mw': capacity_info.get('available_mw', 0)
                })
        
        return buses_22kv
    
    def clear_cache(self):
        """Xóa cache để cập nhật dữ liệu mới sau khi có thay đổi lưới."""
        self._bus_cache.clear()
    
    def update_power_flow(self):
        """Chạy lại power flow và xóa cache."""
        self.loader.run_power_flow()
        self.clear_cache()
    
    def add_ev_station_load(self, lat: float, lon: float, power_mw: Optional[float] = None):
        """
        Thêm tải trạm sạc EV vào lưới điện và cập nhật power flow.
        
        Args:
            lat: Latitude của trạm sạc
            lon: Longitude của trạm sạc
            power_mw: Công suất (MW), mặc định dùng ev_station_power_mw
        """
        try:
            import pandapower as pp
        except ImportError:
            raise ImportError("pandapower required for adding loads")
        
        if power_mw is None:
            power_mw = self.ev_station_power_mw
        
        bus_info = self._get_bus_info(lat, lon)
        bus_idx = bus_info.get('bus_idx')
        
        if bus_idx is not None and bus_idx >= 0:
            pp.create_load(
                self.loader.net,
                bus=bus_idx,
                p_mw=power_mw,
                name=f"EV_Station_{lat:.4f}_{lon:.4f}"
            )
            self.update_power_flow()
    
    def get_config(self) -> Dict:
        """Trả về cấu hình hiện tại của adapter."""
        return {
            'ev_station_power_mw': self.ev_station_power_mw,
            'distance_threshold_min_km': DISTANCE_THRESHOLD_MIN_KM,
            'distance_threshold_max_km': DISTANCE_THRESHOLD_MAX_KM,
            'capacity_safety_margin': CAPACITY_SAFETY_MARGIN,
            'penalty_distance_weight': PENALTY_DISTANCE_WEIGHT,
            'penalty_capacity_weight': PENALTY_CAPACITY_WEIGHT,
            'grid_data_folder': self.grid_data_folder
        }


def create_adapter_for_location(
    location: str,
    base_path: str = None
) -> CSPRLGridAdapter:
    """
    Factory function để tạo adapter cho một location cụ thể.
    
    Args:
        location: Tên location (e.g., "DongDa", "HoanKiem")
        base_path: Đường dẫn cơ sở đến thư mục data
        
    Returns:
        CSPRLGridAdapter configured cho location
    """
    if base_path is None:
        # Default to CSPRL/data
        base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        
    grid_folder = os.path.join(base_path, f"hanoi_{location.lower()}")
    
    # Fallback to citywide if location-specific data not available
    if not os.path.exists(grid_folder):
        grid_folder = os.path.join(base_path, "hanoi_citywide")
    
    return CSPRLGridAdapter(grid_folder)
