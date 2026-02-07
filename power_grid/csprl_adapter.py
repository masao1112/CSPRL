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
from pandapower.plotting.plotly import simple_plotly

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
DISTANCE_THRESHOLD_MIN_KM = 0.4  # Bắt đầu penalty
DISTANCE_THRESHOLD_MAX_KM = 1.5  # Penalty tối đa

# Penalty weights
PENALTY_DISTANCE_WEIGHT = 0.3    # Weight cho distance penalty
PENALTY_CAPACITY_WEIGHT = 0.7   # Weight cho capacity penalty

# Capacity margin - dự phòng 20% cho growth
CAPACITY_SAFETY_MARGIN = 0.2

# Terminate thresholds (Phase 2: Hybrid approach)
TERMINATE_BUS_OVERLOAD_THRESHOLD = 1.5    # 150% - hard terminate
TERMINATE_TRAFO_LOADING_THRESHOLD = 100.0  # 100% - hard terminate
TERMINATE_PENALTY = -5.0                   # Penalty when terminate due to grid


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
        net = self.loader.create_network()
        simple_plotly(net, on_map=True)

        if auto_run_power_flow:
            self.loader.run_power_flow()
        
        # Build bus-to-trafo mapping for 22kV buses
        self._bus_to_trafo: Dict[int, int] = {}  # bus_idx -> trafo_idx
        self._trafo_capacity: Dict[int, float] = {}  # trafo_idx -> sn_mva
        self._trafo_buses: Dict[int, List[int]] = {}  # trafo_idx -> [bus_idx, ...]
        self._build_bus_trafo_mapping()
    
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
    
    def _build_bus_trafo_mapping(self):
        """
        Build mapping từ bus 22kV đến transformer 110kV/22kV.
        
        Đọc trafo.csv và tạo:
        - _bus_to_trafo: bus_idx -> trafo_idx
        - _trafo_capacity: trafo_idx -> sn_mva
        - _trafo_buses: trafo_idx -> [bus_idx, ...]
        """
        import os
        trafo_file = os.path.join(self.grid_data_folder, "trafo.csv")
        
        if not os.path.exists(trafo_file):
            return
        
        if not PANDAS_AVAILABLE:
            return
            
        trafo_df = pd.read_csv(trafo_file)
        
        # Lọc chỉ lấy trafo 110kV/22kV (vn_lv_kv ≈ 22-23)
        trafo_22kv = trafo_df[(trafo_df['vn_lv_kv'] >= 20) & (trafo_df['vn_lv_kv'] <= 25)]
        
        for _, row in trafo_22kv.iterrows():
            trafo_idx = int(row['index'])
            lv_bus = int(row['lv_bus'])
            sn_mva = float(row['sn_mva'])
            
            self._bus_to_trafo[lv_bus] = trafo_idx
            self._trafo_capacity[trafo_idx] = sn_mva
            
            if trafo_idx not in self._trafo_buses:
                self._trafo_buses[trafo_idx] = []
            self._trafo_buses[trafo_idx].append(lv_bus)
    
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
            
            # Thêm các thuộc tính mới
            new_attrs = attrs.copy()
            new_attrs['grid_bus_idx'] = bus_info.get('bus_idx', -1)
            new_attrs['grid_distance_km'] = bus_info.get('distance_km', float('inf'))
            new_attrs['grid_available_mw'] = bus_info.get('available_mw', 0)
            # Removed 'grid_feasible' to avoid misleading agents with hardcoded assumptions
            
            extended_nodes.append((node_id, new_attrs))
        
        return extended_nodes
    
    def check_feasibility(self, node: Tuple, actual_power_mw: float = None) -> Dict:
        """
        Kiểm tra khả năng đặt trạm sạc tại một node.
        
        Args:
            node: CSPRL node tuple (node_id, attrs_dict)
            actual_power_mw: Công suất thực tế của trạm (BẮT BUỘC nếu khác 0)
            
        Returns:
            Dict với:
            - feasible: bool
            - reason: str (mô tả lý do nếu không khả thi)
            - available_mw: float
            - required_mw: float
            - distance_km: float
            - penalty: float (penalty score nếu không khả thi)
            - bus_idx: int (index của bus được kết nối)
        """
        node_id, attrs = node
        lat = attrs.get('y', 0)
        lon = attrs.get('x', 0)
        
        bus_info = self._get_bus_info(lat, lon)
        
        available_mw = bus_info.get('available_mw', 0)
        distance_km = bus_info.get('distance_km', float('inf'))
        
        if actual_power_mw is not None:
             required_mw = actual_power_mw * (1 + CAPACITY_SAFETY_MARGIN)
        else:
             # Fallback nhưng cảnh báo ngầm (nên tránh dùng default)
             required_mw = self.ev_station_power_mw * (1 + CAPACITY_SAFETY_MARGIN)

        
        # Kiểm tra các điều kiện
        reasons = []
        penalty = 0.0
        
        # Điều kiện 1: Công suất (Local check only - chưa tính cumulative)
        if required_mw > 0 and available_mw < required_mw:
            shortage = required_mw - available_mw
            # reasons.append(f"Thiếu công suất: cần {required_mw:.2f} MW, còn {available_mw:.2f} MW")
            # NOTE: We soften the local check because cumulative check handles the strict penalty
            # But we still apply some penalty here to guide individual placement
            penalty += PENALTY_CAPACITY_WEIGHT * min(1.0, shortage / required_mw)
        
        # Điều kiện 2: Khoảng cách
        if distance_km > DISTANCE_THRESHOLD_MAX_KM:
            reasons.append(f"Quá xa lưới điện: {distance_km:.2f} km (>= {DISTANCE_THRESHOLD_MAX_KM} km)")
            penalty += PENALTY_DISTANCE_WEIGHT * 1.0
        elif distance_km > DISTANCE_THRESHOLD_MIN_KM:
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
            'bus_name': bus_info.get('bus_name', 'Unknown'),
            'bus_idx': bus_info.get('bus_idx', -1)
        }
    
    def calculate_grid_penalty(self, station_nodes: List[Any]) -> float:
        """
        Tính tổng penalty từ ràng buộc lưới điện cho một charging plan, có tính đến CUMULATIVE LOAD.
        
        Args:
            station_nodes: List các phần tử. Mỗi phần tử có thể là:
                           - Node tuple: (node_id, attrs_dict) -> Dùng công suất mặc định
                           - Tuple (Node, capacity_mw): -> Dùng công suất thực tế
            
        Returns:
            Penalty score (số âm hoặc 0)
            - 0.0 = Tất cả trạm đều khả thi
            - Số âm = Có vi phạm ràng buộc
        """
        total_penalty = 0.0
        bus_loads = {} # bus_idx -> {'required': 0.0, 'available': 0.0}
        
        for item in station_nodes:
            # Check input format
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], (int, float)):
                # Format: (Node, capacity_mw)
                node, capacity_mw = item
                result = self.check_feasibility(node, actual_power_mw=capacity_mw)
            else:
                # Format: Node only
                node = item
                # Fallback to default but warn if possible
                result = self.check_feasibility(node)
            
            # 1. Add Distance Penalty (local)
            # We assume check_feasibility calculates distance penalty correctly in 'penalty' field
            # BUT it also calculates capacity penalty locally.
            # We want to separate them if we do cumulative check.
            
            # Let's extract distance component manually to be safe
            dist_km = result['distance_km']
            dist_penalty = 0.0
            if dist_km > DISTANCE_THRESHOLD_MAX_KM:
                dist_penalty = PENALTY_DISTANCE_WEIGHT * 1.0
            elif dist_km > DISTANCE_THRESHOLD_MIN_KM:
                dist_ratio = (dist_km - DISTANCE_THRESHOLD_MIN_KM) / (DISTANCE_THRESHOLD_MAX_KM - DISTANCE_THRESHOLD_MIN_KM)
                dist_penalty = PENALTY_DISTANCE_WEIGHT * dist_ratio
                
            total_penalty -= dist_penalty
            
            # 2. Accumulate Load for Bus
            bus_idx = result.get('bus_idx', -1)
            if bus_idx != -1:
                if bus_idx not in bus_loads:
                    bus_loads[bus_idx] = {'required': 0.0, 'available': result['available_mw']}
                bus_loads[bus_idx]['required'] += result['required_mw']

        # 3. Check Cumulative Capacity
        for bus_idx, data in bus_loads.items():
            required = data['required']
            available = data['available']
            
            if required > available and required > 0:
                shortage = required - available
                # Penalty proportional to overload ratio
                ratio = shortage / required
                bus_penalty = PENALTY_CAPACITY_WEIGHT * min(1.0, ratio)
                total_penalty -= bus_penalty
        
        return total_penalty
    
    def _get_trafo_loading(self, bus_loads: Dict[int, Dict]) -> Dict[int, float]:
        """
        Tính transformer loading dựa trên bus loads.
        
        Args:
            bus_loads: Dict {bus_idx: {'required': MW, 'available': MW}}
            
        Returns:
            Dict {trafo_idx: loading_percent}
        """
        trafo_loading = {}
        
        for trafo_idx, capacity_mva in self._trafo_capacity.items():
            if capacity_mva <= 0:
                continue
                
            # Tổng load của tất cả buses thuộc trafo này
            total_load_mw = 0.0
            for bus_idx in self._trafo_buses.get(trafo_idx, []):
                if bus_idx in bus_loads:
                    total_load_mw += bus_loads[bus_idx]['required']
            
            # Loading % = (load_mw / capacity_mva) * 100
            # Giả định power factor = 0.9 để convert MVA -> MW
            capacity_mw = capacity_mva * 0.9
            loading_pct = (total_load_mw / capacity_mw) * 100.0 if capacity_mw > 0 else 0.0
            trafo_loading[trafo_idx] = loading_pct
        
        return trafo_loading
    
    def check_grid_terminate(self, station_nodes: List[Any]) -> Tuple[bool, str, float]:
        """
        Kiểm tra xem grid có bị overload nghiêm trọng không.
        
        Hybrid logic:
        1. Hard terminate nếu bất kỳ bus > 150% capacity
        2. Hard terminate nếu bất kỳ trafo > 100% loading
        
        Args:
            station_nodes: List các (node, power_mw) hoặc node tuples
            
        Returns:
            Tuple (should_terminate, reason, penalty)
        """
        # Collect bus loads như calculate_grid_penalty
        bus_loads = {}
        
        for item in station_nodes:
            if isinstance(item, tuple) and len(item) >= 2:
                node = item[0]
                actual_power_mw = item[1] if isinstance(item[1], (int, float)) else None
            else:
                node = item
                actual_power_mw = None
            
            if not isinstance(node, tuple) or len(node) < 2:
                continue
            
            node_id, attrs = node[0], node[1]
            lat = attrs.get('y', 0)
            lon = attrs.get('x', 0)
            
            result = self.check_feasibility((node_id, attrs), actual_power_mw)
            
            bus_idx = result.get('bus_idx', -1)
            if bus_idx != -1:
                if bus_idx not in bus_loads:
                    bus_loads[bus_idx] = {'required': 0.0, 'available': result['available_mw']}
                bus_loads[bus_idx]['required'] += result['required_mw']
        
        # Check 1: Bus overload > 150%
        for bus_idx, data in bus_loads.items():
            if data['available'] > 0:
                load_ratio = data['required'] / data['available']
                if load_ratio > TERMINATE_BUS_OVERLOAD_THRESHOLD:
                    return (
                        True, 
                        f"Bus {bus_idx} overload: {load_ratio*100:.1f}% (threshold: {TERMINATE_BUS_OVERLOAD_THRESHOLD*100:.0f}%)",
                        TERMINATE_PENALTY
                    )
        
        # Check 2: Trafo loading > 100%
        trafo_loading = self._get_trafo_loading(bus_loads)
        for trafo_idx, loading_pct in trafo_loading.items():
            if loading_pct > TERMINATE_TRAFO_LOADING_THRESHOLD:
                return (
                    True,
                    f"Transformer {trafo_idx} overload: {loading_pct:.1f}% (threshold: {TERMINATE_TRAFO_LOADING_THRESHOLD:.0f}%)",
                    TERMINATE_PENALTY
                )
        
        return (False, "", 0.0)
    
    def get_grid_metrics(self, station_nodes: List[Any]) -> Dict[str, float]:
        """
        Trả về các metrics về trạng thái grid.
        
        Args:
            station_nodes: List các (node, power_mw) hoặc node tuples
            
        Returns:
            Dict với các metrics:
            - grid_utilization: % tổng load / tổng capacity
            - max_bus_load_ratio: Bus có tải cao nhất (ratio)
            - max_trafo_loading: Trafo có loading cao nhất (%)
            - capacity_violations: Số bus bị quá tải (> 100%)
            - trafo_violations: Số trafo bị quá tải (> 100%)
            - avg_distance_km: Khoảng cách TB đến grid
            - total_grid_penalty: Tổng penalty từ calculate_grid_penalty
        """
        # Collect data
        bus_loads = {}
        distances = []
        
        for item in station_nodes:
            if isinstance(item, tuple) and len(item) >= 2:
                node = item[0]
                actual_power_mw = item[1] if isinstance(item[1], (int, float)) else None
            else:
                node = item
                actual_power_mw = None
            
            if not isinstance(node, tuple) or len(node) < 2:
                continue
            
            node_id, attrs = node[0], node[1]
            result = self.check_feasibility((node_id, attrs), actual_power_mw)
            
            distances.append(result['distance_km'])
            
            bus_idx = result.get('bus_idx', -1)
            if bus_idx != -1:
                if bus_idx not in bus_loads:
                    bus_loads[bus_idx] = {'required': 0.0, 'available': result['available_mw']}
                bus_loads[bus_idx]['required'] += result['required_mw']
        
        # Calculate metrics
        total_required = sum(d['required'] for d in bus_loads.values())
        total_available = sum(d['available'] for d in bus_loads.values())
        
        grid_utilization = (total_required / total_available * 100) if total_available > 0 else 0.0
        
        max_bus_load_ratio = 0.0
        capacity_violations = 0
        for data in bus_loads.values():
            if data['available'] > 0:
                ratio = data['required'] / data['available']
                max_bus_load_ratio = max(max_bus_load_ratio, ratio)
                if ratio > 1.0:
                    capacity_violations += 1
        
        trafo_loading = self._get_trafo_loading(bus_loads)
        max_trafo_loading = max(trafo_loading.values()) if trafo_loading else 0.0
        trafo_violations = sum(1 for loading in trafo_loading.values() if loading > 100.0)
        
        avg_distance = sum(distances) / len(distances) if distances else 0.0
        
        total_penalty = self.calculate_grid_penalty(station_nodes)
        
        return {
            'grid_utilization': round(grid_utilization, 2),
            'max_bus_load_ratio': round(max_bus_load_ratio, 3),
            'max_trafo_loading': round(max_trafo_loading, 2),
            'capacity_violations': capacity_violations,
            'trafo_violations': trafo_violations,
            'avg_distance_km': round(avg_distance, 3),
            'total_grid_penalty': round(total_penalty, 4),
        }
    
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
    
    def add_ev_station_load(self, lat: float, lon: float, power_mw: float):
        """
        Thêm tải trạm sạc EV vào lưới điện và cập nhật power flow.
        
        Args:
            lat: Latitude của trạm sạc
            lon: Longitude của trạm sạc
            power_mw: Công suất (MW) - BẮT BUỘC
        """
        try:
            import pandapower as pp
        except ImportError:
            raise ImportError("pandapower required for adding loads")
        
        # if power_mw is None:
        #     power_mw = self.ev_station_power_mw
        
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
