"""
Distance Matrix Loader for RL Environment

Module để load và sử dụng ma trận khoảng cách trong môi trường 
Reinforcement Learning cho bài toán tối ưu trạm sạc.

Usage:
    from road_network import DistanceMatrixLoader
    
    loader = DistanceMatrixLoader('road_network/data')
    dist = loader.get_distance(node_from, node_to)
    dist = loader.get_route_distance((lat1, lng1), (lat2, lng2))
"""

import os
import pickle
from typing import Tuple, Union, List

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree # type: ignore[attr-defined]


class DistanceMatrixLoader:
    """
    Loader để tra cứu khoảng cách từ ma trận đã tính sẵn.
    
    Thay thế việc tính toán Haversine hoặc gọi API bằng việc 
    tra cứu O(1) từ ma trận khoảng cách đường đi thực.
    """
    
    def __init__(self, data_dir: str = 'data'):
        """
        Khởi tạo loader và load dữ liệu vào RAM.
        
        Args:
            data_dir: Thư mục chứa các file output từ distance_matrix.py
        """
        self.data_dir = data_dir
        
        # Load distance matrix
        matrix_file = os.path.join(data_dir, 'dist_matrix.npy')
        self.dist_matrix = np.load(matrix_file)
        
        # Load node mapping
        mapping_file = os.path.join(data_dir, 'node_mapping.pkl')
        with open(mapping_file, 'rb') as f:
            mapping = pickle.load(f)
        self.node_to_idx = mapping['node_to_idx']
        self.idx_to_node = mapping['idx_to_node']
        
        # Load nodes metadata
        metadata_file = os.path.join(data_dir, 'nodes_metadata.csv')
        self.nodes_df = pd.read_csv(metadata_file)
        
        # Build KD-Tree for nearest node lookup
        coords = self.nodes_df[['lat', 'lng']].values
        self.kdtree = cKDTree(coords)
    
    @property
    def num_nodes(self) -> int:
        """Số lượng nodes trong mạng lưới."""
        return len(self.node_to_idx)
    
    def get_distance(self, node_from: int, node_to: int) -> float:
        """Tra cứu khoảng cách đường đi từ node_from đến node_to (mét)."""
        idx_from = self.node_to_idx[node_from]
        idx_to = self.node_to_idx[node_to]
        return float(self.dist_matrix[idx_from, idx_to])
    
    def get_distance_by_idx(self, idx_from: int, idx_to: int) -> float:
        """Tra cứu khoảng cách theo matrix index (nhanh hơn get_distance)."""
        return float(self.dist_matrix[idx_from, idx_to])
    
    def find_nearest_node(self, lat: float, lng: float) -> int:
        """
        Tìm node gần nhất với tọa độ GPS (luôn trả về 1 node_id).
        Dùng cho distance lookup / RL.
        """
        _, idx = self.kdtree.query((lat, lng), k=1)
        return int(self.nodes_df.iloc[idx]["node_id"])
    
    def find_k_nearest_nodes(self, lat: float, lng: float, k: int) -> List[int]:
        """
        Tìm k node gần nhất với tọa độ GPS.
        ddùng cho phân tích / candidate generation.
        """
        _, indices = self.kdtree.query((lat, lng), k=k)
        return self.nodes_df.iloc[indices]["node_id"].astype(int).tolist()
    
    def get_route_distance( self, coord_from: Tuple[float, float], coord_to: Tuple[float, float]) -> float:
        """
        Tính khoảng cách đường đi giữa 2 tọa độ GPS bất kỳ.
        """
        node_from = self.find_nearest_node(*coord_from)
        node_to = self.find_nearest_node(*coord_to)
        return self.get_distance(node_from, node_to)
    
    def get_node_coords(self, node_id: int) -> Tuple[float, float]:
        """Lấy tọa độ GPS của một node."""
        row = self.nodes_df[self.nodes_df['node_id'] == node_id].iloc[0]
        return float(row['lat']), float(row['lng'])
    
    def get_all_distances_from(self, node_id: int) -> np.ndarray:
        """Lấy tất cả khoảng cách từ một node đến mọi node khác."""
        idx = self.node_to_idx[node_id]
        return self.dist_matrix[idx].copy()
    
if __name__ == "__main__":
    loader = DistanceMatrixLoader("data")

    print("[TEST] Num nodes:", loader.num_nodes)

    # Pick 2 random nodes
    nodes = list(loader.node_to_idx.keys())
    n1, n2 = nodes[0], nodes[1]

    d1 = loader.get_distance(n1, n2)
    d2 = loader.get_distance(n2, n1)

    print(f"[TEST] Distance {n1} -> {n2}: {d1:.1f} m")
    print(f"[TEST] Distance {n2} -> {n1}: {d2:.1f} m")

    assert d1 >= 0
    assert d2 >= 0


