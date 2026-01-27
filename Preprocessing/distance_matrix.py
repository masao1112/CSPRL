"""
Hanoi Road Network Distance Matrix Calculator

Script tính toán ma trận khoảng cách ngắn nhất giữa các nút trong mạng lưới 
đường bộ Hà Nội để sử dụng trong môi trường Reinforcement Learning.

Usage:
    python hanoi_distance_matrix.py --place "Dong Da District, Hanoi, Vietnam" --processes 14
    python hanoi_distance_matrix.py --place "Hanoi, Vietnam" --processes 14
"""

import os
import json
import pickle
import argparse
from functools import partial
from multiprocessing import Pool, cpu_count
from datetime import datetime

import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox
from tqdm import tqdm


def load_or_download_graph(place_name: str, network_type: str = 'drive', cache_dir: str = 'cache'):
    """
    Tải đồ thị mạng lưới đường bộ từ OSMnx hoặc load từ cache.
    
    Args:
        place_name: Tên địa điểm (vd: "Dong Da District, Hanoi, Vietnam")
        network_type: Loại mạng lưới ('drive', 'walk', 'bike', 'all')
        cache_dir: Thư mục cache
        
    Returns:
        networkx.MultiDiGraph: Đồ thị mạng lưới đường bộ
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    # Tạo tên file cache từ place_name
    cache_name = place_name.lower().replace(' ', '_').replace(',', '').replace('.', '')
    cache_file = os.path.join(cache_dir, f"graph_{cache_name}_{network_type}.graphml")
    
    if os.path.exists(cache_file):
        print(f"[INFO] Loading graph from cache: {cache_file}")
        G = ox.load_graphml(cache_file)
    else:
        print(f"[INFO] Downloading graph for: {place_name}")
        G = ox.graph_from_place(place_name, network_type=network_type)
        
        # Thêm các thuộc tính cần thiết
        print("[INFO] Adding edge speeds and travel times...")
        G = ox.add_edge_speeds(G)
        G = ox.add_edge_travel_times(G)
        
        # Lưu cache
        print(f"[INFO] Saving graph to cache: {cache_file}")
        ox.save_graphml(G, cache_file)
    
    return G


def preprocess_nodes(G):
    """
    Tiền xử lý danh sách các nút và tạo bảng ánh xạ.
    
    Args:
        G: networkx.MultiDiGraph
        
    Returns:
        tuple: (nodes_list, node_to_idx, idx_to_node, nodes_gdf)
    """
    print("[INFO] Preprocessing nodes...")
    
    # Lấy danh sách nodes
    nodes_list = list(G.nodes())
    num_nodes = len(nodes_list)
    
    # Tạo mapping
    node_to_idx = {node: idx for idx, node in enumerate(nodes_list)}
    idx_to_node = {idx: node for idx, node in enumerate(nodes_list)}
    
    # Lấy GeoDataFrame của nodes
    nodes_gdf = ox.graph_to_gdfs(G, edges=False)
    
    print(f"[INFO] Total nodes: {num_nodes}")
    print(f"[INFO] Node ID range: {min(nodes_list)} to {max(nodes_list)}")
    
    return nodes_list, node_to_idx, idx_to_node, nodes_gdf


def compute_distance_row(source_node, G, nodes_list, node_to_idx):
    """
    Tính khoảng cách ngắn nhất từ 1 node nguồn đến tất cả các node khác.
    
    Args:
        source_node: Node nguồn
        G: Đồ thị
        nodes_list: Danh sách tất cả nodes
        node_to_idx: Mapping node_id -> index
        
    Returns:
        tuple: (source_idx, distance_row)
    """
    num_nodes = len(nodes_list)
    dist_row = np.full(num_nodes, np.inf, dtype=np.float32)
    
    try:
        # Tính shortest path từ source đến tất cả các nodes
        distances = nx.single_source_dijkstra_path_length(G, source_node, weight='length')
        
        for target_node, dist in distances.items():
            if target_node in node_to_idx:
                target_idx = node_to_idx[target_node]
                dist_row[target_idx] = dist
    except nx.NetworkXError:
        # Node không thể đến được từ source
        pass
    
    source_idx = node_to_idx[source_node]
    return source_idx, dist_row


def compute_distance_matrix(G, nodes_list, node_to_idx, num_processes=None):
    """
    Tính ma trận khoảng cách NxN sử dụng multiprocessing.
    
    Args:
        G: Đồ thị mạng lưới
        nodes_list: Danh sách nodes
        node_to_idx: Mapping node_id -> index
        num_processes: Số processes (mặc định = số CPU cores)
        
    Returns:
        numpy.ndarray: Ma trận khoảng cách NxN (float32)
    """
    if num_processes is None:
        num_processes = cpu_count()
    
    num_nodes = len(nodes_list)
    print(f"[INFO] Computing {num_nodes}x{num_nodes} distance matrix...")
    print(f"[INFO] Using {num_processes} processes")
    print(f"[INFO] Estimated matrix size: {num_nodes * num_nodes * 4 / (1024**3):.2f} GB")
    
    # Khởi tạo ma trận
    dist_matrix = np.full((num_nodes, num_nodes), np.inf, dtype=np.float32)
    
    # Tạo partial function với các tham số cố định
    compute_func = partial(compute_distance_row, G=G, nodes_list=nodes_list, node_to_idx=node_to_idx)
    
    # Chạy multiprocessing với progress bar
    start_time = datetime.now()
    
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(compute_func, nodes_list),
            total=num_nodes,
            desc="Computing distances",
            unit="nodes"
        ))
    
    # Tổng hợp kết quả vào ma trận
    print("[INFO] Assembling distance matrix...")
    for source_idx, dist_row in results:
        dist_matrix[source_idx] = dist_row
    
    elapsed = datetime.now() - start_time
    print(f"[INFO] Computation completed in {elapsed}")
    
    return dist_matrix


def validate_matrix(dist_matrix):
    """
    Kiểm tra tính hợp lệ của ma trận khoảng cách.
    
    Args:
        dist_matrix: Ma trận khoảng cách
        
    Returns:
        dict: Kết quả validation
    """
    num_nodes = dist_matrix.shape[0]
    
    # Kiểm tra đường chéo = 0
    diagonal = np.diag(dist_matrix)
    diag_zeros = np.sum(diagonal == 0)
    
    # Kiểm tra giá trị âm
    negative_count = np.sum(dist_matrix < 0)
    
    # Kiểm tra infinity
    inf_count = np.sum(np.isinf(dist_matrix))
    inf_ratio = inf_count / (num_nodes * num_nodes) * 100
    
    # Thống kê giá trị hữu hạn
    finite_values = dist_matrix[np.isfinite(dist_matrix)]
    
    validation = {
        'num_nodes': num_nodes,
        'diagonal_zeros': int(diag_zeros),
        'diagonal_correct': bool(diag_zeros == num_nodes),
        'negative_count': int(negative_count),
        'no_negatives': bool(negative_count == 0),
        'inf_count': int(inf_count),
        'inf_ratio_percent': round(float(inf_ratio), 2),
        'min_distance': float(np.min(finite_values)) if len(finite_values) > 0 else None,
        'max_distance': float(np.max(finite_values)) if len(finite_values) > 0 else None,
        'mean_distance': float(np.mean(finite_values)) if len(finite_values) > 0 else None,
        'median_distance': float(np.median(finite_values)) if len(finite_values) > 0 else None,
    }
    
    print("\n[VALIDATION RESULTS]")
    print(f"  Nodes: {validation['num_nodes']}")
    print(f"  Diagonal = 0: {validation['diagonal_correct']} ({diag_zeros}/{num_nodes})")
    print(f"  No negatives: {validation['no_negatives']}")
    print(f"  Inf values: {validation['inf_count']} ({validation['inf_ratio_percent']}%)")
    if validation['mean_distance']:
        print(f"  Distance range: {validation['min_distance']:.1f}m - {validation['max_distance']:.1f}m")
        print(f"  Mean distance: {validation['mean_distance']:.1f}m")
    
    return validation


def save_outputs(dist_matrix, nodes_gdf, node_to_idx, idx_to_node, validation, output_dir='data'):
    """
    Lưu ma trận và metadata xuống ổ cứng.
    
    Args:
        dist_matrix: Ma trận khoảng cách
        nodes_gdf: GeoDataFrame của nodes
        node_to_idx: Mapping node_id -> index
        idx_to_node: Mapping index -> node_id
        validation: Kết quả validation
        output_dir: Thư mục output
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Lưu ma trận
    matrix_file = os.path.join(output_dir, 'dist_matrix.npy')
    print(f"[INFO] Saving distance matrix to {matrix_file}")
    np.save(matrix_file, dist_matrix)
    
    # 2. Lưu node mapping
    mapping_file = os.path.join(output_dir, 'node_mapping.pkl')
    print(f"[INFO] Saving node mapping to {mapping_file}")
    mapping = {
        'node_to_idx': node_to_idx,
        'idx_to_node': idx_to_node
    }
    with open(mapping_file, 'wb') as f:
        pickle.dump(mapping, f)
    
    # 3. Lưu nodes metadata
    metadata_file = os.path.join(output_dir, 'nodes_metadata.csv')
    print(f"[INFO] Saving nodes metadata to {metadata_file}")
    
    # Thêm index vào GeoDataFrame
    nodes_df = nodes_gdf.reset_index()
    nodes_df['matrix_idx'] = nodes_df['osmid'].map(node_to_idx)
    nodes_df = nodes_df[['osmid', 'matrix_idx', 'y', 'x', 'street_count']].rename(
        columns={'y': 'lat', 'x': 'lng', 'osmid': 'node_id'}
    )
    nodes_df.to_csv(metadata_file, index=False)
    
    # 4. Lưu statistics
    stats_file = os.path.join(output_dir, 'matrix_stats.json')
    print(f"[INFO] Saving statistics to {stats_file}")
    
    stats = {
        'created_at': datetime.now().isoformat(),
        'matrix_shape': list(dist_matrix.shape),
        'matrix_dtype': str(dist_matrix.dtype),
        'matrix_size_bytes': int(dist_matrix.nbytes),
        'validation': validation
    }
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n[SUCCESS] All outputs saved to {output_dir}/")
    print(f"  - dist_matrix.npy: {os.path.getsize(matrix_file) / (1024**2):.1f} MB")
    print(f"  - node_mapping.pkl: {os.path.getsize(mapping_file) / 1024:.1f} KB")
    print(f"  - nodes_metadata.csv: {os.path.getsize(metadata_file) / 1024:.1f} KB")
    print(f"  - matrix_stats.json: {os.path.getsize(stats_file) / 1024:.1f} KB")


def main():
    parser = argparse.ArgumentParser(description='Compute Hanoi road network distance matrix')
    parser.add_argument('--place', type=str, default='Dong Da District, Hanoi, Vietnam',
                        help='Place name for OSMnx query')
    parser.add_argument('--network-type', type=str, default='drive',
                        choices=['drive', 'walk', 'bike', 'all'],
                        help='Network type')
    parser.add_argument('--processes', type=int, default=None,
                        help='Number of processes (default: CPU count)')
    parser.add_argument('--output', type=str, default='data',
                        help='Output directory')
    parser.add_argument('--cache', type=str, default='cache',
                        help='Cache directory for graph')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("HANOI ROAD NETWORK DISTANCE MATRIX CALCULATOR")
    print("=" * 60)
    print(f"Place: {args.place}")
    print(f"Network type: {args.network_type}")
    print(f"Processes: {args.processes or cpu_count()}")
    print(f"Output: {args.output}")
    print("=" * 60)
    
    # Step 1: Load or download graph
    G = load_or_download_graph(args.place, args.network_type, args.cache)
    
    # Step 2: Preprocess nodes
    nodes_list, node_to_idx, idx_to_node, nodes_gdf = preprocess_nodes(G)
    
    # Step 3: Compute distance matrix
    dist_matrix = compute_distance_matrix(G, nodes_list, node_to_idx, args.processes)
    
    # Step 4: Validate matrix
    validation = validate_matrix(dist_matrix)
    
    # Step 5: Save outputs
    save_outputs(dist_matrix, nodes_gdf, node_to_idx, idx_to_node, validation, args.output)
    
    print("\n[DONE] Distance matrix computation completed!")


if __name__ == '__main__':
    main()
