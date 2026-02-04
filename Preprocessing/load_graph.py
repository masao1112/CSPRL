import os
import osmnx as ox
from shapely.geometry import Point, MultiPolygon, LineString, Polygon
from shapely.ops import split
import numpy as np
import pickle

"""
create graph and grid for Dong Da district, Hanoi
"""
n_x, n_y = 32, 32
# coordinates for Dong Da, Hanoi
rec = [
    (105.80, 21.00), # Bottom-Left
    (105.80, 21.03), # Top-Left
    (105.85, 21.03), # Top-Right
    (105.85, 21.00)  # Bottom-Right
]


def grid_preparation(my_rec):
    """
    put a grid over a rectangle
    :param my_rec: list of four GPS coordinates
    """
    result = Polygon(my_rec)
    # compute splitter
    minx, miny, maxx, maxy = result.bounds
    dx = (maxx - minx) / n_x  # width of a small part
    dy = (maxy - miny) / n_y  # height of a small part
    horizontal_splitters = [LineString([(minx, miny + k * dy), (maxx, miny + k * dy)]) for k in range(n_y)]
    vertical_splitters = [LineString([(minx + k * dx, miny), (minx + k * dx, maxy)]) for k in range(n_x)]
    splitters = horizontal_splitters + vertical_splitters
    # split
    for splitter in splitters:
        result = MultiPolygon(split(result, splitter))
    my_grids = list(result.geoms)
    return my_grids


def grid_ret_index(point, grids):
    for j in range(len(grids)):
        if point.within(grids[j]):
            r, c = (n_x - int(j / n_x)) - 1, int(j % n_x)
            return r, c
    return None, None


def grid_location(my_node, grids):
    # finding in which cell of the grid the node is
    lon, lat = my_node[1]['x'], my_node[1]['y']
    point = Point(lon, lat)
    row, column = grid_ret_index(point, grids)
    my_node[1]["row"], my_node[1]["column"] = row, column


if __name__ == '__main__':
    location = "DongDa"
    if not os.path.exists(f"../Graph/{location}"):
        os.makedirs(f"../Graph/{location}")

    print(f"Downloading {location} Urban Map...")
    # Focus on Dong Da district center
    G = ox.graph_from_point((21.011, 105.827), dist=2500, network_type="drive")

    # calculate for each node in which cell it is and for each grid cell how many nodes it has inside
    node_list = list(G.nodes(data=True))
    print(f"Nodes found: {len(node_list)}")
    
    grids = grid_preparation(rec)
    grid_density = np.zeros((n_x, n_y))
    
    outside_count = 0
    for node in node_list:
        grid_location(node, grids)
        row, column = node[1]["row"], node[1]["column"]
        if row is not None and column is not None:
            grid_density[row][column] += 1
        else:
            outside_count += 1
    
    print(f"Grid density calculated. {outside_count} nodes were outside the grid.")

    # Save the graph files with UTF-8 encoding
    ox.save_graphml(G, filepath=f"../Graph/{location}/{location}.graphml")
    with open(f"../Graph/{location}/node_list_{location}.txt", 'w', encoding='utf-8') as file:
        file.write(str(node_list))
    pickle.dump(grid_density, open(f"../Graph/{location}/grid_density_{location}.pkl", "wb"))
    print(f"Success! Files saved in ../Graph/{location}/")
