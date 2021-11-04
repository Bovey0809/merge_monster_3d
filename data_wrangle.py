# %%
import open3d as o3d
import multiprocessing
from os import path
import numpy as np
import struct
import glob
import pandas as pd
from multiprocessing import pool
import multiprocessing


# %%
def read_points(points_path):
    points_list = []
    with open(points_path, 'rb') as f:
        byte = f.read(16)
        while byte:
            x, y, z, intensity = struct.unpack('ffff', byte)
            points_list.append([x, y, z])
            byte = f.read(16)
    return points_list


def pcd2img(pcd: o3d.geometry.PointCloud, filename=None):
    """save or show pcd to image with filename.

    Args:
        pcd (o3d.geometry.PointCloud): point cloud type from open3d.
        filename: if None then show img, else save to local. 
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    filename = filename if filename else "./output_pcd.png"
    vis.capture_screen_image(filename)
    vis.destroy_window()


# %%
points_paths = glob.glob('data/sunrgbd/points/*.bin')

# %%
# Try to figure out the format of the bin file first.
# save one point cloud and show it.
# save one point cloud
tmp_path = points_paths[0]
pcd_np = np.array(read_points(tmp_path))
# %%
pcd_o3d = o3d.geometry.PointCloud()
pcd_o3d.points = o3d.utility.Vector3dVector(pcd_np)
pcd2img(pcd_o3d)

# %%
