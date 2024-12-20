import numpy as np
from scipy.ndimage import zoom
from skimage.segmentation import slic
from skimage.measure import regionprops
from scipy.spatial import cKDTree
import torch
from torch_geometric.data import Data

def resize_volume(img, target_shape=(64, 64, 64)):
    """Resize a 3D volume to target shape."""
    factors = [t / c for t, c in zip(target_shape, img.shape)]
    return zoom(img, factors, order=1)

def normalize_volume(volume):
    """Normalize the volume"""
    min_val = np.nanmin(volume)
    max_val = np.nanmax(volume)
    volume = (volume - min_val) / (max_val - min_val)
    volume = np.nan_to_num(volume, nan=0.0)  # Replace NaNs with 0
    return volume.astype(np.float32)

def slice_to_graph(slice, num_superpixels=1000):
    """Convert a 2D slice to a graph representation."""
    slice = (slice - slice.min()) / (slice.max() - slice.min())
    segments = slic(slice, n_segments=num_superpixels, compactness=10, sigma=1, start_label=0, channel_axis=None)
    regions = regionprops(segments + 1, intensity_image=slice)

    node_features = []
    centroids = []
    for region in regions:
        node_features.append([
            region.mean_intensity,
            region.area
        ])
        centroids.append(region.centroid)

    node_features = torch.tensor(node_features, dtype=torch.float32)
    centroids = np.array(centroids)

    tree = cKDTree(centroids)
    _, indices = tree.query(centroids, k=6)

    edge_index = []
    edge_attr = []
    for i, neighbors in enumerate(indices):
        for j in neighbors[1:]:
            edge_index.append([i, j])
            edge_index.append([j, i])
            distance = np.linalg.norm(centroids[i] - centroids[j])
            edge_attr.append([1 / (1 + distance)])
            edge_attr.append([1 / (1 + distance)])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
