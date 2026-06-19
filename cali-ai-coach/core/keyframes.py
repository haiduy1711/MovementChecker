import numpy as np
from sklearn.cluster import KMeans


def extract_key_frames(pose_array: np.ndarray, n_clusters: int = 30) -> np.ndarray:
    """
    Key frame extraction via K-Means clustering on pelvis-normalized 2D coordinates.

    Returns sorted array of frame indices closest to each cluster centroid.
    """
    F = pose_array.shape[0]
    if F == 0:
        return np.array([], dtype=np.int64)

    mean_conf = pose_array[:, :, 2].mean(axis=1)
    valid_mask = mean_conf > 0
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        step = max(1, F // n_clusters)
        return np.arange(0, F, step)[:n_clusters].astype(np.int64)

    xy = pose_array[valid_indices, :, :2].astype(np.float32)
    pelvis = xy[:, 0:1, :]
    xy_norm = xy - pelvis
    vectors = xy_norm.reshape(len(valid_indices), -1)
    vectors = np.nan_to_num(vectors, nan=0.0, posinf=0.0, neginf=0.0)

    k = min(n_clusters, len(valid_indices))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(vectors)

    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_

    rep_valid_indices = []
    for cluster_id in range(k):
        mask = labels == cluster_id
        if not mask.any():
            continue
        cluster_vectors = vectors[mask]
        cluster_orig_idx = np.where(mask)[0]
        dists = np.linalg.norm(cluster_vectors - centroids[cluster_id], axis=1)
        best = cluster_orig_idx[np.argmin(dists)]
        rep_valid_indices.append(int(valid_indices[best]))

    return np.array(sorted(set(rep_valid_indices)), dtype=np.int64)
