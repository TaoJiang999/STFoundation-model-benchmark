import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
from sklearn.preprocessing import StandardScaler

def morans_i(coords, features, k_neighbors=8):
        """
        Calculate Moran's I spatial autocorrelation statistic for vector-valued features.
        
        Moran's I measures spatial autocorrelation of the aligned features using vector operations.
        Values range from -1 to +1:
        - +1: Perfect positive spatial autocorrelation (clustering)
        - 0: No spatial autocorrelation (random distribution)
        - -1: Perfect negative spatial autocorrelation (dispersion)
        
        Parameters
        ----------
        k_neighbors : int, optional
            Number of nearest neighbors to consider, by default 6
            
        Returns
        -------
        float
            Moran's I statistic
        """
        # Use aligned features for Moran's I calculation
        # features = features.values  # Shape: (n_samples, n_features)

        # Calculate spatial weights matrix using k-nearest neighbors
        n = len(features)
        nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(coords)
        distances, indices = nbrs.kneighbors()

        # Remove self-connections (first neighbor is the point itself)
        indices = indices[:, 1:]
        distances = distances[:, 1:]

        with np.errstate(divide='ignore'):
            weights = 1.0 / (distances ** 2)
        weights[distances == 0] = 0.0 

        row_indices = np.repeat(np.arange(n), k_neighbors)
        col_indices = indices.flatten()
        data = weights.flatten()
        W_sparse = csr_matrix((data, (row_indices, col_indices)), shape=(n, n))

        W_sum = W_sparse.sum()
        if W_sum == 0:
            return 0.0
        
        x_mean = np.mean(features, axis=0)
        x_centered = features - x_mean

        denominator = np.sum(x_centered ** 2)
        if denominator == 0:
            return 0.0
        
        numerator = np.sum(x_centered * W_sparse.dot(x_centered))

        moran_i = (n / (2 * W_sum)) * (numerator / denominator)

        # # Create spatial weights matrix using inverse square distance
        # W = np.zeros((n, n))
        # for i in range(n):
        #     for j_idx, dist in zip(indices[i], distances[i]):
        #         if dist > 0:  # Avoid division by zero
        #             W[i, j_idx] = 1.0 / (dist ** 2)

        # # Calculate mean vector across all samples
        # x_mean = np.mean(features, axis=0)  # Shape: (n_features,)

        # # Center the features
        # x_centered = features - x_mean  # Shape: (n_samples, n_features)

        # # Calculate numerator: sum(W_ij * dot(x_centered[i], x_centered[j]))
        # numerator = 0
        # for i in range(n):
        #     for j in range(n):
        #         # Dot product between centered feature vectors
        #         dot_product = np.dot(x_centered[i], x_centered[j])
        #         numerator += W[i, j] * dot_product

        # # Calculate denominator: sum(dot(x_centered[i], x_centered[i]))
        # denominator = 0
        # for i in range(n):
        #     # Dot product of vector with itself (squared norm)
        #     denominator += np.dot(x_centered[i], x_centered[i])

        # # Calculate Moran's I
        # if denominator != 0:
        #     moran_i = (n / (2 * np.sum(W))) * (numerator / denominator)
        # else:
        #     moran_i = 0.0

        return moran_i


def asw_score(coords: np.ndarray, labels: np.ndarray,
                   sample_size: int = 50_000, random_state: int = 42) -> float:
    unique_labels = len(set(labels))
    if unique_labels < 2:
        return 0.0
    N = len(coords)
    if N > sample_size:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(N, size=sample_size, replace=False)
        coords = coords[idx]
        labels = labels[idx]

    return silhouette_score(X=coords, labels=labels, metric="euclidean")

def chaos_score(coords: np.ndarray, labels: np.ndarray) -> float:
    """
    优化版 CHAOS score
    - 全局建一次 KD-Tree，替代 K 次独立建树
    - 向量化过滤跨簇近邻，无 Python 循环
    """
    X = StandardScaler().fit_transform(coords)
    N = len(X)

    # ── 1. 全局建树，查簇内最近邻 ───────────────────────
    # 查 k+1 是因为第0列是自身（距离=0）
    # 最坏情况需要查更多邻居才能找到同簇点，这里用保守值
    k_query = min(10, N - 1)  # 一般簇不会极度稀疏，10足够
    nbrs = NearestNeighbors(n_neighbors=k_query + 1, algorithm="kd_tree", n_jobs=-1)
    nbrs.fit(X)
    distances, indices = nbrs.kneighbors(X)    # (N, k+1)

    # 去掉第0列（自身）
    distances = distances[:, 1:]               # (N, k)
    indices   = indices[:, 1:]                 # (N, k)

    # ── 2. 找每个点的簇内最近邻 ─────────────────────────
    neighbor_labels = labels[indices]          # (N, k) 每个邻居的标签
    same_cluster    = neighbor_labels == labels[:, None]  # (N, k) bool mask

    # 取每行第一个同簇邻居的距离
    # argmax 在 bool 数组上返回第一个 True 的位置
    first_same = np.argmax(same_cluster, axis=1)          # (N,)
    has_same   = same_cluster.any(axis=1)                 # (N,) 是否找到同簇邻居

    # 簇大小 > 2 的点才参与计算（与原版逻辑一致）
    cluster_counts = np.bincount(labels, minlength=labels.max() + 1)
    valid_mask = (cluster_counts[labels] > 2) & has_same  # (N,)

    if valid_mask.sum() == 0:
        return 0.0

    intra_dists = distances[np.arange(N), first_same]     # (N,)
    return float(intra_dists[valid_mask].sum() / valid_mask.sum())

def pas_score(coords, labels, k=8):
    """
    Calculate the PAS score for spatial coordinates and labels.
    
    Parameters
    ----------
    k : int, optional
        Number of nearest neighbors to consider, by default 8
        
    Returns
    -------
    float
        PAS score
    """
    nbrs = NearestNeighbors(n_neighbors=k).fit(coords)
    indices = nbrs.kneighbors(return_distance=False)
    return ((labels.reshape(-1, 1) != labels[indices]).sum(1) > k / 2).mean()


def db_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    优化版 Davies-Bouldin Score
    - 向量化替代 K 次 pairwise_distances 循环
    - intra_dist 用均方根距离（与原版等价的向量化写法）
    """
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    n_samples, n_features = X.shape
    n_labels = len(le.classes_)

    # ── 1. 质心 & 簇内平均距离（向量化）────────────────
    counts = np.bincount(labels, minlength=n_labels).astype(np.float64)  # (K,)

    centroids = np.zeros((n_labels, n_features), dtype=np.float64)
    np.add.at(centroids, labels, X)
    centroids /= counts[:, None]                     # (K, D)

    # 每个点到其质心的距离，无需循环
    point_centroids = centroids[labels]              # (N, D)
    diff = X - point_centroids                       # (N, D)
    point_dists = np.linalg.norm(diff, axis=1)       # (N,)

    # 每簇平均距离：bincount 加权求和
    intra_dists = np.bincount(labels, weights=point_dists,
                              minlength=n_labels) / counts   # (K,)

    # ── 2. 质心间距离矩阵（K×K，K通常很小）─────────────
    # K 最多几百，(K,K) 矩阵完全可接受
    diff_c = centroids[:, None, :] - centroids[None, :, :]   # (K, K, D)
    centroid_distances = np.linalg.norm(diff_c, axis=-1)      # (K, K)

    # ── 3. DB 分数 ───────────────────────────────────────
    if np.allclose(intra_dists, 0) or np.allclose(centroid_distances, 0):
        return 0.0

    centroid_distances[centroid_distances == 0] = np.inf
    combined = intra_dists[:, None] + intra_dists[None, :]    # (K, K)
    scores = np.max(combined / centroid_distances, axis=1)    # (K,)
    return float(np.mean(scores))



def chs_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    优化版 Calinski-Harabasz Score
    - 用 np.bincount + 矩阵运算替代 K 次循环
    - 避免中间大临时数组
    - 内存 O(K×D)，而非 O(N×D) × K
    """
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    n_samples, n_features = X.shape
    n_labels = len(le.classes_)

    # ── 1. 每个簇的样本数和均值 ──────────────────────
    counts = np.bincount(labels, minlength=n_labels).astype(np.float64)  # (K,)

    # 用 np.add.at 的向量化替代：一次性算出每簇的和
    cluster_sums = np.zeros((n_labels, n_features), dtype=np.float64)
    np.add.at(cluster_sums, labels, X)
    cluster_means = cluster_sums / counts[:, None]   # (K, D)

    global_mean = np.mean(X, axis=0)                 # (D,)

    # ── 2. 簇间离散度 extra_disp ─────────────────────
    # sum_k( n_k * ||mean_k - mean||^2 )
    diff = cluster_means - global_mean               # (K, D)
    extra_disp = float(np.einsum('k,kd->', counts, diff ** 2))

    # ── 3. 簇内离散度 intra_disp ─────────────────────
    # sum_k sum_{i in k} ||x_i - mean_k||^2
    # = sum_i ||x_i - mean_{label[i]}||^2  → 向量化，无循环
    point_means = cluster_means[labels]              # (N, D) 每个点对应簇均值
    intra_disp = float(np.sum((X - point_means) ** 2))

    if intra_disp == 0.0:
        return 1.0

    return extra_disp * (n_samples - n_labels) / (intra_disp * (n_labels - 1.0))




if __name__ == "__main__":
    import glob
    import argparse
    import scanpy as sc
    from tqdm import tqdm
    shard_inx = [0,1,2]
    simple_path = [f'/home/cavin/jt/benchmark/data/crc/VisiumHD_P2_shard_{shard_inx}.h5ad' for shard_inx in shard_inx]
    adata = [sc.read_h5ad(path) for path in simple_path]
    var = adata[0].var
    adata = sc.concat(adata, axis=0)
    adata.var = var
    del var
    parser = argparse.ArgumentParser(description="空间转录组聚类质量评估")
    parser.add_argument("--model-name",    type=str, required=True,  default="scgpt",help="模型名称")
    arg = parser.parse_args()
    res_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    # res_list = [0.1]
    repeat_list = [0,1,2]
    # repeat_list = [0]
    cell_emb_col = "emb"
    emb_path_list = [f"/home/cavin/jt/benchmark/experiments/embedings/spatial_cluster_no_annotations/{arg.model_name}_human_CRC_shard_{shard}_emb.parquet" for shard in shard_inx]
    if arg.model_name != "pca":
        loaded_emb_df = [pd.read_parquet(s) for s in emb_path_list]
        loaded_emb_df = pd.concat(loaded_emb_df)
        aligned_emb_df = loaded_emb_df.reindex(adata.obs_names)
        del loaded_emb_df
        adata.obsm[cell_emb_col] = aligned_emb_df.to_numpy(dtype=np.float32)
    else:
        save_path = f"/home/cavin/jt/benchmark/experiments/embedings/spatial_cluster_no_annotations/PCA_human_CRC_emb.parquet"
        loaded_emb_df = pd.read_parquet(save_path)
        aligned_emb_df = loaded_emb_df.reindex(adata.obs_names)
        del loaded_emb_df
        adata.obsm[cell_emb_col] = aligned_emb_df.to_numpy(dtype=np.float32)
    
    csv_dict = {}
    
    for repeat in repeat_list:
         for res in res_list:
            csv_dict[f"/home/cavin/jt/benchmark/experiments/results/labels_df/CRC/{arg.model_name}_human_CRC_labels_repeat_{repeat}_resolution_{res}.csv"] = (repeat,res) 
    

    result_save_path = f"/home/cavin/jt/benchmark/experiments/results/cluster_metrics/CRC/human_CRC_{arg.model_name}_metric.csv"
    metrics = {"method": f"{arg.model_name}", "repeat times":0,"resolution":0,"CHS": 0, "DB": 0, "PAS": 0, "Morans_I": 0, "ASW": 0}
    mi = True
    for csv_key in tqdm(csv_dict.keys()):
        repeat,res = csv_dict.get(csv_key)
        print(f"current repeat {repeat}  resolution {res}")
        metrics["repeat times"] = repeat
        metrics["resolution"] = res
        label_df = pd.read_csv(csv_key,header=0,index_col=0,sep=",")
        label_df = label_df.reindex(adata.obs_names)
        adata.obs["cluster"] = label_df.values


        coords = adata.obsm["spatial"]
        emb = adata.obsm[cell_emb_col]
        label = adata.obs["cluster"].values
        if mi:
            metrics["Morans_I"] = morans_i(coords=coords,features=emb)
            print(f"Morans_I: {metrics.get('Morans_I')}")
            mi = False
        metrics["CHS"] = chs_score(emb,label)
        print(f"CHS: {metrics.get('CHS')}")
        metrics["DB"] = db_score(emb,label)
        print(f"DB: {metrics.get('DB')}")
        # metrics["ChaoS"] = chaos_score(emb,label)
        # print(f"ChaoS: {metrics.get('ChaoS')}")
        metrics["PAS"] = pas_score(emb,label)
        print(f"PAS: {metrics.get('PAS')}")
        metrics["ASW"] = asw_score(emb,label)
        print(f"ASW: {metrics.get('ASW')}")
        metrics_df = pd.DataFrame.from_dict(metrics, orient='index').T
        metrics_df.to_csv(result_save_path, index=False,mode="a", header=not pd.io.common.file_exists(result_save_path))

