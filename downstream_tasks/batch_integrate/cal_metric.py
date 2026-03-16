from utils import Evaluator
import argparse
import scanpy as sc
import pandas as pd
import numpy as np


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="空间转录组细胞嵌入质量评估")
    parser.add_argument("--inx", type=int, required=False,  default=0,help="0到6")
    arg = parser.parse_args()

    simple_path = '/home/cavin/jt/benchmark/data/hbc/s1_filtered_cells.h5ad'
    celltype_col = 'cell_type'
    inx = arg.inx
    cell_emb_col = ['X_PCA','X_Harmony','X_scGPT','X_scFoundation','X_Geneformer','X_Nicheformer','X_OminiCell']
    cell_emb_col = [cell_emb_col[inx]]
    emb_path_list = [
        "/home/cavin/jt/benchmark/experiments/embedings/batch_integrate/pca_human_breast_cancer_emb.parquet",
        '/home/cavin/jt/benchmark/experiments/embedings/batch_integrate/Harmony_human_breast_cancer_emb.parquet',
        '/home/cavin/jt/benchmark/experiments/embedings/batch_integrate/scgpt_human_breast_cancer_emb.parquet',
        '/home/cavin/jt/benchmark/experiments/embedings/batch_integrate/scfoundation_human_breast_cancer_emb.parquet',
        '/home/cavin/jt/benchmark/experiments/embedings/batch_integrate/geneformer_human_breast_cancer_emb.parquet',
        '/home/cavin/jt/benchmark/experiments/embedings/batch_integrate/ominicell_human_breast_cancer_emb.parquet',
        '/home/cavin/jt/benchmark/experiments/embedings/batch_integrate/ominicell_human_breast_cancer_emb.parquet'
    ]
    emb_path_list = [emb_path_list[inx]]
    batch_key = "batch"


    loaded_emb_df_list = [pd.read_parquet(path) for path in emb_path_list]
    adata = sc.read_h5ad(simple_path)
    aligned_emb_df_list = [df.reindex(adata.obs_names) for df in loaded_emb_df_list]
    for i,emb_key in enumerate(cell_emb_col):
        adata.obsm[emb_key] = aligned_emb_df_list[i].to_numpy(dtype=np.float32)
    print(adata)
    evaluator = Evaluator(adata,cell_emb_col,batch_key,celltype_col)
    result = evaluator.run_all()
    result.to_csv(f'/home/cavin/jt/benchmark/experiments/results/integrate_metrics/{inx}_human_breast_cancer_integrate.csv')