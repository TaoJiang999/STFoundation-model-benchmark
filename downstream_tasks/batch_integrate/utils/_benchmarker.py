"""
scUnify integrated evaluation framework
scIB-metrics + scGraph metrics integration
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ._scib import ScibWrapper
from ._scgraph import ScGraphWrapper

if TYPE_CHECKING:
    from anndata import AnnData


class Evaluator:
    """Integrated scIB + scGraph evaluation.
    
    Evaluates FM embedding quality using scIB-metrics and scGraph metrics
    in a comprehensive manner.
    
    Parameters
    ----------
    adata
        AnnData object (embeddings must be stored in obsm)
    embedding_keys
        List of embedding obsm keys to evaluate
    batch_key
        obs column name containing batch information
    label_key
        obs column name containing cell-type labels
    n_jobs
        Number of parallel jobs for scIB neighbor computation (default: -1, all cores)
    
    Examples
    --------
    >>> import scunify as scu
    >>> 
    >>> evaluator = scu.eval.Evaluator(
    ...     adata,
    ...     embedding_keys=["X_scgpt", "X_uce", "X_scfoundation"],
    ...     batch_key="batch",
    ...     label_key="cell_type",
    ... )
    >>> 
    >>> # Run scIB metrics only
    >>> scib_results = evaluator.run_scib()
    >>> 
    >>> # Run scGraph metrics only
    >>> scgraph_results = evaluator.run_scgraph()
    >>> 
    >>> # Run all metrics
    >>> all_results = evaluator.run_all()
    >>> 
    >>> # Plot results
    >>> evaluator.plot_results(save_dir="./results")
    """
    
    def __init__(
        self,
        adata: "AnnData",
        embedding_keys: list[str],
        batch_key: str,
        label_key: str,
        n_jobs: int = -1,
    ):
        self.adata = adata
        self.embedding_keys = embedding_keys
        self.batch_key = batch_key
        self.label_key = label_key
        self.n_jobs = n_jobs
        
        # Lazy initialization
        self._scib: ScibWrapper | None = None
        self._scgraph: ScGraphWrapper | None = None
        
        self._scib_results: pd.DataFrame | None = None
        self._scgraph_results: pd.DataFrame | None = None
        self._combined_results: pd.DataFrame | None = None
    
    @property
    def scib(self) -> ScibWrapper:
        """ScibWrapper instance (lazily created)."""
        if self._scib is None:
            self._scib = ScibWrapper(
                self.adata,
                embedding_keys=self.embedding_keys,
                batch_key=self.batch_key,
                label_key=self.label_key,
                n_jobs=self.n_jobs,
            )
        return self._scib
    
    @property
    def scgraph(self) -> ScGraphWrapper:
        """ScGraphWrapper instance (lazily created)."""
        if self._scgraph is None:
            self._scgraph = ScGraphWrapper(
                self.adata,
                embedding_keys=self.embedding_keys,
                batch_key=self.batch_key,
                label_key=self.label_key,
            )
        return self._scgraph
    
    def run_scib(self, min_max_scale: bool = False) -> pd.DataFrame:
        """Run scIB-metrics evaluation.
        
        Parameters
        ----------
        min_max_scale
            Whether to scale results to 0-1
        
        Returns
        -------
        scIB results DataFrame
        """
        self._scib_results = self.scib.run(min_max_scale=min_max_scale)
        return self._scib_results
    
    def run_scgraph(self) -> pd.DataFrame:
        """Run scGraph evaluation.
        
        Returns
        -------
        scGraph results DataFrame
        """
        self._scgraph_results = self.scgraph.run()
        return self._scgraph_results
    
    def run_all(self, min_max_scale: bool = False) -> pd.DataFrame:
        """Compute both scIB and scGraph metrics.
        
        Parameters
        ----------
        min_max_scale
            Whether to scale results to 0-1
        
        Returns
        -------
        Combined results DataFrame
        """
        print("=" * 50)
        print("Running scib-metrics evaluation...")
        print("=" * 50)
        scib_df = self.run_scib(min_max_scale=min_max_scale)
        
        print("\n" + "=" * 50)
        print("Running scGraph evaluation...")
        print("=" * 50)
        scgraph_df = self.run_scgraph()
        
        self._combined_results = self._merge_results(scib_df, scgraph_df)
        return self._combined_results
    
    def _merge_results(
        self,
        scib_df: pd.DataFrame,
        scgraph_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge scIB and scGraph results.
        
        Final result shape: (embeddings x metrics)
        - rows: embedding names (X_scgpt, X_uce, ...)
        - columns: all metrics (scIB + scGraph)
        
        Parameters
        ----------
        scib_df
            scIB results DataFrame (embeddings x metrics)
        scgraph_df
            scGraph results DataFrame (embeddings x metrics)
        
        Returns
        -------
        Merged DataFrame (embeddings x all_metrics)
        """
        # Remove Metric Type row from scIB results (if present)
        if 'Metric Type' in scib_df.index:
            scib_df = scib_df.drop('Metric Type')
        
        # scIB-metrics get_results() already returns (embeddings x metrics)
        # scGraph also returns (embeddings x metrics)
        
        # Find common embeddings
        common_embeddings = scib_df.index.intersection(scgraph_df.index)
        
        if len(common_embeddings) > 0:
            scib_subset = scib_df.loc[common_embeddings]
            scgraph_subset = scgraph_df.loc[common_embeddings]
            # Concat columns (merge metric columns for the same embeddings)
            combined = pd.concat([scib_subset, scgraph_subset], axis=1)
        else:
            # Debug: print indices
            print(f"[DEBUG] scib_df.index: {list(scib_df.index)}")
            print(f"[DEBUG] scgraph_df.index: {list(scgraph_df.index)}")
            # Just concat (indices differ)
            combined = pd.concat([scib_df, scgraph_df], axis=1)
        
        # Remove duplicate columns
        combined = combined.loc[:, ~combined.columns.duplicated()]
        
        combined.index.name = 'Embedding'
        
        return combined
    
    def get_results(self, min_max_scale: bool = False) -> pd.DataFrame:
        """Return results DataFrame.
        
        Parameters
        ----------
        min_max_scale
            Whether to scale results to 0-1
        
        Returns
        -------
        Combined results DataFrame
        """
        if self._combined_results is None:
            return self.run_all(min_max_scale=min_max_scale)
        return self._combined_results
    
    def plot_scib(
        self,
        min_max_scale: bool = False,
        show: bool = True,
        save_dir: str | None = None,
    ):
        """Plot scIB results only.
        
        Parameters
        ----------
        min_max_scale
            Whether to scale results to 0-1
        show
            Whether to display the plot on screen
        save_dir
            Save directory (None to skip saving)
        
        Returns
        -------
        plottable.Table object
        """
        return self.scib.plot(
            min_max_scale=min_max_scale,
            show=show,
            save_dir=save_dir,
        )
    
    def plot_results(
        self,
        min_max_scale: bool = False,
        show: bool = True,
        save_dir: str | None = None,
    ):
        """Plot combined scIB + scGraph results.
        
        Parameters
        ----------
        min_max_scale
            Whether to scale results to 0-1
        show
            Whether to display the plot on screen
        save_dir
            Save directory (None to skip saving)
        
        Returns
        -------
        plottable.Table object
        """
        from ._plotting import plot_combined_table
        
        if self._combined_results is None:
            self.run_all(min_max_scale=min_max_scale)
        
        return plot_combined_table(
            self._combined_results,
            save_dir=save_dir,
            min_max_scale=min_max_scale,
            show=show,
        )
    
    def save_results(self, path: str) -> None:
        """Save results to CSV.
        
        Parameters
        ----------
        path
            Output path (.csv)
        """
        if self._combined_results is None:
            self.run_all()
        self._combined_results.to_csv(path)
        print(f"Results saved to {path}")
    
    @classmethod
    def from_adata(
        cls,
        adata: "AnnData",
        batch_key: str,
        label_key: str,
        embedding_prefix: str = "X_",
        n_jobs: int = -1,
    ) -> "Evaluator":
        """Create Evaluator by auto-detecting embedding keys from AnnData.
        
        Parameters
        ----------
        adata
            AnnData object
        batch_key
            obs column name containing batch information
        label_key
            obs column name containing cell-type labels
        embedding_prefix
            Embedding obsm key prefix (default: "X_")
        n_jobs
            Number of parallel jobs for scIB neighbor computation
        
        Returns
        -------
        Evaluator instance
        
        Examples
        --------
        >>> # Assume adata.obsm has X_scgpt, X_uce, X_scfoundation
        >>> evaluator = Evaluator.from_adata(
        ...     adata,
        ...     batch_key="batch",
        ...     label_key="cell_type",
        ... )
        >>> results = evaluator.run_all()
        """
        # Select FM-related keys starting with embedding_prefix
        fm_keywords = ["scgpt", "uce", "scfoundation", "geneformer", "cellbert"]
        
        embedding_keys = []
        for key in adata.obsm.keys():
            if key.startswith(embedding_prefix):
                model_name = key[len(embedding_prefix):].lower()
                if any(kw in model_name for kw in fm_keywords):
                    embedding_keys.append(key)
        
        if not embedding_keys:
            # If no FM keywords found, use all keys with the prefix
            embedding_keys = [k for k in adata.obsm.keys() 
                            if k.startswith(embedding_prefix) 
                            and k not in ["X_pca", "X_umap", "X_tsne"]]
        
        print(f"Auto-detected embedding keys: {embedding_keys}")
        
        return cls(
            adata=adata,
            embedding_keys=embedding_keys,
            batch_key=batch_key,
            label_key=label_key,
            n_jobs=n_jobs,
        )
