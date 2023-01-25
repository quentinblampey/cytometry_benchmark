import numpy as np
import scanpy as sc
import scyan

if __name__ == "__main__":
    adata = sc.read_h5ad("../POISED/no_pp.h5ad")
    adata.layers["raw"] = adata.X.copy()

    scale = 0.01

    n_batches = len(adata.obs.batch.cat.categories)
    for i in range(5):
        noise = np.random.exponential(
            scale=1, size=(n_batches, adata.n_vars, adata.n_vars)
        )

        adata.X = adata.layers["raw"].copy()

        for j, batch in enumerate(adata.obs.batch.cat.categories):
            spillover = np.eye(adata.n_vars) + scale * noise[j]
            adata.X[adata.obs.batch == batch] = (
                adata.X[adata.obs.batch == batch] @ spillover
            )

        scyan.tools.asinh_transform(adata)
        adata.write_h5ad(f"../POISED/batch_effect/scale_{scale}_I{i}.h5ad")
