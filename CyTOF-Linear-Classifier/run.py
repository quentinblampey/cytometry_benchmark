import argparse
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import scanpy as sc
from tqdm import tqdm


def main():
    adata = sc.read_h5ad(f"../POISED/pp.h5ad")
    label = "labels"

    path = Path(f"../POISED/LDA")

    for i, batch in enumerate(adata.obs.batch.cat.categories):
        adata_train = adata[
            (adata.obs.batch == batch) & (adata.obs.labels != "Unknown")
        ].copy()
        adata_test = adata[adata.obs.batch != batch].copy()

        for name, adata_ in [("train", adata_train), ("test", adata_test)]:
            path_batch = path / f"{batch}_{name}"

            path_batch.mkdir(parents=True, exist_ok=True)

            if "X_corrected" in adata_.obsm:
                df = pd.DataFrame(adata_.obsm["X_corrected"], index=adata_.obs.index)
            else:
                df = pd.DataFrame(adata_.X, index=adata_.obs.index)
            df[label] = adata_.obs[label].values

            df.to_csv(path_batch / "data.csv", header=None)

    subprocess.call(f"Rscript run_poised.R", shell=True)


if __name__ == "__main__":
    main()
