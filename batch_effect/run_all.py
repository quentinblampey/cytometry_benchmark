import argparse
from pathlib import Path

import pandas as pd
import scanpy as sc
import scanpy.external as sce
from sklearn.decomposition import PCA
from tqdm import tqdm


def _pca(adata):
    pca = PCA(n_components=adata.n_vars - 1)
    adata.obsm["X_pca"] = pca.fit_transform(adata.X)
    return pca


def saucie(adata, _):
    import SAUCIE

    pca = _pca(adata)

    batch_to_categorical = {b: i for i, b in enumerate(adata.obs.batch.cat.categories)}
    labels = adata.obs.batch.apply(batch_to_categorical.get)

    data = adata.obsm["X_pca"]

    saucie = SAUCIE.SAUCIE(data.shape[1], lambda_b=1)
    loadtrain = SAUCIE.Loader(data, shuffle=True, labels=labels)

    saucie.train(loadtrain, steps=1000)
    loadeval = SAUCIE.Loader(data, shuffle=False, labels=labels)

    reconstruction, _ = saucie.get_reconstruction(loadeval)
    adata.obsm["X_corrected"] = pca.inverse_transform(reconstruction)

    batch_correction_embedding, _ = saucie.get_embedding(loadeval)
    adata.obsm["saucie_embedding"] = batch_correction_embedding


def combat(adata, _):
    adata.obsm["X_corrected"] = sc.pp.combat(adata, key="batch", inplace=False)


def harmony(adata, _):
    pca = _pca(adata)
    sce.pp.harmony_integrate(adata, "batch")
    adata.obsm["X_corrected"] = pca.inverse_transform(adata.obsm["X_pca_harmony"])


def cydar(adata, filename):
    import subprocess

    r_arg = f"cydar_{filename}"
    data_folder = Path(r_arg)
    data_folder_out = Path(f"{r_arg}_out")

    for batch in adata.obs.batch.cat.categories:
        adata_ = adata[adata.obs.batch == batch]

        path_folder = data_folder / batch
        path_folder.mkdir(parents=True, exist_ok=True)
        (data_folder_out / batch).mkdir(parents=True, exist_ok=True)

        for file in adata_.obs.file.cat.categories:
            adata_[adata_.obs.file == file].to_df().to_csv(path_folder / f"{file}.csv")

    subprocess.call(f"Rscript run_cydar.R {r_arg}", shell=True)

    for batch in adata.obs.batch.cat.categories:
        path_folder = data_folder_out / batch
        for file in adata[adata.obs.batch == batch].obs.file.cat.categories:
            df = pd.read_csv(path_folder / f"{file}.csv", index_col=0)
            adata.X[(adata.obs.batch == batch) & (adata.obs.file == file)] = df.values

    adata.obsm["X_corrected"] = adata.X


dict_correction = {
    "saucie": saucie,
    "combat": combat,
    "harmony": harmony,
    "cydar": cydar,
}


def main(args):
    path_data = Path("../POISED/batch_effect")
    correction = dict_correction[args.method]

    filenames = ["pp"] + [
        f"scale_{0.002 * factor}_I{i}" for factor in range(1, 6) for i in range(5)
    ]

    for filename in tqdm(filenames):
        output = path_data / f"{filename}_{args.method}.h5ad"
        if output.exists():
            continue

        adata = sc.read_h5ad(path_data / f"{filename}.h5ad")
        correction(adata, filename)
        adata.write_h5ad(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        required=True,
        help="Choose among: saucie, combat, cydar, and harmony.",
    )

    main(parser.parse_args())
