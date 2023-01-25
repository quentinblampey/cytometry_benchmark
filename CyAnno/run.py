import shutil
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
from tqdm import tqdm

from CyAnno import main as run_cyanno


def make_csv(filename, k):
    if k == "all":
        batch = "all"
    else:
        batch = f"B{k}"
    df = pd.read_csv("poised_metadata.csv", index_col=0)
    df = df[df.batch != batch].copy()
    df["y_pred"] = ""

    p = Path(".")
    l_possible = [
        x for x in p.iterdir() if x.name.startswith(f"poised_out_{filename}_{batch}")
    ]

    assert len(
        l_possible
    ), f"Output dir starting by 'poised_out_{filename}_{batch}' not found"
    assert len(l_possible) < 2, f"Multiple output dir for batch {batch}"

    p = l_possible[0]

    for file in set(df.file):
        out = p / (file[:-16] + "labelled_expr.csv")
        df_out = pd.read_csv(out)
        df.loc[df.file == file, "y_pred"] = df_out["Predictedlabels"].values

    df = pd.DataFrame({"id": df.index.values, "y_pred": df.y_pred.values})
    df.to_csv(f"../predictions/cyanno_poised_{filename}_{batch}.csv", index=None)

    shutil.rmtree(p)


def prepare_input(filename):
    name_input_folder = f"poised_{filename}"
    shutil.copytree("poised", name_input_folder)

    for path in Path(name_input_folder).rglob("*.csv"):
        text = path.read_text()
        text = text.replace("ProcessedData", f"cyanno_{filename}")
        path.write_text(text)


def prepare_data(filename):
    path = Path(f"../POISED/cyanno_{filename}")

    if path.exists():
        return

    path_handgated = path / "HandGatedLabelledCells"
    path_live = path / "LiveLabelledCells"

    path_handgated.mkdir(parents=True)
    path_live.mkdir(parents=True)

    adata = anndata.read_h5ad(f"../POISED/batch_effect/{filename}.h5ad")

    for file in adata.obs.file.cat.categories:
        adata_ = adata[adata.obs.file == file].copy()
        adata_.X = np.sinh(adata_.X) * 5  # reverse asinh transform
        df = adata_.to_df()
        df["labels"] = adata_.obs.labels
        df = df.reset_index(drop=True)

        df.to_csv(path_live / f"{file}.csv")

    for patient in adata.obs.patient.cat.categories:
        for suffix, cat in [
            ("PeaStim", "Peanut stimulated"),
            ("UnStim", "Unstimulated"),
        ]:
            for pop in adata.obs.labels.cat.categories:
                adata_ = adata[
                    (adata.obs.patient == patient)
                    & (adata.obs.labels == pop)
                    & (adata.obs.category == cat)
                ].copy()
                adata_.X = np.sinh(adata_.X) * 5  # reverse asinh transform
                df = adata_.to_df()
                df["labels"] = adata_.obs.labels
                df = df.reset_index(drop=True)

                df.to_csv(path_handgated / f"{patient}_{pop}_{suffix}.csv")


def clean(filename):
    # shutil.rmtree(Path(f"../POISED/cyanno_{filename}"))
    shutil.rmtree(Path(f"poised_{filename}"))


def run_one(filename):
    if filename is None:
        for k in [2, 4, 5, 6, 7, 8, 10]:
            run_cyanno(filename, k)
            make_csv("None", k)
        return

    path_is_running = Path(f"../predictions/.cyanno_{filename}")
    if path_is_running.exists():
        print(f"Path {path_is_running} exists. Continuing.")
        return
    path_is_running.touch()

    prepare_input(filename)
    prepare_data(filename)

    for k in [2, 4, 5, 6, 7, 8, 10]:
        run_cyanno(filename, k)
        make_csv(filename, k)

    clean(filename)


def main():
    run_cyanno(None, "all")
    make_csv("None", "all")


if __name__ == "__main__":
    main()
