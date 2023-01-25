# Copied from Harmonypy and updated
#
# LISI - The Local Inverse Simpson Index
# Copyright (C) 2018  Ilya Korsunsky
#               2019  Kamil Slowikowski <kslowikowski@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.neighbors import NearestNeighbors

scyan_marker_indices = [
    19,
    27,
    30,
    22,
    23,
    18,
    0,
    37,
    36,
    16,
    10,
    32,
    13,
    15,
    31,
    9,
    6,
    2,
    7,
]


def compute_lisi(
    adata_,
    label: str,
    obsm_key: Optional[str] = None,
    perplexity: float = 30,
    n_cells=100_000,
    random_state=0,
):
    """Compute the Local Inverse Simpson Index (LISI) for each column in metadata.

    LISI is a statistic computed for each item (row) in the data matrix X.

    The following example may help to interpret the LISI values.

    Suppose one of the columns in metadata is a categorical variable with 3 categories.

        - If LISI is approximately equal to 3 for an item in the data matrix,
          that means that the item is surrounded by neighbors from all 3
          categories.

        - If LISI is approximately equal to 1, then the item is surrounded by
          neighbors from 1 category.

    The LISI statistic is useful to evaluate whether multiple datasets are
    well-integrated by algorithms such as Harmony [1].

    [1]: Korsunsky et al. 2019 doi: 10.1038/s41592-019-0619-0
    """
    # We need at least 3 * n_neigbhors to compute the perplexity
    adata = sc.pp.subsample(adata_, n_obs=n_cells, copy=True, random_state=random_state)
    X = adata.X if obsm_key is None else adata.obsm[obsm_key]
    X = X[:, scyan_marker_indices]
    knn = NearestNeighbors(n_neighbors=perplexity * 3, algorithm="kd_tree").fit(X)
    distances, indices = knn.kneighbors(X)

    indices = indices[:, 1:]
    distances = distances[:, 1:]

    labels = pd.Categorical(adata.obs[label])
    n_categories = len(labels.categories)
    simpson = compute_simpson(distances.T, indices.T, labels, n_categories, perplexity)
    return (1 / simpson).mean()


def compute_simpson(
    distances: np.ndarray,
    indices: np.ndarray,
    labels: pd.Categorical,
    n_categories: int,
    perplexity: float,
    tol: float = 1e-5,
):
    n = distances.shape[1]
    P = np.zeros(distances.shape[0])
    simpson = np.zeros(n)
    logU = np.log(perplexity)
    # Loop through each cell.
    for i in range(n):
        beta = 1
        betamin = -np.inf
        betamax = np.inf
        # Compute Hdiff
        P = np.exp(-distances[:, i] * beta)
        P_sum = np.sum(P)
        if P_sum == 0:
            H = 0
            P = np.zeros(distances.shape[0])
        else:
            H = np.log(P_sum) + beta * np.sum(distances[:, i] * P) / P_sum
            P = P / P_sum
        Hdiff = H - logU
        n_tries = 50
        for t in range(n_tries):
            # Stop when we reach the tolerance
            if abs(Hdiff) < tol:
                break
            # Update beta
            if Hdiff > 0:
                betamin = beta
                if not np.isfinite(betamax):
                    beta *= 2
                else:
                    beta = (beta + betamax) / 2
            else:
                betamax = beta
                if not np.isfinite(betamin):
                    beta /= 2
                else:
                    beta = (beta + betamin) / 2
            # Compute Hdiff
            P = np.exp(-distances[:, i] * beta)
            P_sum = np.sum(P)
            if P_sum == 0:
                H = 0
                P = np.zeros(distances.shape[0])
            else:
                H = np.log(P_sum) + beta * np.sum(distances[:, i] * P) / P_sum
                P = P / P_sum
            Hdiff = H - logU
        # distancesefault value
        if H == 0:
            simpson[i] = -1
        # Simpson's index
        for label_category in labels.categories:
            ix = indices[:, i]
            q = labels[ix] == label_category
            if np.any(q):
                P_sum = np.sum(P[q])
                simpson[i] += P_sum * P_sum
    return simpson


def main(args):
    path_data = Path("../POISED/batch_effect")

    filenames = [f"{args.filename}.h5ad"] + [
        f"{args.filename}_{method}.h5ad"
        for method in ["harmony", "saucie", "combat", "cydar"]
    ]

    for filename in filenames:
        print(f"\n--- Running on {filename}...")
        adata = sc.read_h5ad(path_data / filename)
        obsm_key = "X_corrected" if "X_corrected" in adata.obsm else None

        for obs_key in ["batch", "labels"]:

            res = np.array(
                [
                    compute_lisi(
                        adata[adata.obs.labels != "Unknown"].copy(),
                        obs_key,
                        obsm_key=obsm_key,
                        n_cells=args.n,
                        random_state=i,
                    )
                    for i in range(args.k)
                ]
            )
            print(obs_key, res)
            print(res.mean(), res.std())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--filename",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-k",
        "--k",
        type=int,
        default=1,
    )
    parser.add_argument(
        "-n",
        "--n",
        type=int,
        default=100000,
    )

    main(parser.parse_args())
