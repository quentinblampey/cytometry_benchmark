from collections import Counter

import anndata
import numpy as np
import pandas as pd
import phenograph

from .cell_type_annotation import *
from .random_walk_classifier import *

n_neighbor = 10
thres = 0.5


class AcdcModel:
    def __init__(self, dataset="AML") -> None:
        self.dataset = dataset

        if dataset == "AML":
            path = "ACDC/AML_benchmark/"
            df = pd.read_csv(
                path + "AML_benchmark.csv.gz", sep=",", header=0, compression="gzip"
            )

            df = df.drop(
                [
                    "Time",
                    "Cell_length",
                    "file_number",
                    "event_number",
                    "DNA1(Ir191)Di",
                    "DNA2(Ir193)Di",
                    "Viability(Pt195)Di",
                    "subject",
                ],
                axis=1,
            )

            channels = [item[: item.find("(")] for item in df.columns[:-1]]
            df.columns = channels + ["cell_type"]

            df = df.loc[df["cell_type"] != "NotDebrisSinglets"]

            table = pd.read_csv(path + "AML_table.csv", sep=",", header=0, index_col=0)
        elif dataset == "BMMC":
            self.channels = [
                "CD45",
                "CD45RA",
                "CD19",
                "CD11b",
                "CD4",
                "CD8",
                "CD34",
                "CD20",
                "CD33",
                "CD123",
                "CD38",
                "CD90",
                "CD3",
            ]

            path = "ACDC/BMMC_benchmark/"

            df = pd.read_csv(
                path + "BMMC_benchmark.csv.gz", sep=",", header=0, compression="gzip"
            )
            df = df[df.cell_type != "NotGated"]

            table = pd.read_csv(path + "BMMC_table.csv", sep=",", header=0, index_col=0)
        elif dataset == "PD":
            path = "ACDC/PD_benchmark/"

            adata = anndata.read_h5ad(path + "pop_durva.h5ad")

            table = pd.read_csv(path + "pop_durva.csv", sep=",", header=0, index_col=0)
        elif dataset == "poised":
            path = "../POISED/"

            adata = anndata.read_h5ad(path + "no_pp.h5ad")
            adata.obs["cell_type"] = adata.obs.labels
            adata.X = np.arcsinh((adata.X - 1.0) / 5.0)

            table = pd.read_csv(path + "table.csv", sep=",", header=0, index_col=0)
        elif dataset == "debarcoding":
            path = "ACDC/debarcoding/"

            adata = anndata.read_h5ad(path + "auto_logicle.h5ad")

            table = pd.read_csv(path + "table.csv", sep=",", header=0, index_col=0)
        else:
            raise NameError(f"Invalid dataset name {dataset}")

        self.table = table.fillna(0)

        cts, self.channels = get_label(self.table)

        if dataset in ["PD", "poised", "debarcoding"]:
            self.adata = adata
            self.X0 = adata[:, self.channels].X
            self.ids = adata.obs.index.values
            # - np.quantile(
            #     adata[:, self.channels].X, 0.1, axis=0
            # )
        else:
            self.X0 = np.arcsinh((df[self.channels].values - 1.0) / 5.0)

        self.idx2ct = [key for idx, key in enumerate(self.table.index)]
        self.idx2ct.append("unknown")

        self.ct2idx = {key: idx for idx, key in enumerate(self.table.index)}
        self.ct2idx["unknown"] = len(self.table.index)

        ct_score = np.abs(self.table.values).sum(axis=1)

        ## compute manual gated label
        if dataset == "debarcoding":
            self.y0 = np.zeros(adata.n_obs)
        elif dataset in ["PD", "poised"]:
            self.y0 = np.zeros(adata.n_obs)

            for i, ct in enumerate(adata.obs.cell_type):
                if ct in self.ct2idx:
                    self.y0[i] = self.ct2idx[ct]
                else:
                    self.y0[i] = self.ct2idx["unknown"]
        else:
            self.y0 = np.zeros(df.cell_type.shape)

            for i, ct in enumerate(df.cell_type):
                if ct in self.ct2idx:
                    self.y0[i] = self.ct2idx[ct]
                else:
                    self.y0[i] = self.ct2idx["unknown"]

    def sub(self, n_obs):
        indices = np.random.choice(len(self.X0), n_obs, replace=False)
        self.X0, self.y0, self.ids = (
            self.X0[indices],
            self.y0[indices],
            self.ids[indices],
        )

    def __call__(self, mode=0):
        X = self.X0
        y_true = self.y0

        if mode < 2:
            mk_model = compute_marker_model(
                pd.DataFrame(X, columns=self.channels), self.table, 0.0
            )

            ## compute posterior probs
            score = get_score_mat(X, [], self.table, [], mk_model)
            score = np.concatenate(
                [score, 1.0 - score.max(axis=1)[:, np.newaxis]], axis=1
            )

            ## get indices
            ct_index = get_unique_index(X, score, self.table, thres)

            ## baseline - classify events
            y_pred_index = np.argmax(score, axis=1)

            if mode == -1:
                print("--5")
                return (
                    self.ids,
                    None,
                    y_pred_index,
                    None,
                )

            ## running ACDC
            res_c = get_landmarks(X, score, ct_index, self.idx2ct, phenograph, thres)

            landmark_mat, landmark_label = output_feature_matrix(
                res_c, [self.idx2ct[i] for i in range(len(self.idx2ct))]
            )

            landmark_label = np.array(landmark_label)

            lp, y_pred = rm_classify(X, landmark_mat, landmark_label, n_neighbor)

        if mode == 1:
            return (
                self.ids,
                [self.ct2idx[c] for c in y_pred],
                y_pred_index,
                None,
            )

        k = 6 if self.dataset == "poised" else 30
        res = phenograph.cluster(
            X,
            k=k,
            directed=False,
            prune=False,
            min_cluster_size=10,
            jaccard=True,
            primary_metric="euclidean",
            n_jobs=-1,
            q_tol=1e-3,
        )

        ## running phenograph classification
        index_unknown = len(self.table.index)
        y_pred_oracle = np.zeros_like(y_true)
        for i in range(max(res[0]) + 1):
            most_common = Counter(y_true[res[0] == i]).most_common(2)
            ic, counts = most_common[0]
            if (ic == index_unknown) and (counts <= 0.7 * (res[0] == i).sum()):
                ic, counts = most_common[1]
            y_pred_oracle[res[0] == i] = ic

        if mode == 2:
            return (
                self.ids,
                None,
                None,
                y_pred_oracle,
            )

        return (
            self.ids,
            [self.ct2idx[c] for c in y_pred],
            y_pred_index,
            y_pred_oracle,
        )
