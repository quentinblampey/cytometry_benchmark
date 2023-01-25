import multiprocessing

import anndata
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from src import *

PATH_DATA = "../acdc/ACDC/"


class MpModel:
    def __init__(self, dataset) -> None:
        if dataset == "AML":
            # load AML data and table

            ### LOAD DATA ###
            path = PATH_DATA
            df = pd.read_csv(
                path + "AML_benchmark.csv.gz",
                sep=",",
                header=0,
                compression="gzip",
                engine="python",
            )
            table = pd.read_csv(path + "AML_table.csv", sep=",", header=0, index_col=0)

            ### PROCESS: discard ungated events ###
            df = df[df.cell_type != "NotGated"]
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
        elif dataset == "BMMC":
            path = PATH_DATA
            df = pd.read_csv(
                path + "BMMC_benchmark.csv.gz",
                sep=",",
                header=0,
                compression="gzip",
                engine="python",
            )
            table = pd.read_csv(path + "BMMC_table.csv", sep=",", header=0, index_col=0)

            channels = [
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
            df.columns = channels + ["cell_type"]
            df = df[df.cell_type != "NotGated"]

            ### five cell types below are the ones that we do not have prior information about.
            ### in acdc implementation, they are all catagorized as "unknown", yet since we are not able
            ### to handle unknown cell types, we remove all instances of these types
            ### proportion of "unknown" is 24.49% in total
            df = df.loc[df["cell_type"] != "Megakaryocyte"]
            df = df.loc[df["cell_type"] != "CD11bmid Monocyte"]
            df = df.loc[df["cell_type"] != "Platelet"]
            df = df.loc[df["cell_type"] != "Myelocyte"]
            df = df.loc[df["cell_type"] != "Erythroblast"]
        elif dataset == "PD":
            adata = anndata.read_h5ad(PATH_DATA + "pop_durva.h5ad")

            table = pd.read_csv(
                PATH_DATA + "pop_durva.csv", sep=",", header=0, index_col=0
            )
        elif dataset == "poised":
            path = "../POISED/"

            adata = anndata.read_h5ad(path + "no_pp.h5ad")
            adata.obs["cell_type"] = adata.obs.labels
            adata.X = np.arcsinh((adata.X - 1.0) / 5.0)

            table = pd.read_csv(path + "table.csv", sep=",", header=0, index_col=0)
        elif dataset == "debarcoding":
            path = "../acdc/ACDC/debarcoding/"

            adata = anndata.read_h5ad(path + "auto_logicle.h5ad")

            table = pd.read_csv(path + "table.csv", sep=",", header=0, index_col=0)
        else:
            raise NameError(f"Invalid dataset name {dataset}")

        self.table = table.fillna(0)

        if dataset in ["PD", "poised", "debarcoding"]:
            self.data = adata[:, table.columns].X
            self.ids = adata.obs.index.values
            # - np.quantile(
            #     adata[:, table.columns].X, 0.1, axis=0
            # )
        else:
            X = df[channels].values

            ### transform data
            self.data = np.arcsinh((X - 1.0) / 5.0)

        self.N, d = self.data.shape
        self.emp_bounds = np.array(
            [
                [self.data[:, d].min(), self.data[:, d].max()]
                for d in range(self.data.shape[1])
            ]
        )
        self.ct2idx = {x: i for i, x in enumerate(table.index)}
        self.idx2ct = [key for idx, key in enumerate(table.index)]

        if dataset == "AML":
            # rename table header 'HLA-DR' to 'HLADR' to prevent error from '-'
            temp_headers = list(self.table)
            temp_headers[29] = "HLADR"
            self.table.columns = temp_headers
            self.table.at["Mature B cells", "CD38"] = -1.0

        print("dataset init")

    def sub(self, n_obs):
        indices = np.random.choice(len(self.data), n_obs, replace=False)
        self.data, self.ids = self.data[indices], self.ids[indices]

    def __call__(self, multipro=True):
        n_mcmc_chain = 50
        n_mcmc_samples = 3000

        if multipro:
            print("multipro")
            chains = range(n_mcmc_chain)
            num_cores = multiprocessing.cpu_count()
            accepted_MP = Parallel(n_jobs=num_cores)(
                delayed(MP_mcmc)(
                    self.data, self.emp_bounds, self.table, i, n_mcmc_samples
                )
                for i in chains
            )
            # write_chains_to_file(accepted_MP, PATH_SAMPLES)

            burnt_samples = [sample for chain in accepted_MP for sample in chain[-10:]]
            Y_predict = classify_cells_majority(
                self.data, burnt_samples, self.table, self.ct2idx
            )
        else:
            accepted_MP = []
            for i in range(n_mcmc_chain):
                print("Sampling Chain %d..." % i)
                accepted_MP.append(
                    MP_mcmc(self.data, self.emp_bounds, self.table, i, n_mcmc_samples)
                )
                burnt_samples = [
                    sample for chain in accepted_MP for sample in chain[-20:]
                ]
                Y_predict = classify_cells_majority(
                    self.data, burnt_samples, self.table, self.ct2idx
                )
                # accuracy = sum(self.Y == Y_predict) * 1.0 / self.N
                # print("Accuracy of cell classification on all data:", accuracy)

        return self.ids, Y_predict
