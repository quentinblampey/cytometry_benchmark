import argparse
import random

import numpy as np
import pandas as pd
from ACDC import AcdcModel


def main(args):
    if args.n_obs is not None:
        args.n_seed = 1

    for i in range(1 + args.n_seed):
        print("--- Run:", i)
        np.random.seed(i)
        random.seed(i)

        model = AcdcModel(args.dataset)

        if args.n_obs is not None:
            model.sub(args.n_obs)

        ids, y_pred, y_pred_index, y_pred_oracle = model(mode=args.mode)

        for y, name in [
            (y_pred, "acdc"),
            (y_pred_index, "baseline"),
            (y_pred_oracle, "pheno"),
        ]:
            if args.n_obs is None and y is not None:
                y_ = [model.idx2ct[int(c)] for c in y]
                df = pd.DataFrame({"id": ids, "y_pred": y_})
                df.to_csv(
                    f"../predictions/{name}_{args.dataset}_I{i}-m7.csv", index=None
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="poised",
        help="AML / BMMC / poised",
    )
    parser.add_argument(
        "-n",
        "--n_seed",
        type=int,
        default=1,
        help="Number of seed.",
    )
    parser.add_argument(
        "-obs",
        "--n_obs",
        type=int,
        default=None,
        help="Number of observations.",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=int,
        default=0,
        help="Mode 1 = run only ACDC, 2 = run only phenograph.",
    )

    main(parser.parse_args())
