import argparse
import random

import numpy as np
import pandas as pd
from src import MpModel


def main(args):
    print("--- Run:", 0)
    np.random.seed(0)
    random.seed(0)

    model = MpModel(args.dataset)

    if args.n_obs is not None:
        model.sub(args.n_obs)

    ids, y_pred = model(True)

    if args.n_obs is None:
        y_ = [model.idx2ct[int(c)] for c in y_pred]
        df = pd.DataFrame({"id": ids, "y_pred": y_})
        df.to_csv(f"../predictions/mp_{args.dataset}_I0-m5.csv", index=None)


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
        "-obs",
        "--n_obs",
        type=int,
        default=None,
        help="Number of observations.",
    )

    main(parser.parse_args())
