"""
Aggregate per-prefix LPP metrics (output of lpp.py) into one profile per model.

Default aggregation, as in the paper: minimum entropy, maximum participation ratio and
maximum effective rank over prefix lengths, for a fixed dataset, context length and layer.

    python aggregate.py --in_csv results/lpp_alpaca.csv --context_length 200 \
                        --out_csv results/lpp_profile.csv
"""
import argparse

import pandas as pd

AGGS = ["min", "max", "mean", "median"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--context_length", type=int, default=200)
    ap.add_argument("--layer", default="last", help="'last' or an integer layer index")
    ap.add_argument("--entropy_agg", choices=AGGS, default="min")
    ap.add_argument("--rep_agg", choices=AGGS, default="max", help="Aggregation for PR and ER")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    df = df[df["context_length"] == args.context_length]
    if "layer" in df.columns:
        if args.layer == "last":
            df = df[df["layer"] == df.groupby("model_id")["layer"].transform("max")]
        else:
            df = df[df["layer"] == int(args.layer)]

    prof = df.groupby(["model_id", "dataset"]).agg(
        entropy=("mean_next_token_entropy", args.entropy_agg),
        participation_ratio=("participation_ratio", args.rep_agg),
        effective_rank=("effective_rank", args.rep_agg),
    ).reset_index()
    prof.to_csv(args.out_csv, index=False)
    print(prof.to_string(index=False))


if __name__ == "__main__":
    main()
