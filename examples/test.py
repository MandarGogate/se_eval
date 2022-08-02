#  Copyright (c) 2022 Mandar Gogate, All rights reserved.
#  !pip install tabulate pandas numpy
import warnings

warnings.filterwarnings("ignore")
import argparse
from collections import OrderedDict
from pathlib import Path

import pandas as pd
from tabulate import tabulate

from se_eval import get_se_metric


def str2bool(v: str):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_pairs(testing_root, model_uid, clean_root):
    combined_utterances = [(str(testing_root / model_uid / utterance_path.parts[-1]), str(utterance_path)) for
                           utterance_path in clean_root.iterdir()]
    return combined_utterances


if __name__ == '__main__':
    # dBs = [str(db).replace("-", "m") for db in dB_levels]
    parser = argparse.ArgumentParser(description='Usage for se_eval objective evaluation')
    parser.add_argument('--testing_root', type=str, default="", help='Utterances root for noisy, clean and enhanced')
    parser.add_argument('--clean_root', type=str, required=True, help='Reference root')
    parser.add_argument('--matlab_path', type=str, default="", help='Matlab scripts path')
    parser.add_argument("--latex", type=str2bool, default=False, help="Generate latex table")
    parser.add_argument("--multiprocessing", type=str2bool, default=True, help="Use multiprocessing module for faster evaluation")
    parser.add_argument("--cores", type=int, default=None, help="Number of cores to use for multiprocessing")
    parser.add_argument("--fs", type=int, default=16000, help="Sampling frequency")
    parser.add_argument("--metrics", nargs='+', required=True, help="e.g. pesq, sisdr, stoi")
    parser.add_argument("--model_uids", nargs='+', default=[], help="e.g. noisy, baseline")
    args = parser.parse_args()
    metrics = args.metrics
    metrics_data = OrderedDict({
        "Exp": []
    })

    testing_root = Path(args.testing_root)
    clean_root = Path(args.clean_root)

    for model_uid in args.model_uids:
        metrics_data["Exp"].append(model_uid)
        utterance_pairs = get_pairs(testing_root, model_uid, clean_root)
        mean_scores = get_se_metric(metrics, utterance_pairs, args.fs, args.multiprocessing, args.cores, args.matlab_path)
        for metric in mean_scores:
            if metric not in metrics_data:
                metrics_data[metric] = []
            metrics_data[metric].append(mean_scores[metric])

            df = pd.DataFrame(metrics_data)
            print(tabulate(df, tablefmt="pipe", headers="keys"), "\n")

    if args.latex:
        df = pd.DataFrame(metrics_data, index=metrics_data["Exp"])
        print(df.to_latex())
