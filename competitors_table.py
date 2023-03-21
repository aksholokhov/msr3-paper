import os.path

import pandas as pd
import numpy as np
from datetime import datetime
import seaborn as sns
from matplotlib import pyplot as plt
from pathlib import Path
import argparse
from jcgf_generate_all import logs_folder, tables_folder, figures_folder

parser = argparse.ArgumentParser('competitors_table.py')
# experiment settings
parser.add_argument('--file_name', type=str, default="competitors_0.log")
parser.add_argument('--ic', type=str, default="jones")

args = parser.parse_args()

mean_quantiles = lambda \
        s: f"{100 * np.mean(s):.0f} ({100 * np.percentile(s, q=5):.0f}-{100 * np.percentile(s, q=95):.0f})"
mean_quantiles_2 = lambda s: f"{np.mean(s):.0f} ({np.percentile(s, q=5):.0f}-{np.percentile(s, q=95):.0f})"
mean_quantiles_3 = lambda s: f"{np.mean(s):.2f} ({np.percentile(s, q=5):.2f}-{np.percentile(s, q=95):.2f})"

if __name__ == "__main__":
    data = pd.read_csv(logs_folder / args.file_name)

    data["tp"] = data["fe_tp"] + data["re_tp"]
    data["tn"] = data["fe_tn"] + data["re_tn"]
    data["fp"] = data["fe_fp"] + data["re_fp"]
    data["fn"] = data["fe_fn"] + data["re_fn"]
    data["sensitivity"] = data["tp"] / (data["tp"] + data["fn"])
    data["fpr"] = data["fp"] / (data["fp"] + data["tn"])
    data["f1"] = 2 * data["tp"] / (2 * data["tp"] + data["fp"] + data["fn"])
    data["mf1"] = -data["f1"]
    data["acc"] = (data["tp"] + data["tn"]) / (data["tp"] + data["fn"] + data["tn"] + data["fp"])
    data["macc"] = -data["acc"]
    data["fe_f1"] = 2 * data["fe_tp"] / (2 * data["fe_tp"] + data["fe_fp"] + data["fe_fn"])
    data["fe_acc"] = (data["fe_tp"] + data["fe_tn"]) / (
                data["fe_tp"] + data["fe_fn"] + data["fe_tn"] + data["fe_fp"])
    data["re_f1"] = 2 * data["re_tp"] / (2 * data["re_tp"] + data["re_fp"] + data["re_fn"])
    data["re_acc"] = (data["re_tp"] + data["re_tn"]) / (
                data["re_tp"] + data["re_fn"] + data["re_tn"] + data["re_fp"])

    algorithms = data["model"].unique()

    table_long = pd.DataFrame(index=pd.Index(["Accuracy", "FE Accuracy", "RE Accuracy", "F1", "FE F1", "RE F1",
                                                    "Time", "Iterations"]),
                              columns=pd.Index(algorithms, name="algorithm"))

    for name in algorithms:

        f1_scores_pgd = []
        pgd_acc = []
        pgd_time = []
        pgd_iter = []
        pgd_fe_f1 = []
        pgd_fe_acc = []
        pgd_re_f1 = []
        pgd_re_acc = []

        for trial in data['trial'].unique():

            trial_data = data[data['trial'] == trial]

            model_data = trial_data[trial_data["model"] == name]

            if len(model_data) > 0 and model_data['converged'].mean() > 0.9:
                pgd_argmin = model_data[args.ic].astype(float).argmin()
                f1_scores_pgd.append(model_data.iloc[pgd_argmin]["f1"])
                pgd_acc.append(model_data.iloc[pgd_argmin]["acc"])
                pgd_fe_f1.append(model_data.iloc[pgd_argmin]["fe_f1"])
                pgd_re_f1.append(model_data.iloc[pgd_argmin]["re_f1"])
                pgd_fe_acc.append(model_data.iloc[pgd_argmin]["fe_acc"])
                pgd_re_acc.append(model_data.iloc[pgd_argmin]["re_acc"])
                pgd_time.append(model_data.iloc[pgd_argmin]["time"])
                pgd_iter.append(model_data.iloc[pgd_argmin]["number_of_iterations"])

            # print(f"{name}-{trial}: PGD-{pgd_data['f1'].max():.2f}, SR3-{sr3_data['f1'].max():.2f}")

        if len(f1_scores_pgd) > 0:
            table_long.loc['F1', name] = mean_quantiles(f1_scores_pgd)
            table_long.loc['Accuracy', name] = mean_quantiles(pgd_acc)
            table_long.loc['FE F1', name] = mean_quantiles(pgd_fe_f1)
            table_long.loc['FE Accuracy', name] = mean_quantiles(pgd_fe_acc)
            table_long.loc['RE F1', name] = mean_quantiles(pgd_re_f1)
            table_long.loc['RE Accuracy', name] = mean_quantiles(pgd_re_acc)
            table_long.loc['Time', name] = mean_quantiles_3(pgd_time)
            table_long.loc['Iterations', name] = mean_quantiles_2(pgd_iter)

    table_long.to_latex(tables_folder / f"competitors_table_long_{args.file_name}.tex")
