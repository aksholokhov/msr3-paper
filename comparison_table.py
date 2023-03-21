import os.path

import pandas as pd
import numpy as np
import re
from datetime import datetime
import seaborn as sns
from matplotlib import pyplot as plt
from pathlib import Path
import argparse

parser = argparse.ArgumentParser('pysr3 experiments')
# experiment settings
parser.add_argument('--experiments', type=tuple, default=("L0", "L1", "ALASSO", "SCAD"))
# 5457071 -> went to JCGS
parser.add_argument('--experiment_folder', type=str, default="results/local_debug_2023-03-20_15:09:35")
parser.add_argument('--ic', type=str, default="jones")
parser.add_argument('--plot_etas_experiment', type=bool, default=True)  # true for eta_dynamics


def generate_benchmark_table(args):
    experiment_folder = Path(args.experiment_folder)

    table_long_index = pd.MultiIndex.from_product([["PGD", "MSR3", "MSR3-fast"],
                                                   ["Accuracy", "FE Accuracy", "RE Accuracy", "F1", "FE F1", "RE F1",
                                                    "Time", "Iterations"]], names=["Model", "Metric"])
    table_long = pd.DataFrame(index=table_long_index,
                              columns=pd.Index({}, name="Regularizer")).transpose()

    plots_data = pd.DataFrame(columns=["Model", "Accuracy", "Time", "Regularizer"])

    data_files = {}
    for alg in args.experiments:
        batches = []
        for path in (experiment_folder / "logs").iterdir():
            if path.name.split("_")[0] == alg:
                batches.append(pd.read_csv(path))
        if len(batches) == 0:
            continue
        merged = pd.concat(batches)
        # merged.to_csv(experiment_folder / "logs" / f"{alg}.log")
        data_files[alg] = merged

    for i, (name, data) in enumerate(data_files.items()):
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

        if args.plot_etas_experiment and name == "L1":
            y_axis = "acc"
            for trial in [1, ]:
                trial_data = data[(data['trial'] == trial) & (data['model'] == f"SR3-{name}")]
                plt.semilogx(trial_data["ell"], trial_data[y_axis], 'b', label="MSR3 L1")
                pgd_data = data[(data['trial'] == trial) & (data['model'] == f"{name}")]
                plt.semilogx(trial_data["ell"], [pgd_data.iloc[0][y_axis]] * len(trial_data, ), '--r', label="PGD L1")
            # The first limits are for the "extended range" picture, where eta goes from 1e-4 to 1e2
            # The second limits are for the "normal range" picture, where eta goes from 1e-2 to 1e1
            #plt.text(x=1e-3, y=0.75, s="Loose")
            plt.text(x=1e-2, y=0.75, s="Loose")
            plt.text(x=5e-1, y=0.75, s="Optimal")
            #plt.text(x=3e1, y=0.75, s="Tight")
            plt.text(x=8, y=0.75, s="Tight")
            plt.title(r"Sensitivity of MSR3 to the relaxation parameter $\eta$")
            plt.xlabel(r"$\eta$, MSR3 relaxation parameter")
            plt.ylabel("Accuracy of fixed and random effects identification")
            plt.legend()
            plt.savefig(experiment_folder / "figures" / f"eta_dependence.pdf")
            plt.show()

        f1_scores_pgd = []
        pgd_acc = []
        pgd_time = []
        pgd_iter = []
        pgd_fe_f1 = []
        pgd_fe_acc = []
        pgd_re_f1 = []
        pgd_re_acc = []

        f1_scores_sr3 = []
        sr3_acc = []
        sr3_time = []
        sr3_iter = []
        sr3_fe_f1 = []
        sr3_fe_acc = []
        sr3_re_f1 = []
        sr3_re_acc = []
        sr3_good_reason = []

        f1_scores_sr3p = []
        sr3p_acc = []
        sr3p_time = []
        sr3p_iter = []
        sr3p_fe_f1 = []
        sr3p_fe_acc = []
        sr3p_re_f1 = []
        sr3p_re_acc = []
        sr3p_good_reason = []

        for trial in data['trial'].unique():
            trial_data = data[data['trial'] == trial]

            pgd_data = trial_data[trial_data["model"] == name]

            if len(pgd_data) > 0 and pgd_data['converged'].mean() > 0.9:
                pgd_argmin = pgd_data[args.ic].argmin()
                f1_scores_pgd.append(pgd_data.iloc[pgd_argmin]["f1"])
                pgd_acc.append(pgd_data.iloc[pgd_argmin]["acc"])
                pgd_fe_f1.append(pgd_data.iloc[pgd_argmin]["fe_f1"])
                pgd_re_f1.append(pgd_data.iloc[pgd_argmin]["re_f1"])
                pgd_fe_acc.append(pgd_data.iloc[pgd_argmin]["fe_acc"])
                pgd_re_acc.append(pgd_data.iloc[pgd_argmin]["re_acc"])
                pgd_time.append(pgd_data.iloc[pgd_argmin]["time"])
                pgd_iter.append(pgd_data.iloc[pgd_argmin]["number_of_iterations"])

            sr3_data = trial_data[trial_data["model"] == f"SR3-{name}"]
            if len(sr3_data) > 0 and sr3_data['converged'].mean() > 0.9:
                sr3_argmin = sr3_data[args.ic].argmin()
                f1_scores_sr3.append(sr3_data.iloc[sr3_argmin]["f1"])
                sr3_acc.append(sr3_data.iloc[sr3_argmin]["acc"])
                sr3_fe_f1.append(sr3_data.iloc[sr3_argmin]["fe_f1"])
                sr3_fe_acc.append(sr3_data.iloc[sr3_argmin]["fe_acc"])
                sr3_re_f1.append(sr3_data.iloc[sr3_argmin]["re_f1"])
                sr3_re_acc.append(sr3_data.iloc[sr3_argmin]["re_acc"])
                sr3_time.append(sr3_data.iloc[sr3_argmin]["time"])
                sr3_iter.append(sr3_data.iloc[sr3_argmin]["number_of_iterations"])
                # sr3_good_reason.append(sr3_data[sr3_argmin]["good_stopping_reason"])

            sr3p_data = trial_data[trial_data["model"] == f"SR3-{name}-P"]
            if len(sr3p_data) > 0 and sr3p_data['converged'].mean() > 0.9:
                sr3p_argmin = sr3p_data[args.ic].argmin()
                f1_scores_sr3p.append(sr3p_data.iloc[sr3p_argmin]["f1"])
                sr3p_acc.append(sr3p_data.iloc[sr3p_argmin]["acc"])
                sr3p_fe_f1.append(sr3p_data.iloc[sr3p_argmin]["fe_f1"])
                sr3p_fe_acc.append(sr3p_data.iloc[sr3p_argmin]["fe_acc"])
                sr3p_re_f1.append(sr3p_data.iloc[sr3p_argmin]["re_f1"])
                sr3p_re_acc.append(sr3p_data.iloc[sr3p_argmin]["re_acc"])
                sr3p_time.append(sr3p_data.iloc[sr3p_argmin]["time"])
                sr3p_iter.append(sr3p_data.iloc[sr3p_argmin]["number_of_iterations"])
                # sr3p_good_reason.append(sr3p_data[sr3p_argmin]["good_stopping_reason"])

            # print(f"{name}-{trial}: PGD-{pgd_data['f1'].max():.2f}, SR3-{sr3_data['f1'].max():.2f}")

        mean_df = pd.DataFrame(np.array([
            ["PGD"] * len(pgd_acc) + ["MSR3"] * len(sr3_acc) + ["MSR3-fast"] * len(sr3p_acc),
            pgd_acc + sr3_acc + sr3p_acc,
            pgd_time + sr3_time + sr3p_time,
            [name] * len(pgd_acc + sr3_acc + sr3p_acc)
        ]).T,
                               columns=["Model", "Accuracy", "Time", "Regularizer"]
                               )
        plots_data = plots_data.append(mean_df)

        mean_quantiles = lambda \
                s: f"{100 * np.mean(s):.0f} ({100 * np.percentile(s, q=5):.0f}-{100 * np.percentile(s, q=95):.0f})"
        mean_quantiles_2 = lambda s: f"{np.mean(s):.0f} ({np.percentile(s, q=5):.0f}-{np.percentile(s, q=95):.0f})"
        mean_quantiles_3 = lambda s: f"{np.mean(s):.2f} ({np.percentile(s, q=5):.2f}-{np.percentile(s, q=95):.2f})"

        if len(f1_scores_pgd) > 0:
            table_long.loc[name, ('PGD', 'F1')] = mean_quantiles(f1_scores_pgd)
            table_long.loc[name, ('PGD', 'Accuracy')] = mean_quantiles(pgd_acc)
            table_long.loc[name, ('PGD', 'FE F1')] = mean_quantiles(pgd_fe_f1)
            table_long.loc[name, ('PGD', 'FE Accuracy')] = mean_quantiles(pgd_fe_acc)
            table_long.loc[name, ('PGD', 'RE F1')] = mean_quantiles(pgd_re_f1)
            table_long.loc[name, ('PGD', 'RE Accuracy')] = mean_quantiles(pgd_re_acc)
            table_long.loc[name, ('PGD', 'Time')] = mean_quantiles_3(pgd_time)
            table_long.loc[name, ('PGD', 'Iterations')] = mean_quantiles_2(pgd_iter)

        if len(f1_scores_sr3):
            table_long.loc[name, ('MSR3', 'F1')] = mean_quantiles(f1_scores_sr3)
            table_long.loc[name, ('MSR3', 'Accuracy')] = mean_quantiles(sr3_acc)
            table_long.loc[name, ('MSR3', 'FE F1')] = mean_quantiles(sr3_fe_f1)
            table_long.loc[name, ('MSR3', 'FE Accuracy')] = mean_quantiles(sr3_fe_acc)
            table_long.loc[name, ('MSR3', 'RE F1')] = mean_quantiles(sr3_re_f1)
            table_long.loc[name, ('MSR3', 'RE Accuracy')] = mean_quantiles(sr3_re_acc)
            table_long.loc[name, ('MSR3', 'Time')] = mean_quantiles_3(sr3_time)
            table_long.loc[name, ('MSR3', 'Iterations')] = mean_quantiles_2(sr3_iter)
            # table_long.loc[name, ('MSR3', 'Good Stopping Reason')] = mean_quantiles(sr3_good_reason)

        if len(f1_scores_sr3p):
            table_long.loc[name, ('MSR3-fast', 'F1')] = mean_quantiles(f1_scores_sr3p)
            table_long.loc[name, ('MSR3-fast', 'Accuracy')] = mean_quantiles(sr3p_acc)
            table_long.loc[name, ('MSR3-fast', 'FE F1')] = mean_quantiles(sr3p_fe_f1)
            table_long.loc[name, ('MSR3-fast', 'FE Accuracy')] = mean_quantiles(sr3p_fe_acc)
            table_long.loc[name, ('MSR3-fast', 'RE F1')] = mean_quantiles(sr3p_re_f1)
            table_long.loc[name, ('MSR3-fast', 'RE Accuracy')] = mean_quantiles(sr3p_re_acc)
            table_long.loc[name, ('MSR3-fast', 'Time')] = mean_quantiles_3(sr3p_time)
            table_long.loc[name, ('MSR3-fast', 'Iterations')] = mean_quantiles_2(sr3p_iter)
            # table_long.loc[name, ('MSR3-fast', 'Good Stopping Reason')] = mean_quantiles(sr3p_good_reason)

    table_long = table_long.transpose()
    plots_data["Accuracy"] = plots_data["Accuracy"].astype(float)
    plots_data["Time"] = plots_data["Time"].astype(float)

    # Generate plots
    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(14, 4)
    sns.boxplot(x="Regularizer", y="Accuracy", hue="Model", data=plots_data, palette="Set3", width=0.5, ax=ax[0])
    sns.boxplot(x="Regularizer", y="Time", hue="Model", data=plots_data, palette="Set3", width=0.5, ax=ax[1])
    ax[1].set_yscale("log")
    ax[0].legend(loc='upper right', bbox_to_anchor=(-0.08, 1.02),
                 ncol=1, fancybox=True)
    ax[1].legend(loc='upper left', bbox_to_anchor=(1.03, 1.02),
                 ncol=1, fancybox=True)
    plt.savefig(experiment_folder / "figures" / "benchmark.pdf")
    plt.show()

    # generate small summary table
    index_row = pd.MultiIndex.from_product([data_files.keys(), ["Accuracy", "Time"]], names=["Regilarizer", "Metric"])
    index_col = pd.Index(["PGD", "MSR3", "MSR3-fast"], name="Model")
    table_short = pd.DataFrame(index=index_row, columns=index_col)
    for reg in data_files.keys():
        for model in ["PGD", "MSR3", "MSR3-fast"]:
            for metric in ["Accuracy", "Time"]:
                table_short.loc[(reg,
                                 metric), model] = f'{plots_data[(plots_data["Model"] == model) & (plots_data["Regularizer"] == reg)][metric].mean():.2f}'

    def bold_extreme_values(data, max_=True):
        if max_:
            extrema = data != data.max()
        else:
            extrema = data != data.min()
        return data.where(extrema, data.apply(lambda x: "\\textbf{%s}" % x))

    row_show_max = {"Accuracy": True, "Time": False}
    for row_1 in table_short.index.get_level_values(0).unique():
        for row_2 in table_short.index.get_level_values(1).unique():
            table_short.loc[row_1, row_2] = table_short.loc[row_1, row_2]

    # save everything
    table_short.to_csv(experiment_folder / "tables" / f"performance_table_short.csv")
    table_short.to_latex(experiment_folder / "tables" / f"performance_table_short.tex", escape=False)
    table_short[['PGD', 'MSR3-fast']].to_latex(experiment_folder / "tables" / f"performance_table_short_wo_msr3.tex",
                                               escape=False)
    # table_long[['L1', 'ALASSO', 'SCAD']].to_latex(tables_folder / f"performance_table_long_1_{args.folder_id}.tex")
    table_long.to_latex(experiment_folder / "tables" / f"performance_table_long.tex")


if __name__ == "__main__":
    args = parser.parse_args()
    generate_benchmark_table(args=args)
