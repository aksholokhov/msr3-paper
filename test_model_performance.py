import shutil
import time
from pathlib import Path

import dask
import numpy as np
import pandas as pd
from dask.distributed import Client, as_completed, TimeoutError, LocalCluster
from matplotlib import pyplot as plt
from pysr3.lme.problems import LMEProblem
from sklearn.metrics import mean_squared_error, explained_variance_score, confusion_matrix
from tqdm import tqdm


def golden_search(f, a, b, tol=1e-2):
    import math

    invphi = (math.sqrt(5) - 1) / 2  # 1 / phi
    invphi2 = (3 - math.sqrt(5)) / 2  # 1 / phi^2

    (a, b) = (min(a, b), max(a, b))
    h = b - a
    if h <= tol:
        return (a, b)

    # Required steps to achieve tolerance
    n = int(math.ceil(math.log(tol / h) / math.log(invphi)))

    c = a + invphi2 * h
    d = a + invphi * h
    yc = f(c)
    yd = f(d)

    for k in range(n - 1):
        if yc < yd:
            b = d
            d = c
            yd = yc
            h = invphi * h
            c = a + invphi2 * h
            yc = f(c)
        else:
            a = c
            c = d
            yc = yd
            h = invphi * h
            d = a + invphi * h
            yd = f(d)
    if yc < yd:
        return (a + d) / 2
    else:
        return (c + b) / 2


def run_comparison_experiment(args,
                              models,
                              problem_parameters,
                              lambda_search_bounds,
                              ell_search_grid,
                              true_beta,
                              true_gamma,
                              non_regularized_model=None,
                              tol=0.,
                              dask_folder=".",
                              data_folder='.'
                              ):
    log = pd.DataFrame(columns=("trial", "param", "model", "time", "mse", "evar",
                                "fe_tp", "fe_tn", "fe_fp", "fe_fn",
                                "re_tp", "re_tn", "re_fp", "re_fn",
                                "number_of_iterations", "converged", "good_stopping_reason"))

    def measure_models(model_constructor, model_name, search_bounds, ell, trial_num, problem, discrete=False):
        x, y, columns_labels = problem.to_x_y()

        if non_regularized_model:
            # ALASSO part
            non_regularized_model.fit_problem(problem)

            fe_regularization_weights = 1 / (np.abs(non_regularized_model.coef_['beta']) + 1e-2)
            re_regularization_weights = 1 / (np.abs(non_regularized_model.coef_['gamma']) + 1e-2)
        else:
            fe_regularization_weights = None
            re_regularization_weights = None

        def fit_and_score(param, ic='jones_bic'):
            model = model_constructor(param, ell)
            model.fit_problem(problem,
                              fe_regularization_weights=fe_regularization_weights,
                              re_regularization_weights=re_regularization_weights)
            if not model.logger_.get("converged"):
                if args.verbose:
                    print(f"{model_name} did not converge for trial={trial_num} param={param}", flush=True)
                return np.infty
            return model.logger_.get(ic)

        # Hyperparameter's tuning
        if "L0" in model_name:
            best_param = np.argmin([fit_and_score(k) for k in range(search_bounds[0], search_bounds[1])])
        elif "pco" in model_name:
            best_param = 0  # PCO is too slow for hyper-parameter optimization
        elif "fence" in model_name:
            best_param = None  # Fence does not have any hyper-parameters
        else:
            best_param = golden_search(fit_and_score, a=search_bounds[0], b=search_bounds[1])

        model = model_constructor(best_param, ell)
        tic = time.perf_counter()
        model.fit_problem(problem,
                          fe_regularization_weights=fe_regularization_weights,
                          re_regularization_weights=re_regularization_weights)
        toc = time.perf_counter()
        if not model.logger_.get("converged"):
            if args.verbose:
                print(f"{model_name} did not converge for trial={trial_num} param={best_param}, ell={ell}")

        y_pred = model.predict_problem(problem)
        fe_tn, fe_fp, fe_fn, fe_tp = confusion_matrix(true_beta != 0,
                                                      np.abs(model.coef_["beta"]) >= tol).ravel()
        re_tn, re_fp, re_fn, re_tp = confusion_matrix(true_gamma != 0,
                                                      np.abs(model.coef_["gamma"]) >= tol).ravel()

        results = {
            "trial": trial_num,
            "param": best_param,
            "model": model_name,
            "ell": ell,
            "time": toc - tic,
            "converged": model.logger_.get("converged"),
            "mse": mean_squared_error(y, y_pred),
            "evar": explained_variance_score(y, y_pred),
            "muller": model.logger_.get("muller_ic"),
            "jones": model.logger_.get("jones_bic"),
            "vaida": model.logger_.get("vaida_aic"),
            "vaida_m": model.logger_.get("vaida_aic_marginalized"),
            "fe_tp": fe_tp,
            "fe_tn": fe_tn,
            "fe_fp": fe_fp,
            "fe_fn": fe_fn,
            "re_tp": re_tp,
            "re_tn": re_tn,
            "re_fp": re_fp,
            "re_fn": re_fn,
            "number_of_iterations": model.logger_.get("iteration"),
        }
        return results

    jobs = []
    for i in range(args.trials_from, args.trials_to):
        seed = i + args.random_seed
        np.random.seed(seed)

        problem, true_model_parameters = LMEProblem.generate(**problem_parameters,
                                                             beta=true_beta,
                                                             gamma=true_gamma,
                                                             seed=seed)
        dataset_path = Path(data_folder) / f"problem_{i}.csv"
        if not dataset_path.exists():
            problem.to_dataframe().to_csv(dataset_path)
        for model_name, model_constructor in models.items():
            if model_name.startswith("SR3"):
                for ell in ell_search_grid:
                    job_params = (model_constructor,
                                  model_name,
                                  lambda_search_bounds,
                                  ell,
                                  i,
                                  problem,
                                  "L0" in model_name)
                    jobs.append(job_params)
            else:
                job_params = (model_constructor,
                              model_name,
                              lambda_search_bounds,
                              None,
                              i,
                              problem,
                              "L0" in model_name)
                jobs.append(job_params)

    print(f"Jobs to process: {len(jobs)}")

    futures = []
    all_results = []
    try:
        if args.use_dask:
            print(f"Processing with Dask.Distributed, with {args.n_dask_workers} workers in parallel.")
            Path(dask_folder).mkdir(parents=True, exist_ok=True)
            try:
                with dask.config.set({'temporary_directory': dask_folder}):
                    with LocalCluster(threads_per_worker=1,
                                      processes=True,
                                      host="localhost",
                                      dashboard_address="localhost:0",
                                      worker_dashboard_address="localhost:0",
                                      n_workers=args.n_dask_workers
                                      ) as cluster, \
                            Client(cluster) as client:
                        for job_params in jobs:
                            futures.append(client.submit(measure_models, *job_params))
                        for future in tqdm(as_completed(futures)):
                            try:
                                results = future.result(timeout=3)
                                all_results.append(results)
                            except TimeoutError:
                                continue
            finally:
                # delete temporary dask folder
                shutil.rmtree(dask_folder)
        else:
            print(f"Processing in a single thread. Brace yourself, it may take a while..."
                  f" (Consider passing --use_dask for multithread execution)")
            for job_params in tqdm(jobs):
                results = measure_models(*job_params)
                all_results.append(results)

    finally:
        log = pd.DataFrame.from_records(all_results, columns=["trial", "ell"])
        log = log.sort_values(by=["trial", "ell"])

    return log


def plot_comparison(data, trial_num=0, suffix=None, figures_folder=".", title="Title", xlabel="x", ylabel="y",
                    ic="muller"):
    data["tp"] = data["fe_tp"] + data["re_tp"]
    data["tn"] = data["fe_tn"] + data["re_tn"]
    data["fp"] = data["fe_fp"] + data["re_fp"]
    data["fn"] = data["fe_fn"] + data["re_fn"]

    data["fe_sensitivity"] = data["fe_tp"] / (data["fe_tp"] + data["fe_fn"])
    data["fe_specificity"] = data["fe_tn"] / (data["fe_tn"] + data["fe_fp"])
    data["fe_fpr"] = data["fe_fp"] / (data["fe_fp"] + data["fe_tn"])
    data["fe_f1"] = 2 * data["fe_tp"] / (2 * data["fe_tp"] + data["fe_fp"] + data["fe_fn"])
    data["fe_acc"] = (data["fe_tp"] + data["fe_tn"]) / (data["fe_tp"] + data["fe_fn"] + data["fe_tn"] + data["fe_fp"])

    data["re_sensitivity"] = data["re_tp"] / (data["re_tp"] + data["re_fn"])
    data["re_specificity"] = data["re_tn"] / (data["re_tn"] + data["re_fp"])
    data["re_fpr"] = data["re_fp"] / (data["re_fp"] + data["re_tn"])
    data["re_f1"] = 2 * data["re_tp"] / (2 * data["re_tp"] + data["re_fp"] + data["re_fn"])
    data["re_acc"] = (data["re_tp"] + data["re_tn"]) / (data["re_tp"] + data["re_fn"] + data["re_tn"] + data["re_fp"])

    data["sensitivity"] = data["tp"] / (data["tp"] + data["fn"])
    data["fpr"] = data["fp"] / (data["fp"] + data["tn"])
    data["f1"] = 2 * data["tp"] / (2 * data["tp"] + data["fp"] + data["fn"])
    data["acc"] = (data["tp"] + data["tn"]) / (data["tp"] + data["fn"] + data["tn"] + data["fp"])

    base_size = 5
    fig = plt.figure(figsize=(3 * base_size, 2 * base_size))
    grid = plt.GridSpec(nrows=3, ncols=2)
    fe_plot = fig.add_subplot(grid[0, :2])
    re_plot = fig.add_subplot(grid[1, :2])
    ic_plot = fig.add_subplot(grid[2, :2])

    data = data[data["trial"] == trial_num]

    for model in data["model"].unique():
        model_data = data[data["model"] == model]
        fe_plot.semilogx(model_data["params"], model_data["fe_f1"], label=model)
        re_plot.semilogx(model_data["params"], model_data["re_f1"], label=model)
        ic_plot.loglog(model_data["params"], model_data["muller"], label=f"{ic} for {model}")

    fe_plot.legend(loc="lower left")
    fe_plot.set_xlabel(xlabel)
    fe_plot.set_ylabel(r"F1, selection quality for fixed effects")
    fe_plot.set_title("Fixed-effects selection quality")

    re_plot.legend(loc="lower left")
    re_plot.set_xlabel(xlabel)
    re_plot.set_ylabel(r"F1, selection quality for random effects")
    re_plot.set_title("Random-effects selection quality")

    ic_plot.legend(loc="lower left")
    ic_plot.set_xlabel(xlabel)
    ic_plot.set_ylabel(r"Information Criterion")
    ic_plot.set_title(f"Value of {ic} Information Criterion")

    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    plot_filename = Path(figures_folder) / f"{title}_comparison_{suffix if suffix else ''}.pdf"
    plt.savefig(plot_filename)
    print(f"{title}: plot saved as {plot_filename}")
