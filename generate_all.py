import argparse
import datetime
import pickle
from pathlib import Path

import numpy as np
from pysr3.lme.models import SimpleLMEModel, L1LmeModel, L1LmeModelSR3, L0LmeModel, \
    L0LmeModelSR3, SCADLmeModel, SCADLmeModelSR3
from pysr3.lme.problems import FIXED_RANDOM
from multiprocessing import cpu_count

from bullying.bullying_example import generate_bullying_experiment
from comparison_table import generate_benchmark_table
from alternatives.glmmlasso_wrapper import GMMLassoModel
from intuition import run_intuition_experiment, plot_intuition_picture
from alternatives.lmmlasso_wrapper import lmmlassomodel
from test_model_performance import run_comparison_experiment
from competitors_table import generate_competitors_table

parser = argparse.ArgumentParser('pysr3 experiments')
# experiment settings
parser.add_argument('--experiments', type=str, default="intuition,L0,L1,ALASSO,SCAD,bullying", help='Which experiments to run. List them as one string separated by commas, e.g. "L0,L1". Choices: intuition, L0, L1, ALASSO, SCAD, competitors, bullying')
parser.add_argument('--models', type=str, default='PGD,MSR3,MSR3-fast', help="Which models to include to trials")
parser.add_argument('--trials_from', type=int, default=1, help='Each "trial" represents testing all algorithms listed in "experiments" (except intuition and bullying) on one synthetic problem. This parameter and trials_to define bounds. E.g. trials_from=1 (inclusive) and trials_to=5 (exclusive) means that all algorithms will be tested on 4 problems.')
parser.add_argument('--trials_to', type=int, default=2, help='Each "trial" represents testing all algorithms listed in "experiments" (except intuition and bullying) on one synthetic problem. This parameter and trials_to define bounds. E.g. trials_from=1 (inclusive) and trials_to=5 (exclusive) means that all algorithms will be tested on 4 problems.')
parser.add_argument('--use_dask', type=bool, default=True, help='Whether to use Dask Distributed to parallelize experimens. Highly recommended.')
parser.add_argument('--n_dask_workers', type=int, default=max(cpu_count() - 1, 1), help='Number of Dask workers. Defaults to the number of your CPUs - 1.')
parser.add_argument('--random_seed', type=int, default=0, help='Experiments-wide random seed.')
parser.add_argument('--base_folder', type=str, default=".", help='Path to the base folder (where this file is).')
parser.add_argument('--experiment_name', type=str, default="test_run", help='Name for this run. This script will create a folder named "results/experiment_name" where it will put all outputs of the experiments.')
parser.add_argument('--add_timestamp', type=bool, default=False, help='Whether to add timestamp to experiment_name. Prevents overwriting your previous outputs when launching this script more than once.')
parser.add_argument('--worker_number', type=int, default=1, help='[For SLURM environment] Then number of this worker in SLURM array. Do not confuse it with n_dask_workers: the former is for parallelizing the trials over multiple nodes (e.g. on a SLURM cluster), the latter is for parallelizing experiments for one trial within one node.')
parser.add_argument('--draw_benchmark_data', type=bool, default=True, help='Whether to produce the plots and benchmark tables after the experiments are done executing. Must be False when using SLURM.')
parser.add_argument('--verbose', type=bool, default=True, help="Whether to print log and progress messages or execute silently.")

# problems settings
parser.add_argument('--num_covariates', type=int, default=19, help="Number of covariates for synthetic problems")
parser.add_argument('--correlation_between_adjacent_covariates', type=float, default=0.0, help="Correlations between adjacent pairs of covariates.")
parser.add_argument('--groups_sizes', type=str, default="10,15,4,8,3,5,18,9,6", help="Group sizes")
parser.add_argument('--true_coefs_min', type=float, default=0, help="Magnitude of the smallest coefficient")
parser.add_argument('--true_coefs_max', type=float, default=9.5, help="Magnitude of the largest coefficient")
parser.add_argument('--fit_fixed_intercept', type=bool, default=True, help="Whether to add the intercept as a fixed effect")
parser.add_argument('--fit_random_intercept', type=bool, default=True, help="Whether to add the intercept as a random effect")
parser.add_argument('--obs_var', type=float, default=0.3, help="Variance of the observations")
parser.add_argument('--distribution', type=str, choices=("normal", "uniform"), default="normal", help="Distribution for generating features (covariates)")

# models parameters
parser.add_argument('--elastic_eps', type=float, default=1e-7, help='L2 regularization coefficient to ensure convexity of the relaxed problem')
parser.add_argument('--initializer', type=str, default="None", help='How to initialize model coefficients. Options: EM - do one iteration of EM algorithm, or None -- start with all ones.')
parser.add_argument('--logger_keys', type=tuple,
                    default=('converged', 'vaida_aic', 'jones_bic', 'muller_ic', 'vaida_aic_marginalized'), help="Which quantities should the model record during training.")
parser.add_argument('--tol_oracle', type=float, default=1e-5, help="[For SR3] Tolerance for SR3 oracle's internal numerical subroutines")
parser.add_argument('--tol_solver', type=float, default=1e-4, help="Tolerance for the stop criterion of PGD solver")
parser.add_argument('--take_only_positive_part', type=float, default=True, help="[For SR3] Whether to use only the positive part of the Hessian to avoid negative Hessians.")
parser.add_argument('--take_expected_value', type=float, default=False, help="[For SR3] Whether to use the expected value of the Hessian to avoid negative Hessians.")
parser.add_argument('--max_iter_oracle', type=int, default=1000, help="[For SR3] Maximum number of iterations for the SR3 oracle's numerical subroutines.")
parser.add_argument('--max_iter_solver_pgd', type=int, default=300000, help="Maximum number of iterations for the PGD solver")
parser.add_argument('--max_iter_solver_sr3', type=int, default=10000, help="[For SR3] Maximum number of iterations for the PGD solver.")
parser.add_argument('--fixed_step_size', type=float, default=1e-4, help="Size of the fixed step-size for PGD solver")
parser.add_argument('--eta_min', type=int, default=-2, help="[For SR3] Left boundary (in Log10) for grid search over eta -- the SR3 relaxation parameter")
parser.add_argument('--eta_max', type=int, default=1, help="[For SR3] Right boundary (in Log10) for grid search over eta -- the SR3 relaxation parameter")
parser.add_argument('--eta_num_evals', type=int, default=10, help="[For SR3] Number of uniformly-sampled grid-search points for eta")
parser.add_argument('--information_criterion_for_model_selection', type=str, default='jones', help="Which information criterion to use for the final model selection. Options: jones, vaida, muller.")

args = parser.parse_args()

base_folder = Path(args.base_folder)
dataset_path = base_folder / "bullying" / "bullying_data.csv"

experiments_to_launch = set(args.experiments.split(','))
models_to_launch = set(args.models.split(','))
args.groups_sizes = tuple([int(a) for a in args.groups_sizes.split(',')])

model_parameters = {
    "elastic_eps": args.elastic_eps,
    "initializer": args.initializer,
    "logger_keys": args.logger_keys,
    "tol_oracle": args.tol_oracle,
    "tol_solver": args.tol_solver,
    "max_iter_oracle": args.max_iter_oracle,  # 10000
}

problem_parameters = {
    "groups_sizes": args.groups_sizes,
    "features_labels": [FIXED_RANDOM] * args.num_covariates,
    "fit_fixed_intercept": args.fit_fixed_intercept,
    "fit_random_intercept": args.fit_random_intercept,
    "obs_var": args.obs_var,
    "distribution": args.distribution
}

true_beta = np.array([1, 0] + [1, 0] * int((args.num_covariates - 1) / 2)) * np.linspace(args.true_coefs_min,
                                                                                         args.true_coefs_max,
                                                                                         args.num_covariates + 1)
true_gamma = true_beta.copy()


if __name__ == "__main__":
    # Add a timestamp to the folder's name if required
    if args.add_timestamp:
        experiment_name = args.experiment_name + "_" + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    else:
        experiment_name = args.experiment_name

    # set up folder structure
    experiment_folder = base_folder / "results" / f"{experiment_name}"
    logs_folder = experiment_folder / "logs"
    dask_folder = experiment_folder / f"dask_{args.worker_number}"
    data_folder = experiment_folder / f'synthetic_data'
    figures_folder = experiment_folder / "figures"
    tables_folder = experiment_folder / "tables"

    if not experiment_folder.exists():
        print(f"Creating a folder for the experiments' outputs: {experiment_folder}")
        experiment_folder.mkdir(parents=True, exist_ok=True)
        logs_folder.mkdir(parents=True, exist_ok=True)
        figures_folder.mkdir(parents=True, exist_ok=True)
        tables_folder.mkdir(parents=True, exist_ok=True)
        data_folder.mkdir(parents=True, exist_ok=True)

    # save the inputs of the experiment for reproducibility purposes
    args_pickle_name = logs_folder / f"experiment_inputs_{args.worker_number}.pickle"

    with open(args_pickle_name, 'wb') as f:
        pickle.dump(args, f)
        print(f"This run's input parameters are saved as: \n {args_pickle_name}")

    if "intuition" in experiments_to_launch:
        print("Generate data and fit models for the 'intuition' experiment (Figure 3)")
        seed = 13

        correlation_between_adjacent_covariates = 0.0
        beta_lims = (-3, 3)
        gamma_lims = (0, 3)
        grid_dim = 100

        results, now = run_intuition_experiment(
            seed=seed,
            num_covariates=2,
            model_parameters={
                "ell": 5,
            },
            problem_parameters={
                "groups_sizes": [10, 15, 4, 8, 3, 5],
                "features_labels": [FIXED_RANDOM] * 2,
                "fit_fixed_intercept": True,  # ?
                "fit_random_intercept": False,
                "obs_var": 0.1,
                "chance_missing": 0,
                "chance_outlier": 0.0,
                "outlier_multiplier": 5,
            },
            lam=1.5,
            initial_parameters={
                "beta": -np.ones(3),
                "gamma": np.array([1, 2])
            },
            beta_lims=beta_lims,
            gamma_lims=gamma_lims,
            grid_dim=grid_dim,
            logs_folder=logs_folder,
        )
        plot_intuition_picture(results,
                               beta_lims=beta_lims,
                               gamma_lims=gamma_lims,
                               grid_dim=grid_dim,
                               figures_folder=figures_folder)

    if "L0" in experiments_to_launch:
        experiment = "L0"
        print(f"\nRun comparison experiment for {experiment}-based selection benchmark (Table 2, Figure 4). "
              f"Problems to solve: [{args.trials_from}, {args.trials_to})")
        logs = None
        if logs is None:
            models = {}
            if "PGD" in models_to_launch:
                models["L0"] = lambda j, ell: L0LmeModel(**model_parameters,
                                                    stepping="fixed",
                                                    fixed_step_len=args.fixed_step_size,
                                                    max_iter_solver=args.max_iter_solver_pgd,
                                                    nnz_tbeta=j,
                                                    nnz_tgamma=j)
            if "MSR3" in models_to_launch:
                models["SR3-L0"] = lambda j, ell: L0LmeModelSR3(**model_parameters,
                                                           stepping="fixed",
                                                           nnz_tbeta=j,
                                                           nnz_tgamma=j,
                                                           max_iter_solver=args.max_iter_solver_sr3,
                                                           practical=False,
                                                           take_only_positive_part=args.take_only_positive_part,
                                                           take_expected_value=args.take_expected_value,
                                                           ell=ell
                                                           )
            if "MSR3-fast" in models_to_launch:
                models["SR3-L0-P"] = lambda j, ell: L0LmeModelSR3(**model_parameters,
                                                             stepping="fixed",
                                                             nnz_tbeta=j,
                                                             nnz_tgamma=j,
                                                             max_iter_solver=args.max_iter_solver_sr3,
                                                             practical=True,
                                                             take_only_positive_part=args.take_only_positive_part,
                                                             take_expected_value=args.take_expected_value,
                                                             ell=ell
                                                             )
            if len(models) > 0:
                logs = run_comparison_experiment(
                    args=args,
                    models=models,
                    lambda_search_bounds=(1, args.num_covariates),
                    ell_search_grid=np.logspace(args.eta_min, args.eta_max, args.eta_num_evals),
                    problem_parameters=problem_parameters,
                    tol=np.sqrt(model_parameters["tol_solver"]),
                    true_beta=true_beta,
                    true_gamma=true_gamma,
                    dask_folder=dask_folder,
                    data_folder=data_folder
                )
                logs_path = logs_folder / f"{experiment}_{args.worker_number}.log"
                logs.to_csv(logs_path)
                print(f"{experiment} log saved to: \n {logs_path}")

    if "L1" in experiments_to_launch:
        experiment = "L1"
        print(f"\nRun comparison experiment for {experiment}-based selection benchmark (Table 2, Figure 4). "
              f"Problems to solve: [{args.trials_from}, {args.trials_to})")
        logs = None
        if logs is None:
            models = {}
            if "PGD" in models_to_launch:
                models["L1"] = lambda lam, ell: L1LmeModel(**model_parameters,
                                                      stepping="fixed",
                                                      fixed_step_len=min(args.fixed_step_size / 10 ** lam,
                                                                         args.fixed_step_size),
                                                      max_iter_solver=args.max_iter_solver_pgd,
                                                      lam=10 ** lam)
            if "MSR3" in models_to_launch:
                models["SR3-L1"] = lambda lam, ell: L1LmeModelSR3(**model_parameters,
                                                             stepping="fixed",
                                                             lam=10 ** lam,
                                                             ell=ell,
                                                             max_iter_solver=args.max_iter_solver_sr3,
                                                             take_only_positive_part=args.take_only_positive_part,
                                                             take_expected_value=args.take_expected_value,
                                                             practical=False)
            if "MSR3-fast" in models_to_launch:
                models['SR3-L1-P'] = lambda lam, ell: L1LmeModelSR3(**model_parameters,
                                                               stepping="fixed",
                                                               lam=10 ** lam,
                                                               ell=ell,
                                                               max_iter_solver=args.max_iter_solver_sr3,
                                                               take_only_positive_part=args.take_only_positive_part,
                                                               take_expected_value=args.take_expected_value,
                                                               practical=True)
            if len(models) > 0:
                logs = run_comparison_experiment(
                    args=args,
                    models=models,
                    lambda_search_bounds=(-2, 4),
                    ell_search_grid=np.logspace(args.eta_min, args.eta_max, args.eta_num_evals),
                    true_beta=true_beta,
                    true_gamma=true_gamma,
                    problem_parameters=problem_parameters,
                    tol=np.sqrt(model_parameters["tol_solver"]),
                    dask_folder=dask_folder,
                    data_folder=data_folder
                )
                logs_path = logs_folder / f"{experiment}_{args.worker_number}.log"
                logs.to_csv(logs_path)
                print(f"{experiment} log saved to: \n {logs_path}")

    if "ALASSO" in experiments_to_launch:
        experiment = "ALASSO"
        print(f"\nRun comparison experiment for {experiment}-based selection benchmark (Table 2, Figure 4). "
              f"Problems to solve: [{args.trials_from}, {args.trials_to})")
        logs = None

        if logs is None:
            models = {}
            if "PGD" in models_to_launch:
                models['ALASSO'] = lambda lam, ell: L1LmeModel(**model_parameters,
                                                          stepping="fixed",
                                                          fixed_step_len=min(args.fixed_step_size / 10 ** lam,
                                                                             args.fixed_step_size),
                                                          max_iter_solver=args.max_iter_solver_pgd,
                                                          lam=10 ** lam)
            if "MSR3" in models_to_launch:
                models['SR3-ALASSO'] = lambda lam, ell: L1LmeModelSR3(**model_parameters,
                                                                 stepping="fixed",
                                                                 lam=10 ** lam,
                                                                 ell=ell,
                                                                 max_iter_solver=args.max_iter_solver_sr3,
                                                                 practical=False)
            if "MSR3-fast" in models_to_launch:
                models['SR3-ALASSO-P'] = lambda lam, ell: L1LmeModelSR3(**model_parameters,
                                                                   stepping="fixed",
                                                                   lam=10 ** lam,
                                                                   ell=ell,
                                                                   max_iter_solver=args.max_iter_solver_sr3,
                                                                   take_only_positive_part=args.take_only_positive_part,
                                                                   take_expected_value=args.take_expected_value,
                                                                   practical=True)
            if len(models) > 0:
                logs = run_comparison_experiment(
                    args=args,
                    models=models,
                    non_regularized_model=SimpleLMEModel(**model_parameters,
                                                         initial_parameters={
                                                             "beta": 1 / 2 * np.ones(args.num_covariates + 1),
                                                             "gamma": 1 / 2 * np.ones(args.num_covariates + 1)
                                                         },
                                                         stepping="line-search"),
                    lambda_search_bounds=(-2, 4),
                    ell_search_grid=np.logspace(args.eta_min, args.eta_max, args.eta_num_evals),
                    true_beta=true_beta,
                    true_gamma=true_gamma,
                    problem_parameters=problem_parameters,
                    tol=np.sqrt(model_parameters["tol_solver"]),
                    dask_folder=dask_folder,
                    data_folder=data_folder
                )
                logs_path = logs_folder / f"{experiment}_{args.worker_number}.log"
                logs.to_csv(logs_path)
                print(f"{experiment} log saved to: \n {logs_path}")

    if "SCAD" in experiments_to_launch:
        experiment = "SCAD"
        print(f"\nRun comparison experiment for {experiment}-based selection benchmark (Table 2, Figure 4). "
              f"Problems to solve: [{args.trials_from}, {args.trials_to})")
        model_parameters_copy = model_parameters.copy()
        model_parameters_copy["sigma"] = 0.3
        model_parameters_copy["rho"] = 1.6
        logs = None
        if logs is None:
            models = {}
            if "PGD" in models_to_launch:
                models['SCAD'] = lambda lam, ell: SCADLmeModel(**model_parameters_copy,
                                                          stepping="fixed",
                                                          fixed_step_len=min(args.fixed_step_size / 10 ** lam,
                                                                             args.fixed_step_size),
                                                          max_iter_solver=args.max_iter_solver_pgd,
                                                          lam=10 ** lam)
            if "MSR3" in models_to_launch:
                models['MSR3'] = lambda lam, ell: SCADLmeModelSR3(**model_parameters_copy,
                                                                 stepping="fixed",
                                                                 lam=10 ** lam,
                                                                 ell=ell,
                                                                 max_iter_solver=args.max_iter_solver_sr3,
                                                                 take_only_positive_part=args.take_only_positive_part,
                                                                 take_expected_value=args.take_expected_value,
                                                                 practical=False)
            if "MSR3-fast" in models_to_launch:
                models['SR3-SCAD-P'] = lambda lam, ell: SCADLmeModelSR3(**model_parameters_copy,
                                                                   stepping="fixed",
                                                                   lam=10 ** lam,
                                                                   ell=ell,
                                                                   max_iter_solver=args.max_iter_solver_sr3,
                                                                   take_only_positive_part=args.take_only_positive_part,
                                                                   take_expected_value=args.take_expected_value,
                                                                   practical=True)

            logs = run_comparison_experiment(
                args=args,
                models=models,
                lambda_search_bounds=(-2, 4),
                true_beta=true_beta,
                true_gamma=true_gamma,
                ell_search_grid=np.logspace(args.eta_min, args.eta_max, args.eta_num_evals),
                problem_parameters=problem_parameters,
                tol=np.sqrt(model_parameters["tol_solver"]),
                dask_folder=dask_folder,
                data_folder=data_folder
            )
            logs_path = logs_folder / f"{experiment}_{args.worker_number}.log"
            logs.to_csv(logs_path)
            print(f"{experiment} log saved to: \n {logs_path}")

    if "competitors" in experiments_to_launch:
        experiment = "competitors"
        print(f"\n Run experiment for alternative libraries on L1 tasks (Table 3). Problems to solve [{args.trials_from}, {args.trials_to})")
        logs = None
        # logs = pd.read_csv(logs_folder / "log_l1_experiment_2021-11-03 21:19:20.113759.csv")
        if logs is None:
            # Enable these two if you want to test PCO and Fence methods as well
            # from pco_wrapper import PCOModel
            # from fence_wrapper import FenceModel
            # See the comments below on why we don't include them in the comparison.
            logs = run_comparison_experiment(
                args=args,
                models={
                    "SR3-L1-P": lambda lam, ell: L1LmeModelSR3(**model_parameters,
                                                               stepping="fixed",
                                                               lam=10 ** lam,
                                                               ell=ell,
                                                               max_iter_solver=args.max_iter_solver_sr3,
                                                               take_only_positive_part=args.take_only_positive_part,
                                                               take_expected_value=args.take_expected_value,
                                                               practical=True),
                    "glmmLasso": lambda lam, ell: GMMLassoModel(**model_parameters,
                                                                lam=10 ** lam),
                    "lmmLasso": lambda lam, ell: lmmlassomodel(**model_parameters,
                                                               lam=10 ** lam),
                    # PCO can not solve problems where the number of random effects
                    # (|gamma| * num_groups) exceeds the total number of objects.
                    # "pco": lambda lam, ell: PCOModel(**model_parameters,
                    #                                  lam=10 ** lam),
                    # Fence is too slow and runs out of memory on Macbook Pro 16.
                    # "fence": lambda lam, ell: FenceModel(**model_parameters,
                    #                                      lam=10 ** lam)
                },
                lambda_search_bounds=(-2, 4),
                ell_search_grid=np.logspace(args.eta_min, args.eta_max, args.eta_num_evals),
                true_beta=true_beta,
                true_gamma=true_gamma,
                problem_parameters=problem_parameters,
                tol=np.sqrt(model_parameters["tol_solver"]),
                dask_folder=dask_folder,
                data_folder=data_folder
            )
            logs_path = logs_folder / f"{experiment}_{args.worker_number}.log"
            logs.to_csv(logs_path)
            print(f"Measurements for {experiment} are saved to: \n {logs_path}")
            generate_competitors_table(logs_path, tables_folder)

    if "bullying" in experiments_to_launch:
        print("Run a feature-selection experiment on real-world data from the Bullying study. (Figure 6)")
        generate_bullying_experiment(dataset_path=dataset_path,
                                     figures_directory=figures_folder)

    if args.draw_benchmark_data:
        experiments_to_plot = set(experiments_to_launch) & {"L0", "L1", "ALASSO", "SCAD"}
        if len(experiments_to_plot) > 0:
            generate_benchmark_table({
                "experiments": experiments_to_plot,
                "experiment_folder": experiment_folder,
                "ic": args.information_criterion_for_model_selection
            })
