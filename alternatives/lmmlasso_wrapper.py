import numpy as np
from pandas import DataFrame

from pysr3.lme.problems import LMEProblem
from pysr3.lme.oracles import LinearLMEOracle
from pysr3.logger import Logger


class lmmlassomodel:
    def __init__(self, lam, **kwargs):
        self.lam = lam
        self.logger_ = Logger(list_of_keys=set(
            ['converged', 'iteration', 'muller_ic', 'jones_bic', 'vaida_aic', 'vaida_aic_marginalized']))

    def fit_problem(self, problem: LMEProblem, **kwargs):

        import rpy2.rinterface_lib.embedded
        import rpy2.robjects as robjects
        from rpy2.robjects import numpy2ri, pandas2ri
        r = robjects.r
        r['source']('lmmlasso.r')
        get_lmmlasso_estimate = robjects.globalenv['get_lmmlasso_function']

        objects_groups = np.repeat(problem.group_labels, problem.groups_sizes)
        numpy2ri.activate()
        try:
            result = get_lmmlasso_estimate(np.vstack(problem.fixed_features),
                                           np.concatenate(problem.answers).reshape(-1, 1),
                                           np.vstack(problem.random_features),
                                           objects_groups,
                                           self.lam,
                                           )
        except rpy2.rinterface_lib.embedded.RRuntimeError as e:
            # the method failed to converge
            result = (
                np.zeros(problem.num_fixed_features),
                np.zeros((problem.num_random_features, problem.num_random_features)),
                [1e8],
                [1e8],
                np.concatenate(problem.answers)
            )
            print(e)

        beta = result[0]
        gamma = np.diag(result[1])
        aic = result[2][0]
        bic = result[3][0]
        y_pred = result[4]

        pandas2ri.deactivate()
        numpy2ri.deactivate()

        self.coef_ = {
            "beta": beta,
            "gamma": gamma
        }
        self.logger_.add(key='iteration', value=0)
        numpy2ri.deactivate()
        oracle = LinearLMEOracle(problem)
        self.logger_.add(key='converged', value=True)
        self.logger_.add(key="muller_ic", value=oracle.muller_hui_2016ic(**self.coef_))
        self.logger_.add(key="jones_bic", value=bic)
        self.logger_.add(key="vaida_aic",
                         value=aic)  # need to re-work ddf function for non-diagonal gamma
        self.logger_.add(key="vaida_aic_marginalized", value=oracle.vaida2005aic(**self.coef_, marginalized=True))
        self.y_pred = y_pred
        return self

    def predict_problem(self, problem):
        return self.y_pred


if __name__ == "__main__":
    import pickle

    import rpy2.rinterface_lib.embedded
    import rpy2.robjects as robjects
    from rpy2.robjects import numpy2ri, pandas2ri

    r = robjects.r
    r['source']('lmmlasso.r')
    get_lmmlasso_estimate = robjects.globalenv['get_lmmlasso_function']

    with open('sample_problem.pickle', 'rb') as f:
        problem, true_model_parameters = pickle.load(f)

    objects_groups = np.repeat(problem.group_labels, problem.groups_sizes)

    pandas2ri.activate()
    numpy2ri.activate()
    result = get_lmmlasso_estimate(np.vstack(problem.fixed_features),
                                   np.concatenate(problem.answers).reshape(-1, 1),
                                   np.vstack(problem.random_features),
                                   objects_groups,
                                   10,
                                   )
    beta = result[0]
    gamma = np.diag(result[1])
    aic = result[2]
    bic = result[3]
    y_pred = result[4]
    pandas2ri.deactivate()
    numpy2ri.deactivate()

    coef_ = {
        "beta": beta,
        "gamma": gamma
    }

    pass
