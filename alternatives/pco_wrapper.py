import numpy as np
from pandas import DataFrame

from pysr3.lme.problems import LMEProblem
from pysr3.lme.oracles import LinearLMEOracle
from pysr3.logger import Logger


class PCOModel:
    def __init__(self, lam, **kwargs):
        self.lam = lam
        self.logger_ = Logger(list_of_keys=set(
            ['converged', 'iteration', 'muller_ic', 'jones_bic', 'vaida_aic', 'vaida_aic_marginalized']))

    def fit_problem(self, problem: LMEProblem, **kwargs):

        import rpy2.rinterface_lib.embedded
        import rpy2.robjects as robjects
        from rpy2.robjects import numpy2ri, pandas2ri
        r = robjects.r
        r['source']('alternatives/pco.r')
        get_pco_estimate = robjects.globalenv['get_pco_estimate']

        objects_groups = np.repeat(problem.group_labels, problem.groups_sizes)

        x, y, _ = problem.to_x_y()

        numpy2ri.activate()
        try:
            result = get_pco_estimate([features[:, 1:] for features in problem.fixed_features],
                                      [answer.reshape(-1, 1) for answer in problem.answers],
                                      problem.random_features,
                                      objects_groups,
                                      problem.num_groups,
                                      np.mean([obs_var.mean() for obs_var in problem.obs_vars]),
                                      self.lam,
                                      42
                                      )
        except rpy2.rinterface_lib.embedded.RRuntimeError as e:
            # the method failed to converge
            result = (
                np.zeros(problem.num_fixed_features),
                np.zeros((problem.num_random_features, problem.num_random_features)),
                [0]
            )
            print(e)

        beta = result[0] # np.pad(result[0].reshape(-1), (1, 0), mode='constant')
        gamma = np.diag(result[1])
        sigma = result[2]
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
        self.logger_.add(key="jones_bic", value=oracle.jones2010bic(**self.coef_))
        self.logger_.add(key="vaida_aic", value=oracle.vaida2005aic(**self.coef_))     # need to re-work ddf function for non-diagonal gamma
        self.logger_.add(key="vaida_aic_marginalized", value=oracle.vaida2005aic(**self.coef_, marginalized=True))
        self.y_pred = np.zeros(*y.shape)
        return self

    def predict_problem(self, problem):
        return self.y_pred


if __name__ == "__main__":
    import pickle

    import rpy2.rinterface_lib.embedded
    import rpy2.robjects as robjects
    from rpy2.robjects import numpy2ri, pandas2ri

    r = robjects.r
    r['source']('pco.r')
    get_pco_estimate = robjects.globalenv['get_pco_estimate']

    with open('sample_problem_x4.pickle', 'rb') as f:
        problem, true_model_parameters = pickle.load(f)


    objects_groups = np.repeat(problem.group_labels, problem.groups_sizes)

    pandas2ri.activate()
    numpy2ri.activate()
    res = get_pco_estimate([features[:, 1:] for features in problem.fixed_features],
                           [answer.reshape(-1, 1) for answer in problem.answers],
                           problem.random_features,
                           objects_groups,
                           problem.num_groups,
                           np.mean([obs_var.mean() for obs_var in problem.obs_vars]),
                           2,
                           42
                           )
    beta = np.pad(res[0].reshape(-1), (1, 0), mode='constant')
    gamma = np.diag(res[1])
    sigma = res[2]
    pandas2ri.deactivate()
    numpy2ri.deactivate()

    coef_ = {
        "beta": beta,
        "gamma": gamma
    }

    oracle = LinearLMEOracle(problem)

    muller_ic = oracle.muller_hui_2016ic(**coef_)
    jones_bic = oracle.jones2010bic(**coef_)
    vaida_aic = oracle.vaida2005aic(**coef_, marginalized=True)
    y_pred = np.zeros(*problem.answers.shape)
    pass
