import numpy as np
from pandas import DataFrame

from pysr3.lme.problems import LMEProblem
from pysr3.lme.oracles import LinearLMEOracle
from pysr3.logger import Logger


class GMMLassoModel:
    def __init__(self, lam, **kwargs):
        self.lam = lam
        self.logger_ = Logger(list_of_keys=set(
            ['converged', 'iteration', 'muller_ic', 'jones_bic', 'vaida_aic', 'vaida_aic_marginalized']))

    def fit_problem(self, problem: LMEProblem, **kwargs):

        import rpy2.rinterface_lib.embedded
        import rpy2.robjects as robjects
        from rpy2.robjects import numpy2ri, pandas2ri
        r = robjects.r
        r['source']('glmmlasso.r')
        get_glmmlasso_fit = robjects.globalenv['get_glmmlasso_function']

        x, y, columns_labels = problem.to_x_y()
        columns_labels = columns_labels[:]
        j = 0
        for i, e in enumerate(columns_labels):
            if e == "fixed+random":
                columns_labels[i] = f"f{j}"
                j += 1
        df = DataFrame(data=x, columns=columns_labels)
        df["target"] = y
        all_features = '+'.join([f'f{i}' for i in range(problem.num_fixed_features - 1)])
        numpy2ri.activate()
        try:
            pandas2ri.activate()
            result = get_glmmlasso_fit(df,
                                       f"target~{all_features}",
                                       f"~1+{all_features}",
                                       self.lam)
            pandas2ri.deactivate()
        except rpy2.rinterface_lib.embedded.RRuntimeError as e:
            # the method failed to converge
            result = (
                np.zeros(problem.num_fixed_features),
                np.zeros((problem.num_random_features, problem.num_random_features)),
                np.ones(y.shape),
                [0],
                [[1e8]]
            )
            print(e)

        self.coef_ = {
            "beta": result[0],
            "gamma": np.diag(result[1])
        }
        self.y_pred = result[2]
        self.logger_.add(key='iteration', value=result[3][0])
        self.bic = result[4][0][0]
        numpy2ri.deactivate()
        oracle = LinearLMEOracle(problem)
        self.logger_.add(key='converged', value=True)
        self.logger_.add(key="muller_ic", value=oracle.muller_hui_2016ic(**self.coef_))
        self.logger_.add(key="jones_bic", value=self.bic)
        self.logger_.add(key="vaida_aic", value=oracle.vaida2005aic(**self.coef_))
        self.logger_.add(key="vaida_aic_marginalized", value=oracle.vaida2005aic(**self.coef_, marginalized=True))
        return self

    def predict_problem(self, problem):
        return self.y_pred


if __name__ == "__main__":
    import pickle

    import rpy2.rinterface_lib.embedded
    import rpy2.robjects as robjects
    from rpy2.robjects import numpy2ri, pandas2ri

    r = robjects.r
    r['source']('glmmlasso.r')
    get_glmmlasso_fit = robjects.globalenv['get_glmmlasso_function']

    with open('sample_problem.pickle', 'rb') as f:
        problem, true_model_parameters = pickle.load(f)
    x, y, columns_labels = problem.to_x_y()
    from pandas import DataFrame

    j = 0
    for i, e in enumerate(columns_labels):
        if e == "fixed+random":
            columns_labels[i] = f"f{j}"
            j += 1
    df = DataFrame(data=x, columns=columns_labels)
    df["target"] = y
    all_features = '+'.join([f'f{i}' for i in range(problem.num_fixed_features - 1)])
    pandas2ri.activate()
    numpy2ri.activate()
    res = get_glmmlasso_fit(df,
                            f"target~{all_features}",
                            f"~1+{all_features}",
                            300)
    beta = res[0]
    gamma = np.diag(res[1])
    y_pred = res[2]
    iterations = res[3]
    pandas2ri.deactivate()
    numpy2ri.deactivate()
    pass
