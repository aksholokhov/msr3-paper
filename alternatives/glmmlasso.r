get_glmmlasso_function <- function(data, fixed_formula, random_formula, lambda){
  library(Matrix)
  library(lme4)
  library(glmmLasso)
  fit <- glmmLasso(fix=as.formula(fixed_formula), rnd = list(group=as.formula(random_formula)), data, lambda, family = gaussian(link="identity"),
          switch.NR=FALSE, final.re=FALSE, control = list())
  summary(fit)
  beta <- coefficients(fit)
  gamma <- fit$StdDev
  y_pred <- fitted.values(fit)
  iterations <- fit$conv.step
  bic <- fit$bic
  mylist <- list("beta" = beta, "gamma" = gamma, "y_pred" = y_pred, "iterations" = iterations, "bic" = bic)
  return(mylist)
}

  

