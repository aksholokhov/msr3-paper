get_fence_function <- function(data, formula){
    require(fence)
    library(snow)
    result <- fence.lmer(as.formula(formula), data)
    sel_model <-result$sel_model
    ss <- summary(sel_model)
    mylist <- list("beta" = sel_model@beta, "gamma" = ss$varcor$group)
    return(mylist)
}

# require(fence)
# library(snow)
# #### Example 1 #####
# data(iris)
# full = Sepal.Length ~ Sepal.Width + Petal.Length + Petal.Width + (1 + Petal.Width|Species)
# res <- get_fence_function(iris, full)
