# Custom Training Method

library("surfin")

prm <- tibble(parameter = c("B", "ntree"),
                  class = rep("numeric", 2),
                  label = c("#Common observations among trees", "Number of Trees"))


# The "grid" search does not work correct for this model yet, due to the association between ntree ~ B
subGrid <- function (x, y, len = NULL, search = "grid") 
{
    # "ntree" is the mapping of values of "B" to the interval [100,20_000] (Need to retain linear association)
    if (search == "grid") {
        out <- expand.grid(
            B = sample(seq(10,50,2), replace=FALSE, size=len),
            ntree = B*floor(runif(len, min=100/B, max=10000/B)) #unique(c(seq(1,10,2) %o% 10^(2:3))),
        )
    }
    else {
        out <- tibble(
            B = sample(seq(10,50,2), replace=TRUE, size=len),
            ntree = B*floor(runif(len, min=100/B, max=10000/B))
        )
    }
    out
}


subRF_fit <- function(x, y, wts, param, lev, last, weights, classProbs, ...) { 
    surfin::forest(
        x = x, 
        y = y,
        var.type="ustat",
        ntree= param$ntree,
        B= param$B,
        replace=FALSE, 
        individualTrees=TRUE,
        ...
        )
 }

subRF_predict <- function(modelFit, newdata, preProc = NULL, submodels = NULL){
    surfin::predict(modelFit, newdata, individualTrees=T)
}


method_subRF <- list( type="Regression", library= "surfin", prob=NULL)
method_subRF$parameters <- prm
method_subRF$grid <- subGrid
method_subRF$fit <- subRF_fit
method_subRF$predict <- subRF_predict