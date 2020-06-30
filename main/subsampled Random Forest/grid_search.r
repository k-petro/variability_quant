grid_search <- function(len, x_train, y_train, x_val, y_val){
    
    # Initialize paramters
    best_ll <- -Inf
    best_B <- 0
    best_ntree <- 0

    # Construct grid
    grid <- subGrid(len=len,search="not_grid")

    # Grid Search
    for (row in 1:nrow(grid)){
        B<- grid[row,"B"][[1]]
        ntree <- grid[row,"ntree"][[1]]

        # Train
        model <- forest(
            x_train, y_train, 
            var.type="ustat", 
            B= B, 
            ntree= ntree, 
            replace=FALSE, individualTrees=TRUE
        )

        # Predict
        val_results <- forest.varU(
            predict(model, x_val, individualTrees=TRUE)$predictedAll,
            model
        ) %>% mutate(LL= pointLL(y.hat, var.hat, y_val))

        # Evaluate on Log-Likelihood
        ll <- sum(val_results$LL)

        if (is.na(ll)) next
        if (ll>best_ll){
            
#             print("Updating optimum solution:")
#             print(paste0("Log-Likelihood:", ll))
#             print(paste0("B:", B))
#             print(paste0("ntree:", ntree))
#             cat("\n")
            
            best_ll <- ll
            best_B <- B
            best_ntree <- ntree
        }
    }
    
    # Done
    return( list("ll"= best_ll, "B"= best_B, "ntree"= best_ntree) )
}