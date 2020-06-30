#! /home/ec2-user/anaconda3/envs/R/bin/Rscript

############ DEPENDENCIES ############

# Load external files
ext_files <- c("web_loader.r", "subRF_method.r", "grid_search.r")
for (file in ext_files) source(file)

# Check packages availability
packages <- list("caret","devtools","randomForestCI", "surfin", "grf", "Metrics", "tidyverse")
    # library(rpart) # for kyphosis data
    # library(MASS)
installed <- installed.packages()

# Check if installed and install
for (package in packages){
    if(package %in% rownames(installed) == FALSE){
        install.packages( package )
    }
}

# Load Libraries
for (package in packages) suppressMessages(library(package, character.only = TRUE))

# Custom Function
pointLL <- function(mean, sigma2, datapoint) {
    # Gaussian Log-Likelihood function for single-point
    -.5*log(2*pi*sigma2) - .5*( datapoint - mean)**2/sigma2 
}


############ INPUT ############
args <- commandArgs(trailingOnly=TRUE)

dataset_name <- args[1]
folds <- as.numeric(args[2])
grid_search_len <- as.numeric(args[3])
alpha <- as.numeric(args[4])

cat(dataset_name, folds, grid_search_len, alpha)

# # Print dataset selection
#cat("Available datasets\n")
#cat(names(web_loader),"\n")
#cat("Choose dataset:\t")
#dataset_name <- readLines(con='stdin', n=1, warn=FALSE)

# # folds
#cat("Choose number of folds for testing:\t")
#folds <-as.numeric( readLines(con='stdin', n=1, warn=FALSE))

# # grid search size
#cat("Choose sample size for the randomized grid search:\t")
#grid_search_len <- as.numeric(readLines(con='stdin', n=1, warn=FALSE))

# # confidence level
#cat("Choose the value of the parameter alpha for the confidence intervals:\t")
#alpha <- as.numeric(readLines(con='stdin', n=1, warn=FALSE))


start.time <- Sys.time()
cat("\nStarted.\n")

############ DATA LOAD ############

# Load external dataset
dataset <- web_loader[dataset_name][[1]]()

# Initialize results variable
testing_results_complete <- tibble(
    coverage= numeric(),
    avg_range= numeric(),
    rmse= numeric(),
    ll= numeric()
    )

############ MAIN LOOP ############

# Create k folds of the data to test and average results
# print(folds)
# print(length(dataset[,ncol(dataset)]))
folds <- createFolds(dataset[,ncol(dataset):ncol(dataset)], k = folds, list = TRUE, returnTrain = FALSE)

counter<-1
for (fold in folds){
    cat("fold ",counter," started.\n\n")
    
############ DATASET SPLIT ############
    
    # Split data in train, val, test
    x_test<- dataset[fold,1:(ncol(dataset)-1)]
    y_test<- dataset[fold,ncol(dataset):ncol(dataset)]

    x_train_all<- dataset[-fold,1:(ncol(dataset)-1)]
    y_train_all<- dataset[-fold, ncol(dataset):ncol(dataset), drop=FALSE]

    train_index<- createDataPartition( y_train_all[[1]], p=0.8, list=FALSE)

    x_train<- x_train_all[train_index,]
    y_train<- y_train_all[train_index,]

    x_val<- x_train_all[-train_index,]
    y_val<- y_train_all[-train_index,]

############ TRAIN  ############

    # Obtain optimum hyperparamteres with randomized Grid Search
    grid_results <- grid_search(len=grid_search_len, x_train, y_train, x_val, y_val)

    # Train model
    model = forest(x_train, y_train, 
                   var.type="ustat", 
                   B= grid_results$B, ntree= grid_results$ntree, 
                   replace=FALSE, individualTrees=TRUE
                  )

    # Predict
    predictions = predict(model, x_test, individualTrees=TRUE)
    results = forest.varU(predictions$predictedAll,model)

    # Add columns to results
    results <- results %>% 
        bind_cols( tibble(y_test) ) %>% 
        mutate(confInt_low= qnorm(alpha/2, y.hat, sqrt(var.hat))) %>%
        mutate(confInt_up= qnorm(1-alpha/2, y.hat, sqrt(var.hat))) %>% 
        mutate(on_target= if_else( y_test>=confInt_low & y_test<=confInt_up, 1, 0 ) ) %>% 
        mutate(range= abs(confInt_up-confInt_low) ) %>% 
        mutate(LL= pointLL(y.hat, var.hat, y_test))
    
############ METRICS ############
    
    # Results for the fold
    testing_results <- tibble(
        # Confidence Interval Metrics
        "coverage"= sum( results$on_target ) / length(y_test) *100,
        "avg_range"= mean(results$range),
        # Model Metrics
        "rmse"= rmse(y_test, predictions$predicted),
        "ll"= sum(results$LL)
    )

    # Update results for the dataset
    testing_results_complete <- bind_rows( testing_results_complete, testing_results)    
    counter <- counter +1
}

############ OUTPUT ############

# Log average performance across folds
write.table( c("averages"=sapply( testing_results_complete, mean)), file = paste0("./results/",dataset_name,"_avg.txt"))
# Log average performance across folds
write.table( testing_results_complete, file = paste0("./results/",dataset_name,"_all.txt")) #, col.names= TRUE)


finish.time <- Sys.time()
cat(dataset_name," dataset ran for: ")
print(finish.time-start.time)
cat("Results written in folder.\n\n")

#Done.
