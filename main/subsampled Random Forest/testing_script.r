#! /home/ec2-user/anaconda3/envs/R/bin/Rscript

source("web_loader.r")


# User Input
folds <- 20
validation_grid_size <- 100
alpha <- .05

# Iterate over all datasets
for (dataset_name in names(web_loader)){
    command_string <- paste0("./training.r ",
                             dataset_name," ",
                             as.character(folds)," ",
                             as.character(validation_grid_size)," ",
                             as.character(alpha)
                        )

    system(command=command_string, 
       intern=TRUE, ignore.stdout=FALSE, ignore.stderr=FALSE
       )    
}
