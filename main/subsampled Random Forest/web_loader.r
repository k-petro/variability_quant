#' List with functions to load data.

packages <- list("dplyr","rio","data.table")
installed <- installed.packages()

# Check if installed and install
for (package in packages){
    if(package %in% rownames(installed) == FALSE){
        install.packages("/home/ec2-user/master.tar", 
                         repo=NULL, type="source"
                        )
    }
}
 
# Load Libraries
suppressMessages(library("rio"))
suppressMessages(library("dplyr"))
suppressMessages(library("data.table"))

web_loader <- list( 
    housing = function() read.csv(
        url("https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"), 
        sep= "", header= FALSE
    ),
    concrete= function() import(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
    ),     
    wine= function() read.csv(
        url("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"), 
        sep= ";", header= TRUE
    ),
    energy= function() import(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
    ) %>% select( c(1:9) ),
    yacht= function() read.csv(
        url("http://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data"), 
        sep= "", header= FALSE
    ),
    gas_sensor= function(){
        temp <- tempfile()
        download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/00487/gas-sensor-array-temperature-modulation.zip",temp)
        zipped_csv_names <- grep('\\.csv$', unzip(temp, list=TRUE)$Name, ignore.case=TRUE, value=TRUE)
        unzip(temp, files=zipped_csv_names)
        comb_tbl <- bind_rows( 
            lapply(
                zipped_csv_names,
                function(x){
                    fread(x, sep=',', header=TRUE,stringsAsFactors=FALSE)
                    }
                )
            )
        system(command="rm *.csv", intern=TRUE) 
        comb_tbl <- comb_tbl %>% select( c(3:ncol(comb_tbl),2) )
#         return( comb_tbl )
    }
)