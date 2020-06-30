#REMEMBER: Check that response is column "-1"

""" Dictionary with dataloads from urls available in UCI Datasets. """

import pandas as pd
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen

def _load_gas(): 
    zipfile = ZipFile( BytesIO(
        urlopen("https://archive.ics.uci.edu/ml/machine-learning-databases/00487/gas-sensor-array-temperature-modulation.zip").read()
    ) )
    df= pd.concat(
            [pd.read_csv( zipfile.open(file.filename) )
             for file in zipfile.filelist if file.filename[-4:]==".csv"]
    )
    return( df.iloc[:,slice(df.shape[1],0,-1)] )



# Selection of available datasets
datasets= ("housing","concrete","wine","energy","yacht","gas_sensor")

# Loader as a Dictionary with loading method
web_dataloader = {"datasets":datasets,
        datasets[0]: lambda: pd.read_csv(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
            header=None,
            delim_whitespace=True,
        ),
        datasets[1]: lambda: pd.read_excel(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
        ),
        datasets[2]: lambda: pd.read_csv(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
            delimiter=";",
        ),
        datasets[3]: lambda: pd.read_excel(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
        ).iloc[:, :-1],
        datasets[4]: lambda: pd.read_csv(
            "http://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data",
            header=None,
            delim_whitespace=True,
        ),
        datasets[5]: lambda: _load_gas()                  
    }

#source: https://github.com/stanfordmlgroup/ngboost/blob/master/examples/experiments/regression_exp.py