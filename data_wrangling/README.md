# Data Wrangling

Pre-processing and wrangling the historical data is done through two steps which uses two different scripts: bash script `CSVCombiner.sh` and jupyter notebook `Files_Aggregator.ipynb`. The compressed raw data is in the `Raw/` directory in forms of `tar.xz` (doesn't necessarily have to use this version, `.zip` version in the cluster imz044 also works). Xzip provides the highest compression ratio for faster download/upload across machines while still providing fast uncompression rates.

## Steps

To use the scripts:
1. Uncompress data and file directory management
2. Clean and aggregate a particular stock across different time / folder
3. Aggregate and serialise cleaned stock(s) data

## Data Uncompression

In \*NIX OS, the uncompression of the data can be done through following commands
```bash
$ cd [project_root_directory]/data/Raw/
$ unxz -k 2015_Data.tar.xz 2016_Data.tar.xz 2017_Data.tar.xz
```

Make sure that the uncompressed data folder named `2015_Data/`, `2016_Data/`, and `2017_Data/`

## Stock Data Aggregation
In \*NIX OS, this step can be done by running the `CSVCombiner.sh` through `$ bash CSVCombiner.sh` from this directory. 2 inputs are required; the first one is the company code (i.e. HK0700 for Tencent), and the second is extended company code which is basically the same as company code but having extra '0' following HK.

If the combined file for the specified stock already present in the `[project_root_directory]/data/.staging` directory, the script won't create a new one. Delete the existing file in the staging directory when new combined file is desired.

The result of the script will be in the staging directory, having the file name of `{company code}_cleaned.csv`.

## Stock(s) Historical Data Serialisation and Aggregation
For the model to use the data, it is placed in a numpy's Panel for the ease of access of 3D matrix. The data is also serialised into a pickle file as it provides quick and easy serialisation, especially when pausing and resuming the long-running script such as training the model.

`Files_Aggregator.ipynb` is used to achieve the file serialisation and to combine multiple stocks into a single file, for which is being used by the model later. Refer to the documentation on the jupyter notebook for further explanation of how the script works.
