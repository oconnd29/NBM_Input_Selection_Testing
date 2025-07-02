## Overview

These functions and scripts are designed to process, clean, and prepare wind turbine data—originally developed for datasets from the **Kelmarsh** and **Penmanshiel** Wind Farms.

To apply these methods to a new dataset, you can:
- Directly download the Kelmarsh and Penmanshiel Datasets from Zonodo and rund the scripts as is.
  - KELMARSH:      https://zenodo.org/records/5841834
  - PENMANSHIEL:   https://zenodo.org/records/8253010
- Use the core utility functions inside the `Utils` folder.
- Or adapt the provided notebooks by adjusting variable names, turbine identifiers, and dataset structure to match your new source.

### Notebooks

#### `0_prep_Kelmarsh_or_Penmanshiel.ipynb`
Prepares raw SCADA data for analysis:
- Unzips and loads raw files.
- Loops through turbines and applies initial cleaning.
- Corrects timestamps and concatenates multi-year data.
- Returns a dictionary per farm, with corrected DataFrames for each turbine.

#### `1_createHealthyDatasets.ipynb`
Creates a clean datasets of “healthy” turbine operations:
- Loads SCADA and Alarm data.
- Keeps only mean-value features.
- Defines 6 month windows of data 
- Filters out:
  - Critical failure periods.
  - Maintenance windows.
  - High-NaN targets or columns.
- Splits datasets into consecutive Train / Validation / Test sets - 4 months/2 months/1 month
- Saves results as per-farm dictionaries of healthy datasets, the keys follow naming convention: 'TXX_DSXX'. 

#### `2_RunFS.ipynb`
Runs **Feature Selection (FS)** on the healthy datasets:
- Applies multiple FS techniques (correlation, mutual information, decision tree weights and estimated shapley valeus of an FFNN).
- Outputs ranked feature lists for each turbine dataset.

#### `3_train_prediction_models.ipynb`
Trains predictive models:
- Loads previously ordered features and healthy datasets.
- Sets up modeling configurations and hyperparameter search.
- Trains multiple machine learning models (LSTM, CNN, FFNN).
- uses validation set to test each set of hyperparamas. the results for all are stored.
- Saves the metrics on the valid set and the model state dict for each model.

#### `4_plot_best_hp_set.ipynb`
Visualizes model performance:
- goes through validation results and finds the hp set that performed best
- Loads the best hyperparameter configurations.
  Plots a heatmap of MSE (or any available metric) for chosen datasets across feature selection method
- Plots predictions vs actuals for a single (farm, dataset, subsets, feature selector)

### Folder: `Utils/`

This folder includes all reusable utility functions:
- Data cleaning tools
- Time windowing utilities
- Feature selection implementations
- Model evaluation helpers
- Plotting and logging functions

You can directly import from `Utils` in custom scripts or adapt them to other datasets and domains.
