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
- Returns a dictionary per farm, with cleaned DataFrames for each turbine.

#### `1_createHealthyDatasets.ipynb`
Creates a clean dataset of “healthy” turbine operations:
- Loads SCADA and Alarm data.
- Keeps only mean-value features.
- Defines consistent time windows.
- Filters out:
  - Critical failure periods.
  - Maintenance windows.
  - High-NaN targets or columns.
- Splits datasets into consecutive Train / Validation / Test sets.
- Saves results as per-turbine dictionaries of clean datasets.

#### `2_RunFS.ipynb`
Runs **Feature Selection (FS)** on the healthy datasets:
- Applies multiple FS techniques (e.g., correlation, mutual information).
- Evaluates features based on relevance and redundancy.
- Outputs ranked feature lists for each turbine dataset.
- Stores feature selection metadata for reuse in training.

#### `3_train_prediction_models.ipynb`
Trains predictive models:
- Loads previously selected features and clean datasets.
- Sets up modeling configurations and hyperparameter search.
- Trains multiple machine learning models (e.g., XGBoost, Random Forest).
- Evaluates using metrics (e.g., MAE, RMSE).
- Saves best-performing model and logs training results.

#### `4_plot_best_hp_set.ipynb`
Visualizes model performance:
- Loads the best hyperparameter configurations.
- Plots predictions vs actuals for Test sets.
- Provides diagnostic plots to compare model behavior across turbines.

### Folder: `Utils/`

This folder includes all reusable utility functions:
- Data cleaning tools
- Time windowing utilities
- Feature selection implementations
- Model evaluation helpers
- Plotting and logging functions

You can directly import from `Utils` in custom scripts or adapt them to other datasets and domains.
