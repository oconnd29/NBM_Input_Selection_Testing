import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
from torch.utils.data import Dataset
import torch
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler


"""
Classes
"""

class TimingCallback:
    def __init__(self):
        self.logs = []

    def on_epoch_begin(self, epoch):
        self.starttime = time.time()

    def on_epoch_end(self, epoch):
        elapsed_time = time.time() - self.starttime
        self.logs.append(elapsed_time)
        print(f"Epoch {epoch + 1} took {elapsed_time:.4f} seconds")

class TimeseriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)


class RunningRMSE:
    def __init__(self):
        self.sum_squared_error = 0.0
        self.count = 0

    def update(self, y_true, y_pred):
        """
        Update the running RMSE with a new pair of true and predicted values.

        Parameters:
        y_true (float): The true value at the current time step.
        y_pred (float): The predicted value at the current time step.
        """
        error = y_true - y_pred
        self.sum_squared_error += error ** 2
        self.count += 1

    def compute_rmse(self):
        """
        Compute the RMSE up to the current time step.

        Returns:
        float: The current RMSE.
        """
        if self.count == 0:
            return float('nan')  # Avoid division by zero
        return np.sqrt(self.sum_squared_error / self.count)

class RunningEWMA:
    def __init__(self, lamda):
        self.lamda = lamda
        self.current_ewma = None

    def update(self, new_value):
        """
        Update the running EWMA with a new value.

        Parameters:
        new_value (float): The new value to update the EWMA with.
        """
        if self.current_ewma is None:
            # Initialize the EWMA with the first value
            self.current_ewma = new_value
        else:
            # Update the EWMA
            self.current_ewma = self.lamda * new_value + (1 - self.lamda) * self.current_ewma

    def get_ewma(self):
        """
        Get the current EWMA value.

        Returns:
        float: The current EWMA value.
        """
        return self.current_ewma


"""
Functions
"""
    
def plot_alarmDF(alarm_df, alarm_code_dict, ax=None):
    # If no axis is provided, create a new figure and axis
    if ax is None:
        print('Please provide variable to plot')
    else:
        fig = ax.get_figure()
        y_min, y_max = ax.get_ylim()
        y_max = y_max * 0.9
        y_range = y_max - y_min

    # Extract all unique alarm codes
    unique_codes = set()
    for alarms in alarm_df['Alarms']:
        for alarm in alarms:
            unique_codes.add(alarm['Code'])

    # Sort unique_codes numerically
    unique_codes = sorted(unique_codes, key=lambda x: int(x))  # Assuming format 'ALM<number>'
    color_map = plt.cm.get_cmap('tab10', len(unique_codes))
    code_colors = {}

    # Assign colors to alarm codes, avoiding pure white
    for i, code in enumerate(unique_codes):
        color = color_map(i)
        if np.allclose(color[:3], [1.0, 1.0, 1.0], atol=0.05):   # If the color is exactly white
            color = (0.5, 0.5, 0.5, 1.0)  # Shift to another color
        code_colors[code] = color
        print(code, ':',color)

    # Each span/arrow should cover/locate @  1/20th of the y-axis - plus 1 for div 0 problem
    y_portion = 1 / (len(unique_codes) + 1)

    # Define the y-axis range for each unique alarm  
    y_margin = ax.margins()[1]
    span_range = 1 - y_margin - y_margin

    # Arrows should be at portion height in SCADA variable in this case
    y_ranges_arrows = {code: (((i * y_portion * y_range) + y_min),
                              (((i + 1) * y_portion * y_range) + y_min))
                       for i, code in enumerate(unique_codes)}
    # Spans should cover only 90% of the figure, i.e., not the margins.
    y_ranges_spans = {code: (y_margin + (i * span_range * y_portion),
                             y_margin + ((i + 1) * span_range * y_portion))
                      for i, code in enumerate(unique_codes)}

    # Plotting alarm intervals with assigned colors
    for time, alarms in tqdm(alarm_df['Alarms'].items()):
        for alarm in alarms:
            start = alarm['Timestamp start']
            end = alarm['Timestamp end']
            code = alarm['Code']
            color = code_colors[code]
            y_min_span, y_max_span = y_ranges_spans[code]
            y_min_arrow, y_max_arrow = y_ranges_arrows[code]

            # Using axvspan to highlight the interval with specific y-ranges
            ax.axvspan(start, end, facecolor=color, alpha=1, ymin=y_min_span, ymax=y_max_span)

            # Add an arrow at the start of the span
            ax.annotate('', 
                    xy=(start, y_max_arrow), 
                    xytext=(start, y_max_arrow + y_portion * y_max / 2),
                    arrowprops=dict(facecolor=color, 
                                    edgecolor='black',  # Black outline
                                    linewidth=0.5,      # Thin outline
                                    shrink=0.05, 
                                    width=2, 
                                    headwidth=8))

    # Formatting the plot
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Create legend elements
    legend_elements = []

    for code, color in code_colors.items():
        category = alarm_code_dict[code]['Global contract category']
        message = alarm_code_dict[code]['Message']

        # Check if category is not NaN
        if category is not None and not (isinstance(category, float) and np.isnan(category)):
            category_lower = category.lower()
            message_lower = message.lower()
            if 'maintenance' in category_lower:
                label = f"Maintenance: {message}"
            elif any(keyword in category_lower for keyword in ['stop', 'fault', 'outage']):
                label = f"Critical: {message}"
            else:
                label = f"Operational Alarm: {message}, {alarm_code_dict[code]['Global contract category']}"
        elif 'generator' in message:
            label = f"Critical: {message}"
        else:
            label = f"Operational Alarm: {message}"

        legend_elements.append(
            plt.Line2D([0], [0], color=color, lw=4, label=label)  # Match legend color with arrow color
        )

    # Set legend to have 3 rows
    num_columns = int(np.ceil(len(unique_codes) / 3))  # Calculate number of columns for 3 rows

    if len(legend_elements) != 0:
        ax.legend(handles=legend_elements,
                  title="Alarm Codes",
                  bbox_to_anchor=(0.5, -0.3),  # Adjust for space below the figure
                  loc='upper center',         # Centered below the figure
                  ncol=num_columns,           # Set the number of columns for 3 rows
                  fontsize=8)
    return fig, ax


def segment_timeseries(input_array, target_array, window_size):
    """
    Segments the input and target arrays into training sets with a specified window size.
    
    Args:
    - input_array (np.ndarray): Array containing the input variables.
    - target_array (np.ndarray): Array containing the target variable.
    - window_size (int): The size of the window for segmentation (default is 24).
    
    Returns:
    - X_train (np.ndarray): Segmented input data.
    - y_train (np.ndarray): Segmented target data.
    """
    
    num_points = len(input_array)
    assert len(target_array) == num_points, "Input and target arrays must have the same length"
    # assert input_array.shape[1] == 11, "Input array must have 11 variables"
    # assert target_array.shape[1] == 1, "Target array must have 1 variable"

    X = []
    y = []

    for i in range(window_size - 1, num_points):  # if window = 10 and 6400 samples:   i=4,5,6...6400
        X_window = input_array[i - window_size + 1: i+1, :]
        # print(X_window)
        # print(len(X_window))
        y_value = target_array[i]
        # print(y_value)
        X.append(X_window)
        y.append(y_value)

    X = np.array(X)
    y = np.array(y).flatten()  # Flatten y_train to 1D array
    
    return X, y

def segment_timeseries_v2(input_array, target_array, window_size, steps_ahead):
    """
    Segments the input and target arrays into training sets with a specified window size and steps ahead.
    
    Args:
    - input_array (np.ndarray): Array containing the input variables.
    - target_array (np.ndarray): Array containing the target variable.
    - window_size (int): The size of the window for segmentation.
    - steps_ahead (int): The number of steps ahead from the end of the window to take the target value.
    
    Returns:
    - X_train (np.ndarray): Segmented input data.
    - y_train (np.ndarray): Segmented target data.
    """
    
    num_points = len(input_array)
    assert len(target_array) == num_points, "Input and target arrays must have the same length"
    
    X = []
    y = []

    # Adjust the loop to stop early based on the steps_ahead parameter
    for i in range(window_size - 1, num_points - steps_ahead):
        X_window = input_array[i - window_size + 1: i + 1, :]
        y_value = target_array[i + steps_ahead]  # Take the target value steps_ahead beyond the end of the window
        
        X.append(X_window)
        y.append(y_value)

    X = np.array(X)
    y = np.array(y).flatten()  # Flatten y_train to 1D array
    
    return X, y
    

def MinMaxScale_datasets(datasets, target_var):
    # Create a new dictionary to hold the scaled datasets
    scaled_datasets = {}
    
    # Loop through the datasets and normalize each, storing the results in scaled_dataset_dict
    for key, ds in datasets.items():
        # Initialize the scaler
        scaler = MinMaxScaler()
        
        # Separate the features and the target variable in the train set
        train_features = ds['train'].drop(columns=[target_var])
        train_target = ds['train'][target_var]
        
        # Normalize the features of the train set
        train_scaled = pd.DataFrame(scaler.fit_transform(train_features), 
                                    columns=train_features.columns, 
                                    index=train_features.index)
        # Add the target variable back to the scaled train set
        train_scaled[target_var] = train_target
        
        # Repeat for the validation set
        valid_features = ds['valid'].drop(columns=[target_var])
        valid_target = ds['valid'][target_var]
        
        valid_scaled = pd.DataFrame(scaler.transform(valid_features),
                                    columns=valid_features.columns,
                                    index=valid_features.index)
        valid_scaled[target_var] = valid_target
        
        # Repeat for the test set
        test_features = ds['test'].drop(columns=[target_var])
        test_target = ds['test'][target_var]
        
        test_scaled = pd.DataFrame(scaler.transform(test_features),
                                   columns=test_features.columns,
                                   index=test_features.index)
        test_scaled[target_var] = test_target
        
        # Store the scaled data in the new dictionary
        scaled_datasets[key] = {
            'train': train_scaled,
            'valid': valid_scaled,
            'test': test_scaled
        }
        
    return scaled_datasets

def correlation_plot(df, plot = False):
    """
    This function takes a DataFrame and plots a correlation matrix.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the data for correlation analysis.
    
    Returns:
    correlation_matrix
    """
    # Compute the correlation matrix
    correlation_matrix = df.corr()

    # Plot the correlation matrix using a heatmap
    if plot == True:
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, annot_kws={"size": 2},xticklabels=df.columns, yticklabels=df.columns)
        plt.xticks(fontsize=4)
        plt.yticks(fontsize=4)
        plt.title('Correlation Matrix of Sensor Measurements')
        plt.show()
    return correlation_matrix 

def top_correlated_vars(base_var, correlation_matrix , top_n):
    """
    Finds the top N variables that correlate most with the specified base variable.
    
    Parameters:
    correlation_matrix: 
    base_var (str): The base variable to find correlations with.
    top_n (int): The number of top correlated variables to return (including the base variable itself).
    
    Returns:
    List[str]: List of the top N correlated variables.
    """
    # Get correlations of the base variable with all other variables
    base_var_correlations = correlation_matrix[base_var]

    # Sort correlations in descending order and get the top N
    top_correlated = base_var_correlations.abs().sort_values(ascending=False).head(top_n).index.tolist()

    return top_correlated