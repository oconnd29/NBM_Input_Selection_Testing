import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import mutual_info_regression

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import shap
from sklearn.preprocessing import StandardScaler
import itertools
import time

def select_n_features_DT(train_set, target_var, n):

    # Separate features and target variables
    X_train = train_set.drop(columns=[target_var])
    y_train = train_set[target_var]
    
    model = DecisionTreeRegressor()  # For regression
    
    # Fit the model on the training data
    model.fit(X_train, y_train)
    
    # Extract and display the feature importance scores
    feature_importances = model.feature_importances_
    
    # Scale so all are between 0-1
    scaled_importances = feature_importances / max(feature_importances)
    
    # Create a DataFrame to view the importance scores along with the feature names
    DT_importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': scaled_importances
    })
    
    # Sort by importance
    DT_importance_df = DT_importance_df.sort_values(by='Importance', ascending=False)
    
    top_n_features = DT_importance_df['Feature'][0:n].values
    top_n_importances = DT_importance_df['Importance'][0:n].values

    return top_n_features, top_n_importances

def select_n_features_LC(train_set, target_var, n):

    # Compute corr matrix:
    df_corr_matrix = train_set.corr()
    correlation_df = pd.DataFrame(df_corr_matrix, index=train_set.columns[:], columns=train_set.columns[:])
    
    #Select row to highlight and print ordered influence
    highlight_row = correlation_df.loc[target_var]
    
    # get abs value of LC
    target_corrs = abs(highlight_row)

    # Remove self variable
    target_corrs = target_corrs.drop([target_var])

    #Normalise to 0-1
    target_corrs = target_corrs / max(target_corrs)

    # Create a DataFrame to hold the scores
    LC_importance_df  = pd.DataFrame({
        'Feature': target_corrs.index,
        'LC_values': target_corrs.values
    })
    
    # Sort the DataFrame by the mutual information scores in descending order
    LC_importance_df  = LC_importance_df.sort_values(by='LC_values', ascending=False)
    
    top_n_features = LC_importance_df ['Feature'][0:n].values
    top_n_importances = LC_importance_df ['LC_values'][0:n].values

    return top_n_features, top_n_importances

def select_n_features_MI(train_set, target_var, n):

    # Separate features and target variables
    X_train = train_set.drop(columns=[target_var])
    y_train = train_set[target_var]
    
    # Compute the mutual information
    mi_scores = mutual_info_regression(X_train, y_train)
    mi_scores = mi_scores/max(mi_scores)
    
    # Create a DataFrame to hold the scores
    mi_importance_df  = pd.DataFrame({
        'Feature': X_train.columns,
        'Mutual Information': mi_scores
    })
    
    # Sort the DataFrame by the mutual information scores in descending order
    mi_importance_df  = mi_importance_df.sort_values(by='Mutual Information', ascending=False)
    
    top_n_features = mi_importance_df ['Feature'][0:n].values
    top_n_importances = mi_importance_df ['Mutual Information'][0:n].values

    return top_n_features, top_n_importances


def select_n_features_MLPSHAP(train_set, valid_set, target_var, n, epochs = 200, background_samples = 400):

       
    # Step 3: Define a Simple MLP Model
    class SimpleMLP(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super(SimpleMLP, self).__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_dim, 1)  # Output layer (1-dimensional for regression)
        
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x
        
    #HP's for MLP
    hp_grid = {
    'batch_size' : [32,64],
    'hidden_dim' : [64,128],
    'learning_rate' : [0.01,0.001],
    }
    hp_combinations = list(itertools.product(*list(hp_grid.values())))

    print('hp_sets',hp_combinations)
    
    # Separate features and target variables
    X_train = train_set.drop(columns=[target_var])
    y_train = train_set[target_var]
    X_valid = valid_set.drop(columns=[target_var])
    y_valid = valid_set[target_var]
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_valid_tensor = torch.tensor(X_valid.values, dtype=torch.float32)
    y_valid_tensor = torch.tensor(y_valid.values, dtype=torch.float32).unsqueeze(1)
    X_train_numpy = X_train_tensor.cpu().detach().numpy()  

    best_valid_loss = float('inf')
    
    for i, hp_set in enumerate(hp_combinations):
        start_time = time.time()
        #load in hp_values
        batch_size, hidden_dim, learning_rate = hp_set
        print(f'hp_set: {i}')
            
        # Step 2: Create a PyTorch Dataset and DataLoader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
                
        # Initialize model, loss function, and optimizer
        input_dim = X_train_tensor.shape[1]      
        model = SimpleMLP(input_dim, hidden_dim)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
       
        # Step 4: Train the shap Model
        for epoch in range(epochs):
            model.train()
            train_count = 0
            for X_batch, y_batch in train_loader:
                # Skip batch if it contains NaNs   
                if torch.isnan(X_batch).any() or torch.isnan(y_batch).any():
                    train_count = train_count + 1
                    continue  
                    
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

            if (epoch) % 50 == 0 or epoch == 0:
                print(f'Epoch {epoch}')
                
                
        # Validation phase: Set the model to evaluation mode, turn of regularisation, and other
        model.eval() 
        total_val_loss = 0
        processed_samples = 0
        total_abs_error = 0
        valid_count = 0
        with torch.no_grad():
            for batch_idx, (X_batch, y_batch) in enumerate(valid_loader):
                
                # Skip batch if it contains NaNs   
                if torch.isnan(X_batch).any() or torch.isnan(y_batch).any():
                    valid_count = valid_count + 1

                    # # Create NaN tensors with the same shape as the model outputs, this is to fill preds with empties. (these are to store predictions)
                    # nans_shape = (y_batch.shape[0],)  # Shape should match the flattened output
                    # nan_tensor = torch.full(nans_shape, float('nan'))
        
                    # # Append NaNs for batches with NaNs
                    # y_true_valid.extend(nan_tensor.tolist())
                    # y_preds_valid.extend(nan_tensor.tolist())                    
                    continue 
                    
                #Foward
                y_pred = model(X_batch)
    
                #Comp Loss
                batch_loss = criterion(y_pred, y_batch)
                
                #Add to the loss from previous batches.
                total_val_loss += batch_loss.item()
                
                # #Compute AE
                # batch_abs_error = torch.abs(y_batch - y_pred.squeeze(1))
                # total_abs_error += sum(batch_abs_error)
                   
                # Update the count of processed samples
                processed_samples += len(X_batch)
                
                # # Flatten tensors and convert them to lists before extending
                # y_true_valid.extend(y_batch.view(-1).tolist())
                # y_preds_valid.extend(y_pred.view(-1).tolist())

            # #compute the mae across the processed samples in the valid set.
            # mae_eval = total_abs_error/ processed_samples
            
            # Compute the mean validation loss using the number of processed samples
            mean_val_loss_eval = total_val_loss / processed_samples

            end_time = time.time()
            print(f'Set {i} ,best_val_loss(MAE): {mean_val_loss_eval}, Took: {((end_time-start_time)/60):.2f}mins')
            
            # Save the best model(based on valid set)
            if mean_val_loss_eval < best_valid_loss :
                best_valid_loss = mean_val_loss_eval
                best_model = model
                best_hp_set = hp_set
            print('best set and model so far',best_hp_set, best_model)
                
    #Load in best model from hp_grid
    print(f'Best hp_set: {best_hp_set}')
    model = best_model
    device = torch.device("cpu")
    
    # Define a prediction function that converts PyTorch outputs to NumPy
    def model_predict(X):
        X_tensor = torch.tensor(X).float().to(device)  # Convert NumPy back to PyTorch tensor
        with torch.no_grad():
            predictions = model(X_tensor).cpu().numpy()  # Get predictions and convert to NumPy
        return predictions
        
    # Initialize KernelExplainer with the model prediction function and background data
    explainer = shap.KernelExplainer(model_predict, shap.sample(X_train_numpy, background_samples))  # Use a small subset of training data

    print(f"Explainer Type: {type(explainer)}")
    print(f"Explainer Attributes: {explainer.__dict__}")
    
    # Compute SHAP values for the validation set
    shap_values = explainer.shap_values(shap.sample(X_train_numpy, background_samples)) 
    
    # Get mean absolute SHAP values for each feature
    shap_values_abs_mean = np.mean(np.abs(shap_values), axis=0)
    
    # Step 6: Create SHAP-based "target_corrs" (normalized feature importances)
    # Convert to pandas Series for easier manipulation
    shap_importances = pd.Series(shap_values_abs_mean[:,0], index=X_train.columns)
    
    # Sort SHAP importances in descending order
    shap_importances_sorted = shap_importances.sort_values(ascending=False)
    
    # Normalize to 0-1 range
    shap_importances_normalized = shap_importances_sorted / shap_importances_sorted.iloc[0]
    
    # Select top features based on the desired number of features
    top_n_features = list(shap_importances_normalized.index[:n])

    #Return the importances
    top_n_importances = list(shap_importances_normalized.iloc[:n])

    
    return top_n_features, top_n_importances
    

