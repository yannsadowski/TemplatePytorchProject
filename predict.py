import os
import yaml
import hydra
from omegaconf import DictConfig
import torch
import pandas as pd
from src.models.LSTMModel import LSTMModel
from src.data.data import transform_data, create_sequences
from src.predict.predict import predict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

@hydra.main(config_path="conf/", config_name="default", version_base="1.1")
def predict_main(dict_config: DictConfig):
    
    # Retrieve specific configurations
    dataset_config = dict_config.data
    model_config = dict_config.models
    predict_config = dict_config.predict
    
    # Determine the device (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check if GPU is available, otherwise raise an error
    if device.type != 'cuda':
        raise RuntimeError("This program requires a GPU to run.")

    # Initialize the model with the configurations
    model = LSTMModel(
        input_size=predict_config["input_size"],
        output_size=1,
        num_layers_lstm=model_config.num_layers_lstm,
        num_layers_dense=model_config.num_layers_dense,
        hidden_size_multiplier=model_config.hidden_size_multiplier,
        dropout=model_config.dropout,
        norm_type=model_config.norm_type
    )
    
    # Load the model
    model.load_state_dict(torch.load(get_path(predict_config.model_path)))
    model.to(device)  # Send model to GPU if available
    
    # Initialize the list for storing predictions 
    all_predictions = []

    # Load the Iris dataset from CSV file
    df = pd.read_csv(get_path(dataset_config['path']))
    
    # Using pandas' str accessor for splitting and selecting the month part
    df['month'] = df['month'].str.split('-').str[1].astype(int) / 12

    df['Target'] = (df['total_passengers'].pct_change().shift(-1)) * 100
    df = df.dropna()
    # 1. Extract features (X) and Target value (y)
    X =  df.drop(columns=['Target'])  # All columns expect target
    y = df['Target']

    X = create_sequences(X, dataset_config['sequences_size'])
    y = y[:len(y) - dataset_config['sequences_size'] + 1]

    # Iterate over each sample with a progress bar
    for sequence in tqdm(X, desc="Processing data", unit="sample"):
    
        # Transform data using the pre-trained transformer
        data = transform_data(sequence, dataset_config['transform'])
        
        # Perform the prediction using the model
        prediction = predict(model, data)
        
        # Store the prediction
        all_predictions.append(prediction)
    
   

    # Calculate regression metrics
    mse = mean_squared_error(y, all_predictions)
    mae = mean_absolute_error(y, all_predictions)
    r2 = r2_score(y, all_predictions)

    # Print the results
    print("Mean Squared Error (MSE): ", mse)
    print("Mean Absolute Error (MAE): ", mae)
    print("R² Score: ", r2)



# Function to get the correct file path
def get_path(path):
    # Get the current directory where the script is located
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # Build the full path using the relative path
    return os.path.join(current_directory, path)


if __name__ == "__main__":
    predict_main()
