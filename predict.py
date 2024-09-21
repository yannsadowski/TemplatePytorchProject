import os
import yaml
import hydra
from omegaconf import DictConfig
import torch
import pandas as pd
from src.models.CNNModel import CNNModel
from src.data.data import transform_data
from src.predict.predict import predict
import re
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
    model = CNNModel(
        input_channels=model_config.input_channels,  
        num_classes=model_config.num_classes,  
        num_conv_layers=model_config.num_conv_layers,  
        num_dense_layers=model_config.num_dense_layers, 
        hidden_size_multiplier=model_config.hidden_size_multiplier,  
        dropout=model_config.dropout, 
        kernel_size=model_config.kernel_size
    )
    
    # Load the model
    model.load_state_dict(torch.load(get_path(predict_config.model_path)))
    model.to(device)  # Send model to GPU if available
    
    # Load classes mapping
    class_mapping = load_class_mapping(get_path(predict_config.class_mapping_path))
    
    # Initialize the list for storing predictions 
    all_predictions = []

    # List to store all image paths
    pictures_path = []

    # Function to extract the numerical part of the filename
    def extract_number(file_name):
        return int(re.search(r'\d+', file_name).group())

    # Iterate through the directory to get the paths of each image, sorted numerically
    for root, dirs, files in os.walk(get_path(predict_config['data_path'])):
        # Sort files numerically based on the numbers in their filenames
        files.sort(key=extract_number)
        for file in files:
            pictures_path.append(os.path.join(root, file))
    
    print(pictures_path[0:5])
    
    # Iterate over each sample with a progress bar
    for picture in tqdm(pictures_path, desc="Processing Picture", unit="Picture"):
    
        # Transform data using the pre-trained transformer
        data = transform_data(picture)
        
        # Perform the prediction using the model
        prediction = predict(model, data)
        
        # Get the predicted class using torch.argmax
        final_prediction = torch.argmax(torch.tensor(prediction, device=device), dim=-1)
        
        # Ensure final_prediction is a 1D tensor before adding it
        final_prediction = final_prediction.unsqueeze(0)  # Add dimension if necessary
        
        # Store the prediction
        all_predictions.append(final_prediction)
    
    # Concatenate predictions and convert to numpy array
    if all_predictions:  # Check if the list is not empty
        all_predictions = torch.cat(all_predictions).cpu().numpy()
    else:
        print("No predictions were made.")
        return
    
    # Use class_mapping to get actual class labels
    y_pred = [class_mapping[int(pred)] for pred in all_predictions]
    
    submission_df = pd.DataFrame({
        'label': y_pred
    })

    # Add an 'id' column that corresponds to the index, starting at 1
    submission_df['id'] = submission_df.index + 1

    # Rearrange the columns to have 'id' as the first column
    submission_df = submission_df[['id', 'label']]

    # Save the DataFrame to a CSV file
    output_path = 'predictions_submission.csv'
    submission_df.to_csv(output_path, index=False)



# Function to load class mapping from a YAML file
def load_class_mapping(file_path):
    with open(file_path, 'r') as file:
        # Load the YAML file into a dictionary
        class_mapping = yaml.safe_load(file)
    return class_mapping['classes']

# Function to get the correct file path
def get_path(path):
    # Get the current directory where the script is located
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # Build the full path using the relative path
    return os.path.join(current_directory, path)


if __name__ == "__main__":
    predict_main()
