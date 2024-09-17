import os
import yaml
import hydra
from omegaconf import DictConfig
import torch
import pandas as pd
from models.CNNModel import CNNModel
from src.data.data import transform_data
from src.predict.predict import predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib 
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
    model = BaseModel(
        input_size=predict_config.input_size,
        output_size=predict_config.output_size,
        num_layers_dense=model_config.num_layers_dense,
        hidden_size_multiplier=model_config.hidden_size_multiplier,
        dropout=model_config.dropout,
        norm_type=model_config.norm_type
    )
    
    # Load the model
    model.load_state_dict(torch.load(get_path(predict_config.model_path)))
    model.to(device)  # Send model to GPU if available
    
    # Load classes mapping
    class_mapping = load_class_mapping(get_path(predict_config.class_mapping_path))
    
    # Initialize the list for storing predictions 
    all_predictions = []

    # Load the Iris dataset from CSV file
    df = pd.read_csv(get_path(dataset_config['path']))
    
    all_data = df.iloc[:, :-1].values  # All columns except the last one (features)
    y = df.iloc[:, -1].values   # Last column (target labels)

    # Load the pre-trained transformer
    loaded_transformer = joblib.load(get_path(predict_config.transformer_path))
    
    # Iterate over each sample with a progress bar
    for data_path in tqdm(all_data, desc="Processing data", unit="sample"):
    
        # Transform data using the pre-trained transformer
        data = transform_data(data_path, loaded_transformer)
        
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
    y_true = [label for label in y]
    y_pred = [class_mapping[int(pred)] for pred in all_predictions]

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=class_mapping.values())
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Print results
    print("Accuracy: ", accuracy)
    print("Classification Report: \n", report)
    print("Confusion Matrix: \n", conf_matrix)


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
