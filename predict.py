import os
import yaml
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from src.models.Model import BaseModel
from src.data.data import transform_data
from src.predict.predict import predict
from sklearn.metrics import accuracy_score
from collections import defaultdict
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
    ).to(device)
    
    # Load the model
    model_path = model_config.model_path
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Store predictions 
    all_predictions = []


    # Get the list of data
    all_data = []


    # Iterate over each file with a progress bar
    for data_path in tqdm(all_data, desc="Processing data", unit="Sample"):
      
        # Transform data
        data = transform_data(data_path)

        prediction = predict(model, data)
        
        final_prediction = torch.tensor(prediction, device=device)
        
        
        # Store the prediction and the true label
        all_predictions.append(final_prediction)

    
    # Calculate  accuracy

    
    # Calculate overall accuracy


if __name__ == "__main__":
    predict_main()
