import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import torch
import torchinfo 
from src.data.data import prepare_dataloaders
from src.models.CNNModel import CNNModel
from src.trainer.train import train

@hydra.main(config_path="conf/", config_name="default", version_base="1.1")
def main(dict_config: DictConfig):
    # Retrieve specific configurations
    dataset_config = dict_config.data
    model_config = dict_config.models
    trainer_config = dict_config.trainer
    wandb_config = dict_config.wandb

    # Determine the device (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check if GPU is available, otherwise raise an error
    if device.type != 'cuda':
        raise RuntimeError("This program requires a GPU to run.")

    # Prepare the DataLoaders
    train_loader, val_loader, test_loader = prepare_dataloaders(
        config=dataset_config
    )
    
    # Initialize the model with the configurations
    model = CNNModel(
        input_channels=model_config.input_channels,  
        num_classes=model_config.num_classes,  
        num_conv_layers=model_config.num_conv_layers,  
        num_dense_layers=model_config.num_dense_layers, 
        hidden_size_multiplier=model_config.hidden_size_multiplier,  
        dropout=model_config.dropout, 
        kernel_size=model_config.kernel_size
    ).to(device)
    
    # Connect to Weights and Biases
    wandb.login(key=wandb_config.api_key)
    
    # Convert DictConfig to a standard dictionary for wandb
    wandb_config_dict = OmegaConf.to_container(dict_config, resolve=True, throw_on_missing=True)
    
    # Initialize Weights and Biases
    wandb.init(project=wandb_config.project_name, entity=wandb_config.get('entity', None), config=wandb_config_dict, settings=wandb.Settings(start_method="thread"))

    
    # Print the model for verification
    print(model)
    
    # Start the training process
    train(model, train_loader, val_loader, test_loader, trainer_config,dataset_config)
    
    # Assuming batch_size, number of channels, and image dimensions are available
    batch_size = dataset_config['batch_size']
    input_channels = model_config.input_channels
    image_height = 32  # For CIFAR-10, the image height is 32 pixels
    image_width = 32   # For CIFAR-10, the image width is also 32 pixels

    # Now include the batch size and the full input dimensions in the input_size argument
    summary_info = torchinfo.summary(model, input_size=(batch_size, input_channels, image_height, image_width), device=device, verbose=0)

    model_summary_str = str(summary_info)

    wandb.summary["total_params"] = summary_info.total_params
    wandb.summary["trainable_params"] = summary_info.trainable_params
    wandb.summary["total_mult_adds"] = summary_info.total_mult_adds
    wandb.summary["input_size"] = summary_info.input_size

    with open("model_summary.txt", "w", encoding='utf-8') as f:
        f.write(model_summary_str)
    artifact = wandb.Artifact('model-summary', type='model')
    artifact.add_file('model_summary.txt')
    wandb.log_artifact(artifact)
    
    # finish wandb run
    wandb.finish()
    
if __name__ == "__main__":
    main()
