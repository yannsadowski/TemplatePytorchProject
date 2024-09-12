import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import io

# Regression training

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_targets = []
    all_predictions = []

    for inputs, targets in tqdm(train_loader, desc="Training", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        all_targets.extend(targets.tolist())
        all_predictions.extend(outputs.tolist())  # Utilisation des prédictions continues

    epoch_loss = running_loss / len(train_loader.dataset)
    
    # Calcul des métriques de régression
    epoch_mse = mean_squared_error(all_targets, all_predictions)
    epoch_mae = mean_absolute_error(all_targets, all_predictions)
    epoch_r2 = r2_score(all_targets, all_predictions)

    return epoch_loss, epoch_mse, epoch_mae, epoch_r2

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Validation", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            all_targets.extend(targets.tolist())
            all_predictions.extend(outputs.tolist())  # Utilisation des prédictions continues

    epoch_loss = running_loss / len(val_loader.dataset)

    # Calcul des métriques de régression
    epoch_mse = mean_squared_error(all_targets, all_predictions)
    epoch_mae = mean_absolute_error(all_targets, all_predictions)
    epoch_r2 = r2_score(all_targets, all_predictions)

    return epoch_loss, epoch_mse, epoch_mae, epoch_r2


def train(model, train_loader, val_loader, test_loader, config, dataset_config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)

    # Initialize the learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    best_val_loss = float('inf')

    for epoch in range(config.epochs):
        start_time = time.time()
        print(f"Epoch {epoch+1}/{config.epochs}")

        train_loss, train_mse, train_mae, train_r2 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_mse, val_mae, val_r2 = validate(model, val_loader, criterion, device)
        epoch_time = time.time() - start_time

        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Train MSE: {train_mse:.4f}, Val MSE: {val_mse:.4f}")
        print(f"Train MAE: {train_mae:.4f}, Val MAE: {val_mae:.4f}")
        print(f"Train R²: {train_r2:.4f}, Val R²: {val_r2:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_mse": train_mse,
            "val_mse": val_mse,
            "train_mae": train_mae,
            "val_mae": val_mae,
            "train_r2": train_r2,
            "val_r2": val_r2,
            "epoch_time": epoch_time
        })

        # Step the scheduler based on the validation loss
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Model saved!")

    print("Training complete. Testing model...")

    test_loss, test_mse, test_mae, test_r2 = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    
    # Log a single sequence from the test_loader to WandB
    inputs, targets = next(iter(test_loader))  # Take the first batch from the test loader

    # Extract the first sequence (assuming shape is [batch_size, sequence_length, features])
    single_input_sample = inputs[0].cpu().numpy()  # Take the first sequence in the batch
    single_target_sample = targets[0].cpu().numpy()  # Corresponding target for the first sequence

    # Log only one sequence (single input sample) and its target
    wandb.log({
        "test_single_input": wandb.Table(data=single_input_sample.tolist(), columns=[f"feature_{i}" for i in range(single_input_sample.shape[-1])]),
        "test_single_target": single_target_sample
    })

    wandb.log({
        "test_loss": test_loss,
        "test_mse": test_mse,
        "test_mae": test_mae,
        "test_r2": test_r2
    })

    
