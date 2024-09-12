import torch
import torch.nn.functional as F

# inference classifications
def predict(model,data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    with torch.no_grad():
        data = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)
        output = model(data)
    return output.item()