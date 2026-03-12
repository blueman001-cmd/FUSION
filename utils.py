from tqdm import tqdm
import os
import json
import torch

def load_json_folder(folder_path):
    data_array = []

    # Iterate through files in the folder
    for filename in tqdm(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)

        # Check if the file is a JSON file
        if filename.endswith(".json"):
            # Read and load the JSON file
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)

            # Append the loaded data to the array
            data_array.append(data)


    return data_array
def convert_to_one_hot(label_tensor, num_classes):
    """
    Convert a label tensor to its one-hot encoded representation.

    Args:
    - label_tensor (torch.Tensor): Tensor containing the labels.
    - num_classes (int): Number of classes.

    Returns:
    - torch.Tensor: One-hot encoded tensor.
    """
    # Initialize the one-hot encoded tensor
    one_hot_tensor = torch.zeros(len(label_tensor), num_classes)

    # Fill in the one-hot encoded tensor
    one_hot_tensor[range(len(label_tensor)), label_tensor] = 1

    return one_hot_tensor
def train(model, optimizer, criterion, loader, device):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    model.train()
    total_loss = 0
    y_true = []
    y_pred = []

    for data in loader:  # Iterate in batches over the train dataset.
        # Ensure data is on the correct device and type
        data.x = data.x.to(device, dtype=torch.float32)  # Convert to float32 and ensure on GPU
        data.edge_index = data.edge_index.to(device, dtype=torch.long)  # Ensure edge_index is long and on GPU
        data.y = data.y.to(device, dtype=torch.long)  # Ensure y is long and on GPU
        data.batch = data.batch.to(device) if data.batch is not None else None  # Handle batch index for batched graphs

        # Forward pass
        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass
        loss = criterion(out, data.y)  # Compute loss (CrossEntropyLoss expects long labels, not one-hot)
        loss.backward()  # Derive gradients
        optimizer.step()  # Update parameters based on gradients
        optimizer.zero_grad()  # Clear gradients

        # Compute predictions for metrics
        pred = out.argmax(dim=1)  # Get predicted class
        y_true.extend(data.y.cpu().tolist())  # Move to CPU for sklearn metrics
        y_pred.extend(pred.cpu().tolist())  # Move to CPU for sklearn metrics
        total_loss += loss.item() * data.num_graphs  # Accumulate loss (scale by batch size)

    # Calculate metrics
    average_loss = total_loss / len(loader.dataset)  # Average loss per graph
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    return average_loss, accuracy, precision, recall, f1, model
def test(model, loader):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    model.eval()
    y_true = []
    y_pred = []
    correct = 0

    for data in loader:  # Iterate in batches over the training/test dataset.
        data.edge_index = data.edge_index.to(torch.long)
        data.x = data.x.to(model.parameters().__next__().dtype)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)  # Use the class with the highest probability.
        y_true.extend(data.y.tolist())
        y_pred.extend(pred.tolist())
        # correct += int((pred == data.y).sum())  # Check against ground-truth labels.

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)  # or 'micro' or 'weighted'
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    # Compute accuracy
    # accuracy = correct / len(loader.dataset)

    return accuracy, precision, recall, f1