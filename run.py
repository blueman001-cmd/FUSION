from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data import Data, DataLoader
import torch.nn as nn
import torch.optim as optim
from utils import *
from Model.GATConv import GAT
import argparse

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test a GAT model on graph data.")
    parser.add_argument("--data", type=str, default="data/train_test_data", help="Directory containing JSON graph data for training (default: data/train_test_data).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for DataLoader.")
    parser.add_argument("--model_name", type=str, default="model", help="model name.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs for training.")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"], help="Choose mode: 'train' or 'test' (default: train).")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the trained model file (model.pth) for testing.")
    parser.add_argument("--test_data_folder", type=str, default=None, help="Path to the folder containing test data for testing.")

    args = parser.parse_args()
    inputsize = 0

    if args.mode == "train":
        # Load data
        data = load_json_folder(args.data)
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

        Train_data = []
        for data in train_data:
            # print(data["node_feature"])
            # print(type(data["node_feature"]))
            x = torch.tensor(data["node_feature"], dtype=torch.float64).to(device)  # Move to GPU
            inputsize = x.shape[1]
            sources = [int(num) for num in data["edge_index"][0]]
            target = [int(num) for num in data["edge_index"][1]]
            edges = torch.tensor([sources, target]).to(device)  # Move to GPU
            y = torch.tensor([data["label"]], dtype=torch.long).to(device)  # Move to GPU

            Train_data.append(Data(x=x, edge_index=edges, y=y))

        # Create DataLoader instances
        train_loader = DataLoader(Train_data, batch_size=args.batch_size, shuffle=True)

        # Initialize model, criterion, and optimizer
        model = GAT(hidden_channels=32, input_size=inputsize).to(device)  # Move model to GPU
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

        # Training loop
        for epoch in range(args.epochs):
            # Training
            train_loss, a, p, r, f, model = train(model, optimizer, criterion, train_loader,device)

            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{args.epochs} Train Loss: {train_loss:.4f} Train Accuracy: {a:.4f}, Precision: {p:.4f}, Recall: {r:.4f}, F1 Score: {f:.4f}")
                # torch.save(model.state_dict(), 'model/'+args.model_name+'.pth')
                # print("save model OK")
        
        torch.save(model.state_dict(), 'model/'+args.model_name+'.pth')
        print("save model OK")
        
    elif args.mode == "test":
        # Check if model_path and test_data_folder are provided
        if args.model_path is None or args.test_data_folder is None:
            raise ValueError("In test mode, --model_path and --test_data_folder must be provided.")

        # Load test data
        Test_data = []
        test_data = load_json_folder(args.test_data_folder)
        for data in test_data:
            x = torch.tensor(data["node_feature"], dtype=torch.float64).to(device)  # Move to GPU
            sources = [int(num) for num in data["edge_index"][0]]
            target = [int(num) for num in data["edge_index"][1]]
            edges = torch.tensor([sources, target]).to(device)  # Move to GPU
            y = torch.tensor([data["label"]], dtype=torch.long).to(device)  # Move to GPU

            Test_data.append(Data(x=x, edge_index=edges, y=y))

        # Create DataLoader instance for test data
        test_loader = DataLoader(Test_data, batch_size=args.batch_size, shuffle=False)

        # Initialize model and criterion
        model = GAT(hidden_channels=32, input_size=77).to(device)  # Move model to GPU
        model.load_state_dict(torch.load(args.model_path, map_location=device))  # Load model to GPU
        model.eval()

        # Testing
        a, p, r, f = test(model, test_loader)
        print(f"Accuracy: {a:.4f}, Precision: {p:.4f}, Recall: {r:.4f}, F1 Score: {f:.4f}")