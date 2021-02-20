import torch
import numpy as np
import sys
sys.path.append("../") # moves sys path to parent which allows import of datasets module

from torch.utils.data import DataLoader
from torchvision.models import alexnet
from datasets.bird_dataset import BirdDataset

# Load Datasets
train_dataset = BirdDataset(csv_path="../data/bird_train.csv", root_dir="../data/images")
val_dataset = BirdDataset(csv_path="../data/bird_val.csv", root_dir="../data/images")

train = DataLoader(train_dataset, batch_size=4)
val = DataLoader(val_dataset, batch_size=4)

data_loaders = {"train": train, "val": val}
data_lengths = {"train": len(train_dataset), "val": len(val_dataset)}

# Initialize
model = alexnet(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.classifier[-1] = torch.nn.Linear(4096, 12, bias=True)
optimizer = torch.optim.Adam(model.parameters(), lr=.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1, verbose=True)
loss_func = torch.nn.CrossEntropyLoss()
losses = {"train": [], "val": []}
n_epochs = 30

# Train + validate
best_model = None
min_val_loss = np.inf
for epoch in range(n_epochs):
    print(f"Epoch {epoch}/{n_epochs}")
    print("-" * 10)
    for phase in ["train", "val"]:
        if phase == "train":
            model.train(True) # set model tp training mode
        else:
            model.train(False) # Set model to evaluate mode
        running_loss = 0
        for batch in data_loaders[phase]:
            # Get the input
            inputs, labels = batch
            # Zero the param gradients
            optimizer.zero_grad()
            # Get output
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            # backward + optimize only if in training phase
            if phase == 'train':
                loss.backward()
                # update the weights
                optimizer.step()
            # print statistics
            running_loss += loss.item()
        # update scheduler
        if phase == "train":
            scheduler.step()
        epoch_loss = running_loss / data_lengths[phase]
        # Saves the best model
        if (phase == "val") and epoch_loss < np.inf:
            best_model = model
        print('{} Loss: {:.4f}'.format(phase, epoch_loss))
        losses[phase].append(epoch_loss)
print("Finished Training")

# Basic Visualization
#plt.plot(np.arange(30), losses["train"])
#plt.plot(np.arange(30), losses["val"])

torch.save(best_model, "completed_models/alexnet_bird_classifier")
print("Saved Model")
