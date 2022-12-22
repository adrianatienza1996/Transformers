from src.ViT_model import ViT
from src.data_utils_cv import *

from torchsummary import summary
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np

from tqdm import tqdm
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

model = ViT(num_classes=365, num_blocks=6, num_heads=8, model_dim=768).to(device)
summary(model, (3, 224, 224))

print("Model Loaded")

optimizer = Adam(model.parameters(), lr=0.0001, betas=[0.9, 0.99])
epochs = 12
tr_dl, val_dl = get_data("/home/adrian/Desktop/Transformers/Datasets/Places365_2/Train", "/home/adrian/Desktop/Transformers/Datasets/Places365_2/Test", 92)

loss_fn = nn.CrossEntropyLoss()
best_test_loss = 999999

for epoch in range(epochs):
    print("Epoch: ", str(epoch))
    epoch_train_loss, epoch_test_loss = [], []
    
    for batch in tqdm(iter(tr_dl)):
        batch_loss = train_batch(batch, model, optimizer, loss_fn)
        epoch_train_loss.append(batch_loss)

    epoch_train_loss = np.array(epoch_train_loss).mean()

    for batch in tqdm(iter(val_dl)):
        batch_loss = test_batch(batch, model, loss_fn)
        epoch_test_loss.append(batch_loss)

    epoch_test_loss = np.array(epoch_test_loss).mean()

    if epoch_test_loss < best_test_loss:
        torch.save(model.to("cpu").state_dict(), "/home/adrian/Desktop/Transformers/Saved_Models/vit.pth")
        model.to(device)
        best_test_loss = epoch_test_loss
    
    print("Training loss: " + str(epoch_train_loss)[:5])
    print("Test loss: " + str(epoch_test_loss)[:5])  

    print("Model Saved")