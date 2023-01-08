from model import ViT
from data_utils import *

from torchsummary import summary
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

#################################################################################################################################################################
##########################################################  MODEL INITIALIZATION  ###############################################################################
#################################################################################################################################################################

model = ViT(num_classes=365, 
            num_blocks=12, 
            num_heads=8, 
            model_dim=512).to(device)


summary(model, (3, 224, 224))
print("Model Loaded")



saved_model_path = "C:/Users/adria/Desktop/Repositories/Transformers/ViT/Saved_Model/vit.pth"
model.load_state_dict(torch.load(saved_model_path))

current_epoch = 5
#################################################################################################################################################################
###########################################################  LOADING DATA  ######################################################################################
################################################################################################    #################################################################

data_train_path = "C:/Users/adria/Desktop/Repositories/CV Datasets/Places365/Labeled/Train"
data_test_path = "C:/Users/adria/Desktop/Repositories/CV Datasets/Places365/Labeled/Test"

BS = 72

tr_dl, val_dl = get_data(data_train_path, data_test_path, BS)

#################################################################################################################################################################
###########################################################  TRAINING LOOP HYPERPARAMETERS  #####################################################################
##############################################################################################################################################      ###################

EPOCHS = 12 - current_epoch

optimizer = Adam(model.parameters(), lr=0.0001, betas=[0.9, 0.99])

lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                    milestones=[int(0.75 * EPOCHS - current_epoch), int(0.9 * EPOCHS - current_epoch)], 
                                                    gamma=0.1)

loss_fn = nn.CrossEntropyLoss()
best_test_loss = 2.220


writer = SummaryWriter(log_dir="ViT/logs/losses")

#################################################################################################################################################################
#################################################################  TRAINING LOOP   ##############################################################################
#################################################################################################################################################################


for epoch in range(EPOCHS):
    print("Epoch: ", str(epoch))
    epoch_train_loss, epoch_test_loss = [], []
    pbar = tqdm(tr_dl)
    
    for batch in pbar:
        batch_loss = train_batch(batch, model, optimizer, loss_fn)
        epoch_train_loss.append(batch_loss)
        pbar.set_postfix(MSE=batch_loss)

    epoch_train_loss = np.array(epoch_train_loss).mean()
    writer.add_scalar("Training Loss", epoch_train_loss, global_step=epoch)

    pbar = tqdm(val_dl)
    for batch in pbar:
        batch_loss = test_batch(batch, model, loss_fn)
        epoch_test_loss.append(batch_loss)
        pbar.set_postfix(MSE=batch_loss)

    epoch_test_loss = np.array(epoch_test_loss).mean()
    writer.add_scalar("Validation Loss", epoch_test_loss, global_step=epoch)

    if epoch_test_loss < best_test_loss:
        torch.save(model.to("cpu").state_dict(), saved_model_path)
        model.to(device)
        best_test_loss = epoch_test_loss
    
    print("Training loss: " + str(epoch_train_loss)[:5])
    print("Test loss: " + str(epoch_test_loss)[:5])  
    
    lr_scheduler.step()
    print("Model Saved")