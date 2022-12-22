import torchvision
from torchvision import transforms
import torch
import torch.utils.data as data


scaler = torch.cuda.amp.GradScaler()

device = "cuda" if torch.cuda.is_available() else "cpu"

TRANSFORM_IMG_TRAIN = transforms.Compose([
    transforms.RandAugment(),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])

TRANSFORM_IMG_TEST = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])



def get_data(train_root, test_root, batch_size):

    train_data = torchvision.datasets.ImageFolder(root=train_root, transform=TRANSFORM_IMG_TRAIN)
    train_data_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = torchvision.datasets.ImageFolder(root=test_root, transform=TRANSFORM_IMG_TEST)
    test_data_loader = data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_data_loader, test_data_loader


def train_batch(data, model, opt, loss_fn):
    model.train()
    opt.zero_grad()
    
    img, gt = data
    img = img.to(device)
    gt = gt.to(device)

    with torch.cuda.amp.autocast():
        model_output = model(img)
        loss = loss_fn(model_output, gt)

    scaler.scale(loss).backward()
    scaler.step(opt)

   # Updates the scale for next iteration
    scaler.update()
    return loss.item()


@torch.no_grad()
def test_batch(data, model, loss_fn):
    model.eval()
    print(len(data))
    img, gt = data
    img = img.to(device)
    gt = gt.to(device)

    model_output = model(img)

    loss = loss_fn(model_output, gt)
    return loss.item()
