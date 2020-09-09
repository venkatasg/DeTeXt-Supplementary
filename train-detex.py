import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import random
from torchvision import transforms, datasets, models
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score


EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.003
SIGDIG=3
SEED=1220
THRESHOLD=0.001

TRANSFORMS = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])])

class mnetv2(nn.Module):
    def __init__(self, num_classes):
        super(mnetv2, self).__init__()
        self.mobilenet = models.mobilenet_v2(pretrained=False, progress=False,
                                             num_classes=num_classes)
    def forward(self, x):
        output = x
        return output

if __name__ == '__main__':

    ## Set random seeds for reproducibility on a specific machine
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.Generator().manual_seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    np.random.RandomState(SEED)

    # Loading data
    data = datasets.ImageFolder('../detexify-data/drawings/', transform=TRANSFORMS)
    train_len = round(0.85*len(data))
    dev_len = len(data) - train_len
    train_data, dev_data = torch.utils.data.random_split(data, [train_len, dev_len])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE,
                                               shuffle=True)
    dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=BATCH_SIZE,
                                             shuffle=True)

    # Get the symbol name corresponding to each class ID (integer)
    class_ids = {v:k for k,v in data.class_to_idx.items()}
    class_names = [class_ids[i] for i in class_ids.keys()]

    model = models.mobilenet_v2(pretrained=False, progress=False,
                                num_classes=len(class_ids.keys()))

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_func = nn.CrossEntropyLoss()

    ## GPU shit
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        device_ids=[i for i in range(torch.cuda.device_count())]
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    dev_f1_list = [0]

    for epoch in range(EPOCHS):
        print("<" + "="*40 + F" Epoch {epoch} "+ "="*40 + ">")

        # Calculate total loss for this epoch
        total_train_loss = 0
        model.train()
        for step, (b_x, b_y) in enumerate(tqdm(train_loader)):

            b_x = b_x.to(device)
            b_y = b_y.to(device)

            output = model(b_x)
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_train_loss += loss.mean().item()

        # Calculate the average loss over the training data.
        avg_train_loss = total_train_loss / len(train_loader)
        print(F'\n\t\tAverage Training loss: {avg_train_loss}')

        print("\n\tRunning Validation...")
        # Put model in evaluation mode to evaluate loss on the validation set
        model.eval()

        # Tracking variables
        preds = np.array([])
        targets = np.array([])
        total_eval_loss = 0

        for step, (b_x, b_y) in enumerate(dev_loader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            with torch.no_grad():
                test_output = model(b_x)

            loss = loss_func(test_output, b_y)
            total_eval_loss += loss.mean().item()

            pred_y = torch.max(test_output, 1)[1].data.to('cpu').numpy()
            preds = np.append(pred_y, preds)
            targets = np.append(b_y.to('cpu').numpy(), targets)

        targets = targets.astype(int)
        preds = preds.astype(int)
        dev_f1 = f1_score(targets, preds, average='micro')
        avg_dev_loss = total_eval_loss / len(dev_loader)
        target_names = [class_ids[i] for i in np.unique(np.concatenate((targets,preds)))]
        report = classification_report(targets, preds, target_names=target_names,
                                       digits=SIGDIG, zero_division=0)
        print('Validation loss: %.4f' % avg_dev_loss, '| dev micro-f1: %.2f' % dev_f1)
        if dev_f1 - dev_f1_list[-1] > THRESHOLD:
            model_to_save = model.module if hasattr(model, "module") else model
            torch.save(model_to_save.state_dict(), 'mobilenet.bin')
            with open('mnetreport_epoch:' + str(epoch+1) + '.txt', 'w') as f:
                f.write(report)
            dev_f1_list.append(dev_f1)
        else:
            break

