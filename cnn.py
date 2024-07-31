import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, utils
from torch.utils import data
import time
from tqdm import tqdm
import numpy as np
import torchvision

# class Models(torch.nn.Module):
#     def __init__(self, num_classes):
#         super(Models, self).__init__()
#         self.Conv = torch.nn.Sequential(
#             torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
#             torch.nn.ReLU(),
#             torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             torch.nn.ReLU(),
#             torch.nn.MaxPool2d(kernel_size=2, stride=2),
#
#             torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
#             torch.nn.ReLU(),
#             torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
#             torch.nn.ReLU(),
#             torch.nn.MaxPool2d(kernel_size=2, stride=2))
#
#         self.Classes = torch.nn.Sequential(
#             torch.nn.Linear(16 * 16 * 256, 512),
#             torch.nn.ReLU(),
#             torch.nn.Dropout(p=0.5),
#             torch.nn.Linear(512, num_classes))
#
#     def forward(self, inputs):
#         x = self.Conv(inputs)
#         x = x.view(-1, 16 * 16 * 256)
#         x = self.Classes(x)
#         return x



def get_model(num_classes):
    model = torchvision.models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(p=0.3),
        torch.nn.Linear(512, num_classes)
    )
    return model

model = get_model(num_classes=17)
print(model)

#====================================================================
num_classes = 17
batch_size = 1
learning_rate = 0.01
num_epochs = 1000
#====================================================================

data_transform = {
    "train": transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                # transforms.RandomRotation(10),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                #                      std=[0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                #                      std=[0.229, 0.224, 0.225]),
    ])
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_path = r'D:\17flowers'

#============================================================================
train_dataset = datasets.ImageFolder(root=image_path + "/train",
                                     transform=data_transform["train"])

train_num = len(train_dataset)
print(f'train_num: {train_num}')

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

#=============================================================================
val_dataset = datasets.ImageFolder(root=image_path + "/val",
                                     transform=data_transform["val"])

val_num = len(val_dataset)
print(f'val_num: {val_num}')

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

#=============================================================================
flower_list = train_dataset.class_to_idx
print(f'{flower_list}')

# net = model(num_classes=num_classes)
model.to(device=device)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

model=model.cuda()

save_path = ''

best_acc = 0

for epoch in range(num_epochs):
    # print(f'epoch {epoch}/{num_epochs-1}')
    print("-"*20)

    # running_loss = 0
    # running_corrects = 0
    train_loss = 0
    train_acc = 0
    test_loss = 0
    test_acc = 0

    time_start = time.time()
    print(f"Epoch {epoch+1}/{num_epochs}")
    model.train()
    for step, data in tqdm(enumerate(iter(train_loader))):
    # for step, data in tqdm(enumerate(iter(tr))):
        x, y = data
        x, y = x.cuda(), y.cuda()
        # print(x.shape,  y.shape)
        # print(f"{images}---{labels}")
        # optimizer.zero_grad()
        y_pred = model(x)
        # print(f'y_pred: {y_pred.shape}')
        _, pred = torch.max(y_pred, 1)

        # out_prob = torch.softmax(pred, dim=-1)
        topk_prob_values, topk_class_indices = torch.topk(y_pred, k=5, dim=-1)
        print(f'topk_prob_values: {topk_prob_values}')

        loss = loss_function(y_pred, pred)
        acc_count = torch.sum( pred ==y ).item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss = loss.cpu()
        # acc = acc.cpu()
        train_loss+=loss
        train_acc+=acc_count

        # train_loss.append(loss.data)
        # train_acc.append(acc)
    print(f'train_acc: {train_acc}')
    time_end = time.time()
    # print(train_loss, train_acc)
    # print(f'batch:{batch_size},train_loss:{sum(train_loss)/len(train_loss)/batch_size:.4f},train_Acc:{sum(train_acc)/len(train_acc)/batch_size:.4f},Time: {time_end-time_start}')
    print(f'batch:{step},train_loss:{train_loss/ len(train_loader) / batch_size:.4f},train_Acc:{train_acc/ len(train_loader)/batch_size:.4f},Time: {time_end - time_start}')
    # epoch_loss = running_loss * batch_size / len(train_dataset)
    # epoch_acc = 100 * running_corrects / len(train_dataset)

    #=========================================================================================================================================================
    # model.train(False)
    model.eval()
    with torch.no_grad():
        for step, data in tqdm(enumerate(iter(val_loader))):
            x, y = data
            x, y = x.cuda(), y.cuda()
            # print(f'{y}')

            y_pred = model(x)
            _, pred = torch.max(y_pred, 1)

            loss = loss_function(y_pred, pred)

            acc_count = torch.sum( pred ==y ).item()

            loss=loss.cpu()
            # acc = acc.cpu()
            # test_loss.append(loss.data)
            # test_acc.append(acc)

            test_loss += loss
            test_acc += acc_count
        # print(f'batch:{batch_size},train_loss:{sum(test_loss)/len(test_loss)/batch_size:.4f},train_Acc:{sum(test_acc)/len(test_acc)/batch_size:.4f},Time: {time_end - time_start}')
        # test_acc_value = sum(test_acc)/len(test_acc)/batch_size
        print(f'batch:{step},test_loss:{test_loss/len(val_loader)/batch_size:.4f},test_Acc:{test_acc/len(val_loader)/batch_size:.4f},Time: {time_end - time_start}')
        test_acc_value = test_acc/len(val_loader)/batch_size
        if test_acc_value>best_acc:
            best_acc = test_acc_value
            pth_name= "Acc" + str(best_acc) + ".pth"
            torch.save(model, pth_name)




































































