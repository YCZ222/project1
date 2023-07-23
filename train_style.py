import torch
import torchvision
import random
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import MyDataset
from models import ResNet 
from models import ResNet_fix
from tqdm import tqdm
# add tensorboard
####### download "github desktop"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("use GPU:", torch.cuda.is_available())

# hyper-parameter
class_num = 27  
epochs = 10 
batch_size = 64
seed = 1234
clip_gradient = 5.0
learning_rate = 1e-3
weight_decay = 0.01

# 设置随机种子
def seed_all(seed_value):
    random.seed(seed_value) # Python
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu vars
    
    if torch.cuda.is_available(): 
        print ('CUDA is available')
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        #torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = True
seed_all(seed)

# dataset
train_data = "/home/acer/文档/project/art_images_classification/project1/train.txt"
eval_data = "/home/acer/文档/project/art_images_classification/project1/eval.txt"
test_data = "/home/acer/文档/project/art_images_classification/project1/test.txt"
csvpath = "/home/acer/文档/project/art_images_classification/project1/artemis_dataset_release_v0.csv"
train_dataset = MyDataset(csv_path=csvpath, img_txt=train_data)
eval_dataset = MyDataset(csv_path=csvpath, img_txt=eval_data)
test_dataset = MyDataset(csv_path=csvpath, img_txt=test_data)
# split train/val/test  60000+/5000/5000
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
eval_loader = DataLoader(eval_dataset,batch_size=batch_size,shuffle=False)
test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)


# model
train_model = ResNet(class_num=class_num)
train_model.to(device)


# learning
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(train_model.parameters(), lr=learning_rate, weight_decay=weight_decay) # Adam, weight_decay=weight_decay

writer = SummaryWriter(log_dir="logs1")


for epoch in tqdm(range(epochs)):
    train_model.train()
    train_loss = 0.0

    for images, style_labels, _ in tqdm(train_loader):  # 修改这里的变量名，不使用情感分类标签
        images = images.to(device)
        style_labels = style_labels.to(device)
        outputs = train_model(images)
        loss = loss_fn(outputs, style_labels)  # 使用样式标签进行损失计算
        optimizer.zero_grad()
        loss.backward()
        # clip gradient
        torch.nn.utils.clip_grad_norm_(train_model.parameters(), clip_gradient)
        optimizer.step()

        train_loss += loss.item()

    # add val/test stage
    # model.eval()
    # for images, labels in eval_loader:
    writer.add_scalar("Loss/Train", train_loss, epoch)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss/len(train_loader):.4f}")
    
    # Evaluation stage
    train_model.eval()
    eval_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, style_labels, _ in tqdm(eval_loader):  # 修改这里的变量名，不使用情感分类标签
            images = images.to(device)
            style_labels = style_labels.to(device)

            outputs = train_model(images)
            loss = loss_fn(outputs, style_labels)  # 使用样式标签进行损失计算
            eval_loss += loss.item()

            _, predicted = outputs.max(1)
            total += style_labels.size(0)
            correct += predicted.eq(style_labels).sum().item()


    eval_loss /= len(eval_loader)
    eval_accuracy = 100.0 * correct / total
    writer.add_scalar("Loss/Eval", eval_loss, epoch)
    writer.add_scalar("Accuracy/Eval", eval_accuracy, epoch)
    print(f"Eval Loss: {eval_loss:.4f} - Eval Accuracy: {eval_accuracy:.2f}%")

    
    # Test stage
    train_model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, style_labels, _ in tqdm(test_loader):  # 修改这里的变量名，不使用情感分类标签
            images = images.to(device)
            style_labels = style_labels.to(device)

            outputs = train_model(images)
            loss = loss_fn(outputs, style_labels)  # 使用样式标签进行损失计算
            test_loss += loss.item()

            _, predicted = outputs.max(1)
            total += style_labels.size(0)  # 使用样式标签的大小
            correct += predicted.eq(style_labels).sum().item()  # 使用样式标签进行预测

    test_loss /= len(test_loader)
    test_accuracy = 100.0 * correct / total
    writer.add_scalar("Loss/Test", test_loss, epoch)
    writer.add_scalar("Accuracy/Test", test_accuracy, epoch)
    print(f"Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.2f}%")
writer.close()
