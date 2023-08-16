import torch
import torchvision
import random
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import MyDataset
from models_final import ResNet_multi 
from tqdm import tqdm
# add tensorboard
####### download "github desktop"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("use GPU:", torch.cuda.is_available())

# hyper-parameter
class_num = 9  
epochs = 7 
batch_size = 64
seed = 1234
clip_gradient = 5.0
learning_rate = 1e-4
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
train_model = ResNet_multi(class_num=class_num)
train_model.to(device)


# learning
loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.Adam(train_model.parameters(), lr=learning_rate, weight_decay=weight_decay) # Adam, weight_decay=weight_decay

writer = SummaryWriter(log_dir="./Logs/train_emotion")

total_train_step = 0
total_eval_step = 0
total_test_step = 0

for epoch in tqdm(range(epochs)):
    train_model.train()
    train_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, _, emotion_labels in tqdm(train_loader):
        images = images.to(device)
        emotion_labels = emotion_labels.to(device)
        outputs = train_model(images)
        outputs = torch.sigmoid(outputs)
        loss = loss_fn(outputs, emotion_labels.float())  # 使用情感标签进行损失计算
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(train_model.parameters(), clip_gradient)
        optimizer.step()

        train_loss += loss.item()

        predicted = (outputs > 0.5).float()  # 使用阈值 0.5 进行二值化预测
        total_train += emotion_labels.size(0) * emotion_labels.size(1)  # 样本数 x 标签数
        correct_train += (predicted == emotion_labels).sum().item()

        writer.add_scalar("Loss/Train", loss.item(), total_train_step)
        total_train_step += 1

    accuracy_train = 100.0 * correct_train / total_train
    writer.add_scalar("Accuracy/Train", accuracy_train, epoch)  # 记录准确率到TensorBoard
    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(train_loader):.4f} - Train Accuracy: {accuracy_train:.2f}%")



    train_model.eval()
    eval_loss = 0.0
    correct = 0
    total_eval = 0

    with torch.no_grad():
        for images, _, emotion_labels in tqdm(eval_loader):
            images = images.to(device)
            emotion_labels = emotion_labels.to(device)

            outputs = train_model(images)
            outputs = torch.sigmoid(outputs) 
            loss = loss_fn(outputs, emotion_labels.float())
            eval_loss += loss.item()

            predicted = (outputs > 0.5).float()  # 使用阈值 0.5 进行二值化预测
            total_eval += emotion_labels.size(0) * emotion_labels.size(1)  # 样本数 x 标签数
            correct += (predicted == emotion_labels).sum().item()

            writer.add_scalar("Loss/Eval", loss.item(), total_eval_step)  # 记录loss到TensorBoard
            total_eval_step += 1  # 递增 total_step

    eval_loss /= len(eval_loader)
    eval_accuracy = 100.0 * correct / total_eval
    writer.add_scalar("Accuracy/Eval", eval_accuracy, epoch)  # 记录准确率到TensorBoard
    print(f"Eval Loss: {eval_loss:.4f} - Eval Accuracy: {eval_accuracy:.2f}%")


    # 添加 test 阶段
    train_model.eval()
    test_loss = 0.0
    correct = 0
    total_test = 0


    with torch.no_grad():
        for images, _, emotion_labels in tqdm(test_loader):
            images = images.to(device)
            emotion_labels = emotion_labels.to(device)

            outputs = train_model(images)
            outputs = torch.sigmoid(outputs) 
            loss = loss_fn(outputs, emotion_labels.float())
            test_loss += loss.item()

            predicted = (outputs > 0.5).float()  # 使用阈值 0.5 进行二值化预测
            total_test += emotion_labels.size(0) * emotion_labels.size(1)  # 样本数 x 标签数
            correct += (predicted == emotion_labels).sum().item()

            writer.add_scalar("Loss/Test", loss.item(), total_test_step)  # 记录loss到TensorBoard
            total_test_step += 1  # 递增 total_step_test

    test_loss /= len(test_loader)
    test_accuracy = 100.0 * correct / total_test
    writer.add_scalar("Accuracy/Test", test_accuracy, epoch)  # 记录准确率到TensorBoard
    print(f"Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.2f}%")
    
    model_save_path = f"./models/train_emotion/model_epoch_{epoch+1}.pth"
    torch.save(train_model.state_dict(), model_save_path)
    print(f"Train model saved at {model_save_path}")
writer.close()

