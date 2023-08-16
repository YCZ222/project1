import torch
import torchvision
import random
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import MyDataset
from models_final import FusionResNet
from tqdm import tqdm
# add tensorboard
####### download "github desktop"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("use GPU:", torch.cuda.is_available())

# hyper-parameter
class_num1 = 27  
class_num2 = 9  
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
train_model = FusionResNet(class_num1=class_num1, class_num2=class_num2)
train_model.to(device)


# learning
loss_fn1 = torch.nn.CrossEntropyLoss()
loss_fn2 = torch.nn.BCELoss()
optimizer = torch.optim.Adam(train_model.parameters(), lr=learning_rate, weight_decay=weight_decay) # Adam, weight_decay=weight_decay

writer = SummaryWriter(log_dir="./Logs/train_joint")


total_train_step = 0
total_eval_step = 0
total_test_step = 0
for epoch in tqdm(range(epochs)):
    train_model.train()
    train_loss = 0.0
    correct_style_train = 0
    correct_emotion_train = 0
    total_style_train = 0
    total_emotion_train = 0

    for images, style_labels, emotion_labels in tqdm(train_loader):
        images = images.to(device)
        style_labels = style_labels.to(device)
        emotion_labels = emotion_labels.to(device)

        outputs1, outputs2 = train_model(images)
        loss1 = loss_fn1(outputs1, style_labels)  # 使用样式标签进行损失计算
        outputs2 = torch.sigmoid(outputs2)
        loss2 = loss_fn2(outputs2, emotion_labels.float())  # 使用情感标签进行损失计算
        loss = loss1 + loss2
        optimizer.zero_grad()
        loss.backward()
        # clip gradient
        torch.nn.utils.clip_grad_norm_(train_model.parameters(), clip_gradient)
        optimizer.step()

        train_loss += loss.item()

        _, predicted_style = outputs1.max(1)
        predicted_emotion = (outputs2 > 0.5).float()  # 修正预测方式
        total_style_train += style_labels.size(0)
        total_emotion_train += emotion_labels.size(0) * emotion_labels.size(1)  # 样本数 x 标签数

        correct_style_train += predicted_style.eq(style_labels).sum().item()
        correct_emotion_train += (predicted_emotion == emotion_labels).sum().item()  # 修正正确预测计数

        writer.add_scalar("Loss/Train", loss.item(), total_train_step)
        total_train_step += 1

    accuracy_style_train = 100.0 * correct_style_train / total_style_train
    accuracy_emotion_train = 100.0 * correct_emotion_train / total_emotion_train
    writer.add_scalar("Accuracy/Train_Style", accuracy_style_train, epoch)  # 记录风格分类准确率到 TensorBoard
    writer.add_scalar("Accuracy/Train_Emotion", accuracy_emotion_train, epoch)  # 记录情感分类准确率到 TensorBoard
    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(train_loader):.4f} - Train Accuracy (Style): {accuracy_style_train:.2f}% - Train Accuracy (Emotion): {accuracy_emotion_train:.2f}%")


    # Evaluation stage
    train_model.eval()
    eval_loss = 0.0
    correct_style_eval = 0
    correct_emotion_eval = 0
    total_style_eval = 0
    total_emotion_eval = 0

    with torch.no_grad():
        for images, style_labels, emotion_labels in tqdm(eval_loader):
            images = images.to(device)
            style_labels = style_labels.to(device)
            emotion_labels = emotion_labels.to(device)

            outputs1, outputs2 = train_model(images)
            loss1 = loss_fn1(outputs1, style_labels)
            outputs2 = torch.sigmoid(outputs2)
            loss2 = loss_fn2(outputs2, emotion_labels.float())
            loss = loss1 + loss2

            eval_loss += loss.item()

            _, predicted_style = outputs1.max(1)
            predicted_emotion = (outputs2 > 0.5).float()  # 修正预测方式
            total_style_eval += style_labels.size(0)
            total_emotion_eval += emotion_labels.size(0) * emotion_labels.size(1)  # 样本数 x 标签数

            correct_style_eval += predicted_style.eq(style_labels).sum().item()
            correct_emotion_eval += (predicted_emotion == emotion_labels).sum().item()  # 修正正确预测计数

            writer.add_scalar("Loss/Eval", loss.item(), total_eval_step)  # Record loss for each step
            total_eval_step += 1  # 递增 total_step

    eval_loss /= len(eval_loader)
    eval_accuracy_style = 100.0 * correct_style_eval / total_style_eval
    eval_accuracy_emotion = 100.0 * correct_emotion_eval / total_emotion_eval
    writer.add_scalar("Accuracy/Eval_Style", eval_accuracy_style, epoch)  # 记录风格分类准确率到 TensorBoard
    writer.add_scalar("Accuracy/Eval_Emotion", eval_accuracy_emotion, epoch)  # 记录情感分类准确率到 TensorBoard
    print(f"Eval Loss: {eval_loss:.4f} - Eval Accuracy (Style): {eval_accuracy_style:.2f}% - Eval Accuracy (Emotion): {eval_accuracy_emotion:.2f}%")



    # Test stage
    train_model.eval()
    test_loss = 0.0
    correct_style_test = 0
    correct_emotion_test = 0
    total_style_test = 0
    total_emotion_test = 0

    with torch.no_grad():
        for images, style_labels, emotion_labels in tqdm(test_loader):
            images = images.to(device)
            style_labels = style_labels.to(device)
            emotion_labels = emotion_labels.to(device)

            outputs1, outputs2 = train_model(images)
            loss1 = loss_fn1(outputs1, style_labels)
            outputs2 = torch.sigmoid(outputs2)
            loss2 = loss_fn2(outputs2, emotion_labels.float())
            loss = loss1 + loss2

            test_loss += loss.item()

            _, predicted_style = outputs1.max(1)
            predicted_emotion = (outputs2 > 0.5).float()  # 修正预测方式
            total_style_test += style_labels.size(0)
            total_emotion_test += emotion_labels.size(0) * emotion_labels.size(1)  # 样本数 x 标签数

            correct_style_test += predicted_style.eq(style_labels).sum().item()
            correct_emotion_test += (predicted_emotion == emotion_labels).sum().item()  # 修正正确预测计数

            writer.add_scalar("Loss/Test", loss.item(), total_test_step)  # Record loss for each step
            total_test_step += 1  # 递增 total_step_test

    test_loss /= len(test_loader)
    test_accuracy_style = 100.0 * correct_style_test / total_style_test
    test_accuracy_emotion = 100.0 * correct_emotion_test / total_emotion_test
    writer.add_scalar("Accuracy/Test_Style", test_accuracy_style, epoch)  # 记录风格分类准确率到 TensorBoard
    writer.add_scalar("Accuracy/Test_Emotion", test_accuracy_emotion, epoch)  # 记录情感分类准确率到 TensorBoard
    print(f"Test Loss: {test_loss:.4f} - Test Accuracy (Style): {test_accuracy_style:.2f}% - Test Accuracy (Emotion): {test_accuracy_emotion:.2f}%")
    
    model_save_path = f"./models/train_joint/model_epoch_{epoch+1}.pth"
    torch.save(train_model.state_dict(), model_save_path)
    print(f"Train model saved at {model_save_path}")

writer.close()

