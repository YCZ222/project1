import torch
import torch.nn as nn
import torchvision.models
import torch.nn.functional as F
from transformers import ViTModel

class ResNet(nn.Module):
    def __init__(self, class_num):
        super(ResNet, self).__init__()
        self.model = torchvision.models.resnet18(pretrained=True) # 加载预训练好的resnet18
        self.model= nn.Sequential(*list(self.model.children())[:-1])        # 去掉最后一层原线性层
        self.mlp = nn.Linear(512, class_num) # 定义一个线性层，将维度降到class_num

    def forward(self, image):
        x = self.model(image) # 输出feature map
        x = torch.flatten(x, 1) # 将feature map展平为一维向量
        x = self.mlp(x) # 输入到线性层
        x = F.softmax(x, dim=1) # 返回类别的概率分布
        return x

class ResNet_multi(nn.Module):
    def __init__(self, class_num):
        super(ResNet_multi, self).__init__()
        self.model = torchvision.models.resnet18(pretrained=True) # 加载预训练好的resnet18
        self.model = nn.Sequential(*list(self.model.children())[:-1])        # 去掉最后一层原线性层
        self.mlp = nn.Linear(512, class_num) # 定义一个线性层，将维度降到class_num

    def forward(self, image):
        x = self.model(image) # 输出feature map
        x = torch.flatten(x, 1) # 将feature map展平为一维向量
        x = self.mlp(x) # 输入到线性层
        x = torch.sigmoid(x) # 返回类别的概率分布
        return x


class ResNet_fix(nn.Module):
    def __init__(self, class_num):
        super(ResNet_fix, self).__init__()
        self.model = torchvision.models.resnet18(pretrained=True) # 加载预训练好的resnet18
        self.model= nn.Sequential(*list(self.model.children())[:-1])       # 去掉最后一层原线性层
        self.mlp = nn.Linear(512, class_num) # 定义一个线性层，将维度降到class_num

    def forward(self, image):
        with torch.no_grad():
            x = self.model(image) # 输出feature map
        x = torch.flatten(x, 1) # 将feature map展平为一维向量
        x = self.mlp(x) # 输入到线性层
        x = F.softmax(x, dim=1) # 返回类别的概率分布
        return x

class ViT(nn.Module):
    def __init__(self, class_num):
        super(ViT, self).__init__()
        self.model = ViTModel.from_pretrained("google/vit-base-patch16-224") # 加载预训练好的ViT
        self.mlp = nn.Linear(self.model.config.hidden_size, class_num) # 定义一个线性层，将维度降到class_num

    def forward(self, image):
        x = self.model(image).last_hidden_state[:,0,:] # 输出feature map    batch_size, seq_len(1+49), 768 => batch_size, 768  
        x = self.mlp(x) # 输入到线性层
        x = F.softmax(x, dim=1) # 返回类别的概率分布
        return x

class ViT_fix(nn.Module):
    def __init__(self, class_num):
        super(ViT_fix, self).__init__()
        self.model = ViTModel.from_pretrained("google/vit-base-patch16-224") # 加载预训练好的ViT
        self.mlp = nn.Linear(self.model.config.hidden_size, class_num) # 定义一个线性层，将维度降到class_num

    def forward(self, image):
        with torch.no_grad():
            x = self.model(image).last_hidden_state[:,0,:] # 输出feature map
        x = self.mlp(x) # 输入到线性层
        x = F.softmax(x, dim=1) # 返回类别的概率分布
        return x