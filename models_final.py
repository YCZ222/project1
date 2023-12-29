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
        # x = F.softmax(x, dim=1) # 返回类别的概率分布
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
        # x = torch.sigmoid(x) # 返回类别的概率分布
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
        # x = F.softmax(x, dim=1) # 返回类别的概率分布
        return x

class ViT(nn.Module):
    def __init__(self, class_num):
        super(ViT, self).__init__()
        self.model = ViTModel.from_pretrained("google/vit-base-patch16-224") # 加载预训练好的ViT
        self.mlp = nn.Linear(self.model.config.hidden_size, class_num) # 定义一个线性层，将维度降到class_num

    def forward(self, image):
        x = self.model(image).last_hidden_state[:,0,:] # 输出feature map    batch_size, seq_len(1+49), 768 => batch_size, 768  
        x = self.mlp(x) # 输入到线性层
        # x = F.softmax(x, dim=1) # 返回类别的概率分布
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
        # x = F.softmax(x, dim=1) # 返回类别的概率分布
        return x


class FusionResNet(nn.Module):
    def __init__(self, class_num1, class_num2):
        super(FusionResNet, self).__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-3])
        
        # Task 1
        self.conv1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.relu1 = nn.ReLU(inplace=True)
        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.attn1 = nn.Linear(1024, 512)
        self.mlp1 = nn.Linear(512, class_num1)
        
        # Task 2
        self.conv2 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        self.relu2 = nn.ReLU(inplace=True)
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.attn2 = nn.Linear(1024, 512)
        self.mlp2 = nn.Linear(512, class_num2)
        
    def forward(self, image):
        x = self.model(image) # shared feature map
        
        # Task 1
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.avgpool1(x1)
        
        # Task 2
        x2 = self.conv2(x)
        x2 = self.bn2(x2)
        x2 = self.relu2(x2)
        x2 = self.avgpool2(x2)

        x1 = x1.squeeze()
        x2 = x2.squeeze()

        # Attention mechanism for task 1
        atten1 = torch.sigmoid(self.attn1(torch.cat([x1, x2], -1)))
        atten1 = x1 * atten1 + x2 * (1-atten1)

        # Attention mechanism for task 2
        atten2 = torch.sigmoid(self.attn2(torch.cat([x1, x2], -1)))
        atten2 = x2 * atten2 + x1 * (1-atten2)

        out_task_11 = self.mlp1(atten1) # output for task 11
        out_task_12 = self.mlp2(atten2) # output for task 12
  
        return out_task_11, out_task_12


class VGG_16(nn.Module):
    def __init__(self, class_num):
        super(VGG_16, self).__init__()
        self.model = torchvision.models.vgg16(pretrained=True) # 加载预训练好的VGG16
        self.features = self.model.features  # 提取VGG16的特征部分
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))  # 替换原来的平均池化层
        self.mlp = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, class_num)
        )  # 定义一个多层感知机

    def forward(self, image):
        x = self.features(image)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.mlp(x)
        return x 


class VGG_19(nn.Module):
    def __init__(self, class_num):
        super(VGG_19, self).__init__()
        self.model = torchvision.models.vgg19(pretrained=True)  # 加载预训练好的VGG19
        self.features = self.model.features  # 提取VGG19的特征部分
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))  # 替换原来的平均池化层
        self.mlp = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, class_num)
        )  # 定义一个多层感知机

    def forward(self, image):
        x = self.features(image)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.mlp(x)
        return x 
    
class MobileNetV2(nn.Module):
    def __init__(self, class_num):
        super(MobileNetV2, self).__init__()
        self.model = torchvision.models.mobilenet_v2(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.mlp = nn.Linear(1280, class_num)

    def forward(self, image):
        x = self.model(image)
        x = x.mean([2, 3])  # 对特征图的高度和宽度维度取平均
        x = self.mlp(x)
        return x
    

'''''
class MobileNetV2_joint(nn.Module):
    def __init__(self, class_num1, class_num2):
        super(MobileNetV2_joint, self).__init__()
        self.model = torchvision.models.mobilenet_v2(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        
        self.class_num1 = class_num1
        self.class_num2 = class_num2
        
        # 添加两个输出层
        self.mlp1 = nn.Linear(1280, class_num1)
        self.mlp2 = nn.Linear(1280, class_num2)
        
    def forward(self, image1, image2):
        x1 = self.model(image1) # shared feature map for image 1
        x2 = self.model(image2) # shared feature map for image 2

        x1 = x1.mean([2, 3])  # 对特征图的高度和宽度维度取平均
        x2 = x2.mean([2, 3])  # 对特征图的高度和宽度维度取平均
        
        # 分别通过两个输出层
        out1 = self.mlp1(x1)
        out2 = self.mlp2(x2)

        return out1, out2 
'''''

    


class MobileNetV2_joint(nn.Module):
    def __init__(self, class_num1, class_num2):
        super(MobileNetV2_joint, self).__init__()
        self.style_model = torchvision.models.mobilenet_v2(pretrained=True)
        self.emotion_model = torchvision.models.mobilenet_v2(pretrained=True)
        
        # 修改模型结构，去掉最后的分类层
        self.style_model = nn.Sequential(*list(self.style_model.children())[:-1])
        self.emotion_model = nn.Sequential(*list(self.emotion_model.children())[:-1])
        
        # 风格分类任务的线性层
        self.style_mlp = nn.Linear(1280, class_num1)
        
        # 情感分类任务的线性层
        self.emotion_mlp = nn.Linear(1280, class_num2)

    def forward(self, image):
        # 风格分类任务
        style_x = self.style_model(image)
        style_x = style_x.mean([2, 3])  # 对特征图的高度和宽度维度取平均
        style_x = self.style_mlp(style_x)
        
        # 情感分类任务
        emotion_x = self.emotion_model(image)
        emotion_x = emotion_x.mean([2, 3])  # 对特征图的高度和宽度维度取平均
        emotion_x = self.emotion_mlp(emotion_x)
        
        return style_x, emotion_x