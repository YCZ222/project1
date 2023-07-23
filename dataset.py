import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os


class MyDataset(Dataset):
    def __init__(self, csv_path, img_txt):
        with open(img_txt, 'r') as f:
            lines = f.readlines()
        self.img_list = [line.strip() for line in lines]  # 读取所有图片路径
        self.img_paths = [os.path.join("artemis-download", os.path.basename(img_path)) for img_path in self.img_list]
        self.img_txt = img_txt
        self.style_label_mapping = {
        'Abstract_Expressionism':0, 'Action_painting':1, 'Analytical_Cubism':2,
        'Art_Nouveau_Modern':3, 'Baroque':4, 'Color_Field_Painting':5,
        'Contemporary_Realism':6, 'Cubism':7, 'Early_Renaissance':8, 'Expressionism':9,
        'Fauvism':10, 'High_Renaissance':11, 'Impressionism':12, 'Mannerism_Late_Renaissance':13,
        'Minimalism':14, 'Naive_Art_Primitivism':15, 'New_Realism':16, 'Northern_Renaissance':17,
        'Pointillism':18, 'Pop_Art':19, 'Post_Impressionism':20, 'Realism':21, 'Rococo':22,
        'Romanticism':23, 'Symbolism':24, 'Synthetic_Cubism':25, 'Ukiyo_e':26
        }
        self.emotion_label_mapping = {
        'something else':0,'sadness':1,'contentment':2,'awe':3, 'amusement':4,'excitement':5,
        'fear':6,'disgust':7, 'anger':8
        }
        data = pd.read_csv(csv_path)
        self.style_dict = {}
        self.emotion_dict = {}
        for index, row in data.iterrows():
            art_style = row[0]
            painting = row[1] + ".jpg"
            emotion = row[2]
            if not os.path.exists(os.path.join("artemis-download", painting)):
                continue
            if painting not in self.emotion_dict:
                self.emotion_dict[painting] = []
            self.emotion_dict[painting] = list(set(self.emotion_dict[painting] + [emotion]))
            self.style_dict[painting] = art_style

        self.transform_train = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),  # 添加 Random Crop 操作
            transforms.ToTensor(),
        ])
        self.transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        try:
            img = Image.open(img_path).convert('RGB')
        except OSError:
            print(f"Skipping corrupted image: {img_path}")
            index = (index + 1) % len(self.img_list)  # 跳到下一个图像
            return self.__getitem__(index)

        if self.img_txt == "train.txt":
            transform = self.transform_train  # 在训练集中应用 Random Crop
        else:
            transform = self.transform_test  # 在评估集和测试集中不应用 Random Crop

        if transform is not None:
            img = transform(img)

        label1_str = self.style_dict[os.path.basename(img_path)]
        label1 = self.style_label_mapping[label1_str]
        label1 = torch.tensor(label1, dtype=torch.long)

        
        painting = os.path.basename(img_path)
        emotions = self.emotion_dict.get(painting, [])  # 如果图像无对应情绪标签，则返回空列表

        # 将每个情感类别转换为二进制标签
        label2 = torch.zeros(len(self.emotion_label_mapping), dtype=torch.float32)
        for emotion in emotions:
            index = self.emotion_label_mapping[emotion]
            label2[index] = 1.0

        return img, label1, label2

    def __len__(self):
        return len(self.img_list)


