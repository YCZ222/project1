import os
import random

source_folder = "/home/acer/文档/project/art_images_classification/project1/artemis-download"
train_txt = "/home/acer/文档/project/art_images_classification/project1/train.txt"
eval_txt = "/home/acer/文档/project/art_images_classification/project1/eval.txt"
test_txt = "/home/acer/文档/project/art_images_classification/project1/test.txt"
train_ratio = 0.8
eval_ratio = 0.1
test_ratio = 0.1

# 获取文件夹中所有图片的路径
image_files = [os.path.join(source_folder, filename) for filename in os.listdir(source_folder) if filename.endswith('.jpg')]

# 计算每个数据集的图片数量
num_images = len(image_files)
num_train = int(num_images * train_ratio)
num_eval = int(num_images * eval_ratio)
num_test = int(num_images * test_ratio)

# 随机打乱图片列表
random.shuffle(image_files)

# 将图片路径写入train.txt文件中
with open(train_txt, 'w') as train_file:
    for i, image_file in enumerate(image_files[:num_train]):
        train_file.write(image_file + '\n')

# 将图片路径写入eval.txt文件中
with open(eval_txt, 'w') as eval_file:
    for i, image_file in enumerate(image_files[num_train:num_train+num_eval]):
        eval_file.write(image_file + '\n')

# 将图片路径写入test.txt文件中
with open(test_txt, 'w') as test_file:
    for i, image_file in enumerate(image_files[num_train+num_eval:num_train+num_eval+num_test]):
        test_file.write(image_file + '\n')

print("Finish")
