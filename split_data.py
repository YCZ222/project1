import os
import random
import shutil

source_folder = "artemis-download"
train_folder = "train_data"
eval_folder = "eval_data"
test_folder = "test_data"
train_ratio = 0.8
eval_ratio = 0.1
test_ratio = 0.1




def split_dataset(source_folder, train_folder, eval_folder, test_folder, train_ratio, eval_ratio, test_ratio):
    # 获取文件夹中所有图片的路径
    image_files = [os.path.join(source_folder, filename) for filename in os.listdir(source_folder) if filename.endswith('.jpg')]

    # 计算每个数据集的图片数量
    num_images = len(image_files)
    num_train = int(num_images * train_ratio)
    num_eval = int(num_images * eval_ratio)
    unm_test = int(num_images * test_ratio)

    # 随机打乱图片列表
    random.shuffle(image_files)

    # 拷贝图片到对应的数据集文件夹
    for i, image_file in enumerate(image_files):
        if i < num_train:
            shutil.copy(image_file, train_folder)
        elif i < num_train + num_eval:
            shutil.copy(image_file, eval_folder)
        else:
            shutil.copy(image_file, test_folder)


split_dataset(source_folder="artemis-download",train_folder="train_data",eval_folder="eval_data",test_folder="test_data",train_ratio=0.8,eval_ratio=0.1,test_ratio=0.1)
print("finish")


