import os
import sys
sys.path.append('/home/LiZhongYu/data/jht/BaiDuBigData2019/')

import numpy as np
from skimage import io
from tqdm import tqdm

from paths import mkdir_if_not_exist
from paths import data_path, data_train_path, data_test_path
from paths import train_image_path, test_image_path
from paths import train_file_pre_npy_path, test_file_pre_npy_path
from paths import train_images_npy_path, train_labels_npy_path, test_images_npy_path

# 不存在则创建
mkdir_if_not_exist(dir_list=[data_path, data_train_path, data_test_path])

# 定义标签
label_001 = [1, 0, 0, 0, 0, 0, 0, 0, 0]
label_002 = [0, 1, 0, 0, 0, 0, 0, 0, 0]
label_003 = [0, 0, 1, 0, 0, 0, 0, 0, 0]
label_004 = [0, 0, 0, 1, 0, 0, 0, 0, 0]
label_005 = [0, 0, 0, 0, 1, 0, 0, 0, 0]
label_006 = [0, 0, 0, 0, 0, 1, 0, 0, 0]
label_007 = [0, 0, 0, 0, 0, 0, 1, 0, 0]
label_008 = [0, 0, 0, 0, 0, 0, 0, 1, 0]
label_009 = [0, 0, 0, 0, 0, 0, 0, 0, 1]

# 定义对应关系
id_label_map = {
    '001': label_001,
    '002': label_002,
    '003': label_003,
    '004': label_004,
    '005': label_005,
    '006': label_006,
    '007': label_007,
    '008': label_008,
    '009': label_009
}


def load_file_pre(image_path, file_pre_npy_path):
    if os.path.exists(image_path) and os.path.exists(file_pre_npy_path):
        file_pre = np.load(file_pre_npy_path)
    else:
        file_pre = []
        print(image_path)
        for img_fname in tqdm(os.listdir(image_path)):
            # 提取文件名前缀
            img_fname_pre = img_fname.rsplit('.', maxsplit=1)[0]
            file_pre.append(img_fname_pre)

        # file_pre = [img_fname.rsplit('.', maxsplit=1)[0] for img_fname in tqdm(os.listdir(image_path))]

        file_pre.sort()
        np.save(file_pre_npy_path, file_pre)

    return file_pre


def load_train():
    images = []
    labels = []

    file_pre = np.load(train_file_pre_npy_path)
    for i in tqdm(range(len(file_pre))):
        # 获取图像标签
        label = id_label_map.get(file_pre[i].split('_')[1])

        # 读取图片
        img_fname = file_pre[i] + '.jpg'
        img_fname_path = os.path.join(train_image_path, img_fname)
        image = io.imread(img_fname_path)
        image = image.astype(np.uint8)

        images.append(image)
        labels.append(label)

    images = np.stack(images).astype(np.uint8)
    labels = np.stack(labels, axis=0)

    return images, labels


def load_train_images_labels():
    if os.path.exists(train_images_npy_path) and os.path.exists(train_labels_npy_path):
        images = np.load(train_images_npy_path)
        labels = np.load(train_labels_npy_path)
    else:
        images, labels = load_train()

        np.save(train_images_npy_path, images)
        np.save(train_labels_npy_path, labels)

    return images, labels


def load_test():
    images = []

    file_pre = np.load(test_file_pre_npy_path)
    for i in tqdm(range(len(file_pre))):
        # 读取图片
        img_fname = file_pre[i] + '.jpg'
        img_fname_path = os.path.join(test_image_path, img_fname)
        image = io.imread(img_fname_path)
        image = image.astype(np.uint8)

        images.append(image)

    images = np.stack(images).astype(np.uint8)

    return images


def load_test_images():
    if os.path.exists(test_images_npy_path):
        images = np.load(test_images_npy_path)
    else:
        images = load_test()

        np.save(test_images_npy_path, images)

    return images

def main_save_file_pre():
    # 提取文件名前缀ID，并排序保存
    train_file_pre = load_file_pre(train_image_path, train_file_pre_npy_path)
    test_file_pre = load_file_pre(test_image_path, test_file_pre_npy_path)
    print(train_file_pre.shape)
    print(train_file_pre[:10])
    print(test_file_pre.shape)
    print(test_file_pre[:10])


def main_load_train_images_labels():
    images, labels = load_train_images_labels()
    print(images.shape)
    print(labels.shape)


def main_load_test_images():
    images = load_test_images()
    print(images.shape)


if __name__ == "__main__":
    main_save_file_pre()
    main_load_train_images_labels()
    main_load_test_images()
