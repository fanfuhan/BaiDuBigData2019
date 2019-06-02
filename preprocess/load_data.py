import sys

sys.path.append('/home/LiZhongYu/data/jht/BaiDuBigData2019/')

from paths import train_images_npy_path, train_labels_npy_path
from paths import train_visits_npy_path, train_visits_274_npy_path, train_visits_224_npy_path
from paths import test_images_npy_path, test_visits_npy_path, test_visits_274_npy_path, test_visits_224_npy_path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

np.random.seed(4321)

# 加载数据
X_images = np.load(train_images_npy_path)
visits = np.load(train_visits_npy_path)
visits_274 = np.load(train_visits_274_npy_path)
visits_224 = np.load(train_visits_224_npy_path)
y_labels = np.load(train_labels_npy_path)
N = visits.shape[0]

# visits数据改变形状并拼接
visits = np.reshape(visits, (N, -1))
visits_274 = np.reshape(visits_274, (N, -1))
visits_224 = np.reshape(visits_224, (N, -1))
X_visits = np.concatenate((visits, visits_274, visits_224), axis=1)

# 划分训练集和验证集
X_train_images, X_valid_images, X_train_visits, X_valid_visits, y_train, y_valid = train_test_split(
    X_images, X_visits, y_labels, test_size=0.1, random_state=4321, shuffle=True
)

# 归一化
X_train_images = X_train_images / 255.0
X_valid_images = X_valid_images / 255.0
scalar = MinMaxScaler()
X_train_visits = scalar.fit_transform(X_train_visits)
X_valid_visits = scalar.transform(X_valid_visits)


def load_train_data():
    train = (X_train_images, X_train_visits, y_train)
    valid = (X_valid_images, X_valid_visits, y_valid)

    return train, valid


def load_test_data():
    # 加载测试数据
    X_images_test = np.load(test_images_npy_path)
    visits_test = np.load(test_visits_npy_path)
    visits_274_test = np.load(test_visits_274_npy_path)
    visits_224_test = np.load(test_visits_224_npy_path)
    N = visits_test.shape[0]

    # visits数据改变形状并拼接
    visits_test = np.reshape(visits_test, (N, -1))
    visits_274_test = np.reshape(visits_274_test, (N, -1))
    visits_224_test = np.reshape(visits_224_test, (N, -1))
    X_visits_test = np.concatenate((visits_test, visits_274_test, visits_224_test), axis=1)

    # 归一化
    X_images_test = X_images_test / 255.0
    X_visits_test = scalar.transform(X_visits_test)

    return X_images_test, X_visits_test


def main():
    X_images_test, X_visits_test = load_test_data()
    print(X_images_test.shape, X_visits_test.shape)


if __name__ == "__main__":
    main()
