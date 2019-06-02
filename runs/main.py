import sys
sys.path.append('/home/LiZhongYu/data/jht/BaiDuBigData2019/')

from preprocess.load_data import load_train_data, load_test_data
from paths import test_file_pre_npy_path, result_data_path
from set_gpu import set_gpu
from model import build_model
from paths import model_path
from callbacks import set_callbacks
from runs.model_util import train_model, eval_model, predict_model

# 超参数
num_classes = 9
batch_size = 64
epochs = 200


def main():
    # 指定GPU
    set_gpu()

    # 加载数据
    train, valid = load_train_data()
    test_images, test_visits = load_test_data()

    # 构建模型
    model = build_model(num_classes)

    # 训练模型
    callbacks = set_callbacks(model_path)
    train_model(model, train, valid, callbacks, batch_size, epochs)

    # 评估模型
    eval_model(model, valid)

    # 预测结果
    predict_model(model, test_images, test_visits, test_file_pre_npy_path, result_data_path)


if __name__ == "__main__":
    main()
