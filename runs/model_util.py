from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np


def train_model(model, train, valid, callbacks, batch_size, epochs):
    model.fit(x=[train[0], train[1]],
              y=train[2],
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=([valid[0], valid[1]], valid[2]),
              callbacks=callbacks)


def eval_model(model, valid):
    # 预测
    y_pred = model.predict([valid[0], valid[1]], batch_size=64)
    y_pred = np.argmax(y_pred, axis=1)
    y_valid = np.argmax(valid[2], axis=1)

    # 准确率
    accuracy = accuracy_score(y_valid, y_pred)
    precision = precision_score(y_valid, y_pred, average='macro')
    recall = recall_score(y_valid, y_pred, average='macro')
    f1 = f1_score(y_valid, y_pred, average='macro')

    print("accuracy_score = %.2f" % accuracy)
    print("precision_score = %.2f" % precision)
    print("recall_score = %.2f" % recall)
    print("f1_score = %.2f" % f1)


def predict_model(model, test_images, test_visits, test_file_pre_npy_path, result_data_path):
    class_name = {}
    for i in range(9):
        class_name[i] = str(i + 1).zfill(3)

    AreaID = np.load(test_file_pre_npy_path)
    CategoryID = []

    # 预测
    y_pred = model.predict([test_images, test_visits], batch_size=64)
    y_pred = np.argmax(y_pred, axis=1)

    for i in range(len(y_pred)):
        category = class_name[y_pred[i]]
        CategoryID.append(category)

    # 结果写入文件
    with open(result_data_path, "w") as f:
        for i in range(len(CategoryID)):
            area_category_id = AreaID[i] + "\t" + CategoryID[i] + "\n"
            f.write(area_category_id)
