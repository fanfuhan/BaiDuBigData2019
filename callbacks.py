from keras.callbacks import ModelCheckpoint, TensorBoard


def set_callbacks(model_path):
    # 定义回调函数
    callbacks = [
        # 当标准评估停止提升时，降低学习速率
        # ReduceLROnPlateau(monitor='val_loss',
        #                   factor=0.25,
        #                   patience=2,
        #                   verbose=1,
        #                   mode='auto',
        #                   min_lr=1e-7),
        # 在每个训练期之后保存模型，最后保存的是最佳模型

        ModelCheckpoint(model_path,
                        monitor='val_loss',
                        save_best_only=True,
                        verbose=True),

        # tensorboard 可视化
        TensorBoard(log_dir='../logs',
                    histogram_freq=0,
                    write_graph=False,
                    write_grads=False,
                    write_images=False,
                    update_freq='epoch')
    ]

    return callbacks
