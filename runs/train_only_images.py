import sys
sys.path.append('/home/LiZhongYu/data/jht/BaiDuBigData2019/')
import keras
from keras.applications.densenet import DenseNet121
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input, Dense
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from set_gpu import set_gpu
from paths import model_path
from callbacks import set_callbacks

# 超参数
num_classes = 9
batch_size = 64
epochs = 200


def build_model():
    image_input_tensor = Input(shape=(100, 100, 3))
    image_base_model = DenseNet121(input_tensor=image_input_tensor, weights=None, include_top=False)
    img_input = image_base_model.output

    img_output = GlobalAveragePooling2D(name='avg_pool_my_1')(img_input)
    output = Dense(num_classes, activation='softmax', name='prediction', kernel_initializer='he_normal')(img_output)

    model = Model(inputs=image_input_tensor, outputs=output)

    model.summary()

    keras.utils.plot_model(model, to_file='model_images.png', show_shapes=True, show_layer_names=True, rankdir='TB')

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc', ])

    return model


def train_model(model, callbacks):
    train_image = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.1, rotation_range=40,
                                     width_shift_range=0.2, height_shift_range=0.2,
                                     shear_range=0.2, horizontal_flip=True)
    train_generator = train_image.flow_from_directory('../origin_data/train_image', target_size=(100, 100),
                                                      batch_size=batch_size, subset="training")
    valid_generator = train_image.flow_from_directory('../origin_data/train_image', target_size=(100, 100),
                                                      batch_size=batch_size, subset="validation")

    model.fit_generator(train_generator, steps_per_epoch=300, validation_steps=30, epochs=epochs,
                        validation_data=valid_generator, callbacks=callbacks)


def main():
    # 指定GPU
    set_gpu()

    # 构建模型
    model = build_model()

    # 训练模型
    callbacks = set_callbacks(model_path)
    train_model(model, callbacks)


if __name__ == "__main__":
    main()
