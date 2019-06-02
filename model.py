import keras
from keras.applications.densenet import DenseNet121
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input, Dense
from keras.models import Model


def build_model(num_classes):
    # 分支一：图像输入
    image_input_tensor = Input(shape=(100, 100, 3))
    image_base_model = DenseNet121(input_tensor=image_input_tensor, weights=None, include_top=False)
    img_input = image_base_model.output

    img_output = GlobalAveragePooling2D(name='avg_pool_my_1')(img_input)
    img_output = Dense(128, name='fc_image', kernel_initializer='he_normal', activation='elu')(img_output)

    # 分支二：用户到访特征输入
    visit_input = Input(shape=(4440,))
    visit_output = Dense(1024, kernel_initializer='he_normal', activation='elu')(visit_input)
    visit_output = Dense(256, kernel_initializer='he_normal', activation='elu')(visit_output)

    # 两分支连接
    output = keras.layers.concatenate([img_output, visit_output])
    output = Dense(num_classes, activation='softmax', name='prediction',kernel_initializer='he_normal')(output)

    model = Model(inputs=[image_input_tensor, visit_input], outputs=output)

    model.summary()

    # keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True, rankdir='TB')

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc', ])

    return model
