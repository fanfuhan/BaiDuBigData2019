import os
import inspect


def mkdir_if_not_exist(dir_list):
    for directory in dir_list:
        if not os.path.exists(directory):
            os.makedirs(directory)


curr_filename = inspect.getfile(inspect.currentframe())
root_dir = os.path.dirname(os.path.abspath(curr_filename))

origin_data_path = os.path.join(root_dir, 'origin_data')
data_path = os.path.join(root_dir, 'data')
data_train_path = os.path.join(data_path, 'train')
data_test_path = os.path.join(data_path, 'test')

train_image_path = os.path.join(origin_data_path, 'train_image_aug')
train_visit_path = os.path.join(origin_data_path, 'train_visit')
test_image_path = os.path.join(origin_data_path, 'test_image_aug')
test_visit_path = os.path.join(origin_data_path, 'test_visit')

train_images_npy_path = os.path.join(data_train_path, 'train_images_data.npy')
train_labels_npy_path = os.path.join(data_train_path, 'train_labels.npy')
train_visits_npy_path = os.path.join(data_train_path, 'train_visits.npy')
train_visits_274_npy_path = os.path.join(data_train_path, 'train_visits_274.npy')
train_visits_224_npy_path = os.path.join(data_train_path, 'train_visits_224.npy')
train_file_pre_npy_path = os.path.join(data_train_path, 'train_file_pre.npy')

test_images_npy_path = os.path.join(data_test_path, 'test_images_data.npy')
test_visits_npy_path = os.path.join(data_test_path, 'test_visits.npy')
test_visits_274_npy_path = os.path.join(data_test_path, 'test_visits_274.npy')
test_visits_224_npy_path = os.path.join(data_test_path, 'test_visits_224.npy')
test_file_pre_npy_path = os.path.join(data_test_path, 'test_file_pre.npy')
result_data_path = os.path.join(data_test_path, 'result_data.txt')

model_path = os.path.join(root_dir, 'model_data', 'model.h5')
model_path_visit = os.path.join(root_dir, 'model_data', 'model_visit.h5')
model_path_image = os.path.join(root_dir, 'model_data', 'model_image.h5')



