from preprocess.process_images_labels import main_save_file_pre, \
    main_load_train_images_labels, main_load_test_images

from preprocess.process_visits import main as process_visits


def main():
    main_save_file_pre()
    main_load_train_images_labels()
    main_load_test_images()
    process_visits()


if __name__ == "__main__":
    main()
