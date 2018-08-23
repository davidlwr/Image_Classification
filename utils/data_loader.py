
import csv
import tensorflow as tf
import tensorflow.contrib.data as data
import numpy as np
import random
import cv2

seed = 8            # random int, doesnt mater
parallel_calls = 10  # Number depends on hardware

# Image transformation as per: https://arxiv.org/pdf/1502.07058.pdf
# NOTE: if percent_test_imgs is 0, train_dataset is returned as type None
def get_region_dataset_training(path_label_csv, num_labels, img_width, img_height, batch_size=None, channels=3):

    def _process_imgs(path, label):

        read_type = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR
        image_decoded = cv2.imread(path.decode(), read_type)

        # Resize to 796 x 612 then remove pixels from the sides to 780 x 600: 8 px from top and bottom sides, and 6 px from left and right sides
        # Intuition: meaningful document structures are found in the center of the image due to page margins, remove 2% from side enhances features in the final image
        resized = cv2.resize(image_decoded, (600 + 12, 780 + 16), interpolation=cv2.INTER_AREA)

        # Slice regions
        holistic = resized[8:788, 6:606]
        header = resized[0:225, 0:-1]
        footer = resized[524:-1, 0:-1]
        left = resized[191:591, 0:300]
        right = resized[191:591, 300:600]

        # Resize to 224 x 224 px
        holistic = cv2.resize(holistic, (224,224), interpolation=cv2.INTER_AREA)
        header = cv2.resize(header, (224,224), interpolation=cv2.INTER_AREA)
        footer = cv2.resize(footer, (224,224), interpolation=cv2.INTER_AREA)
        left = cv2.resize(left, (224,224), interpolation=cv2.INTER_AREA)
        right = cv2.resize(right, (224,224), interpolation=cv2.INTER_AREA)

        return holistic, header, footer, left, right, label
        
        
    def _onehot(holistic, header, footer, left, right, label):
        hot = tf.one_hot(label, num_labels)
        if channels == 1:
            holistic = tf.expand_dims(holistic, 2)
            header = tf.expand_dims(header, 2)
            footer = tf.expand_dims(footer, 2)
            left = tf.expand_dims(left, 2)
            right = tf.expand_dims(right, 2)

        return holistic, header, footer, left, right, hot


    # read overview csv and get two lists of paths and labels
    img_paths = []
    img_labels= []

    img_count = 0
    with open(path_label_csv, 'r') as overview:
        reader = csv.reader(overview, delimiter=',')
        for row in reader:
            img_paths.append(row[0])
            img_labels.append(int(row[1]))
            img_count += 1
    
    if batch_size == None:
        batch_size = img_count

    # img_paths_t   = tf.convert_to_tensor(img_paths)
    # img_labels_t  = tf.convert_to_tensor(img_labels)

    # Convert lists to datatsets
    dataset = tf.data.Dataset.from_tensor_slices((img_paths,img_labels))
                             

    dataset = dataset.map(lambda path, label: tuple(tf.py_func(_process_imgs, [path, label], [tf.uint8, tf.uint8, tf.uint8, tf.uint8, tf.uint8, label.dtype])) \
                            , num_parallel_calls=parallel_calls)

    dataset = dataset.map(_onehot, num_parallel_calls=parallel_calls)

    dataset = dataset.apply(data.shuffle_and_repeat(buffer_size=batch_size*1, seed=seed)) \
                     .batch(batch_size) \
                     .prefetch(batch_size)
    return img_count, dataset


