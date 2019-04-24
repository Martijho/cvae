from pathlib import Path
import cv2
import numpy as np

from tqdm import tqdm
import tensorflow as tf
from random import shuffle
'''
median h 165.0
median w 142.0
'''

tf.set_random_seed(1)   # Supposedly fixes memory leak in tf 1.13

class DataGenerator:
    def __init__(self, ds):
        self.ds = ds
        self.iterator = ds.__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.iterator)
        except tf.error.OutOfRangeError:
            self.iterator = self.ds.__iter__()
        return next(self.iterator)


def get_dataset(image_root, batch_size, shape=(128, 128, 3)):
    all_image_paths = [str(p) for p in Path(image_root).glob('*.jpg')]
    shuffle(all_image_paths)
    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    ds = path_ds.map(get_load_and_preprocess_func(shape))
    #ds = ds.shuffle(buffer_size=min(len(all_image_paths), 10000))
    ds = ds.repeat()
    ds = ds.batch(batch_size)
    return DataGenerator(ds)


def get_load_func():

    def load(path):
        image = tf.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        return image

    return load


def get_preprocess_func(shape, flip=True):

    def preprocess(image):
        if flip:
            image = tf.image.flip_left_right(image)
        image = tf.image.resize_images(image, shape[:2], preserve_aspect_ratio=True)
        image = tf.image.resize_image_with_crop_or_pad(image, shape[0], shape[1])
        image /= 255.0  # normalize to [0,1] range
        return image

    return preprocess


def get_load_and_preprocess_func(shape, flip=True):
    load_func = get_load_func()
    preprocess_func = get_preprocess_func(shape, flip=flip)

    def load_and_preprocess(path):
        image = load_func(path)
        image = preprocess_func(image)
        return image

    return load_and_preprocess


def reshape_image(image, output_shape=(128, 128, 3)):
    output = np.ones(output_shape, dtype=np.uint8)*127
    scale = max(output_shape) / max(image.shape)
    image = cv2.resize(image, None, fx=scale, fy=scale)

    output[:image.shape[0], :image.shape[1], :] = image

    return output


if __name__ == '__main__':

    for p in tqdm(list(Path('/home/martin/Desktop/data/toy').glob('*.jpg'))):
        image = cv2.imread(str(p))

        output = reshape_image(image)

        cv2.imwrite(str(p), output)

        #plt.subplot(121)
        #plt.imshow(image)
        #plt.subplot(122)
        #plt.imshow(output)
        #plt.show()

