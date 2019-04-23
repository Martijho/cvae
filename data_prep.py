from pathlib import Path
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import tensorflow as tf
'''
median h 165.0
median w 142.0
'''


def get_dataset(image_root, batch_size):
    all_image_paths = [str(p) for p in Path(image_root).glob('*.jpg')]
    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    ds = path_ds.map(load_and_preprocess_image)
    ds = ds.shuffle(buffer_size=len(all_image_paths))
    ds = ds.repeat()
    ds = ds.batch(batch_size)
    return ds


def load_and_preprocess_image(path):
    image = tf.read_file(path)
    return preprocess_image(image)


def preprocess_image(image_path):
    image = tf.image.decode_jpeg(image_path, channels=3)
    image = tf.image.resize_images(image, [128, 128])
    image /= 255.0  # normalize to [0,1] range

    return image


def reshape_image(image, output_shape=(128, 128, 3)):
    output = np.ones(output_shape, dtype=np.uint8)*127
    scale = max(output_shape) / max(image.shape)
    image = cv2.resize(image, None, fx=scale, fy=scale)

    output[:image.shape[0], :image.shape[1], :] = image

    return output


if __name__ == '__main__':

    for p in tqdm(list(Path('/home/martin/dataset/croped_faces').glob('*.jpg'))):
        image = cv2.imread(str(p))

        output = reshape_image(image)

        cv2.imwrite(str(p), output)

        #plt.subplot(121)
        #plt.imshow(image)
        #plt.subplot(122)
        #plt.imshow(output)
        #plt.show()

