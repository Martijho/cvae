import tensorflow as tf
tf.enable_eager_execution()

from cvae import CVAE, CVAEToolBox

from pathlib import Path
from sklearn.cluster import MiniBatchKMeans, Kmeans
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import cv2

from collections import defaultdict


class ImageFeed:
    def __init__(self, root, tool_box):
        self.root = root
        self.tool_box = tool_box
        self.image_iterator = Path(root).glob('*.jpg')

    def __iter__(self):
        return self

    def __next__(self):
        img_path = str(next(self.image_iterator))
        tensor = self.tool_box.to_tensor(img_path)
        return tensor, img_path


def centroids_to_images(kmeans, toolbox):
    centroids = kmeans.cluster_centers_
    images = []
    for i in range(centroids.shape[0]):
        center = centroids[i][:]
        tensor = toolbox.to_tensor(center, with_batch_dim=True, preprocess=False)
        img = toolbox.from_latent(tensor)
        images.append(img)
    return images


def show_some_labels(kmeans, toolbox, images, n=160):
    label2image = defaultdict(list)

    for i in range(n):
        tensor, path = next(images)
        latent = toolbox.to_latent(tensor).numpy()
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_RGB2BGR)

        label = kmeans.predict(latent)[0]
        print(label)
        if len(label2image[label]) >= 4:
            continue
        label2image[label].append(image)

    for label, images in label2image.items():
        for i, image in enumerate(images[:4]):
            plt.subplot(2, 2, 1 + i)
            plt.imshow(image)
            plt.title(f'label {label}')
            plt.axis('off')
        plt.show()


image_root = '/home/martin/Desktop/data/darknet_data/openimgs_extra_v2'
model = CVAE.from_file('models/oi_cvae_5/oi_cvae_5.state', 'models/oi_cvae_5/oi_cvae_5.weights')
tb = CVAEToolBox(model)
images = ImageFeed(image_root, tb)

set_size = 100000

kmeans = Kmeans(49)
load = False
save = True

if load:
    kmeans.cluster_centers_ = np.load('centroids.npy')
else:
    batch = []
    for _ in tqdm(range(set_size), leave=False, desc='loading  '):
        tensor, _ = next(images)
        latent = tb.to_latent(tensor).numpy()[0]

        batch.append(latent)

    kmeans.fit(np.array(batch))
    if save:
        np.save('centroids', kmeans.cluster_centers_)

center_images = centroids_to_images(kmeans, tb)


for i, image in enumerate(center_images):
    plt.subplot(7, 7, 1 + i)
    plt.imshow(image)
    plt.axis('off')

plt.show()

show_some_labels(kmeans, tb, images)
