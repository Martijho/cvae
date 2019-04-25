from cvae import CVAE, CVAEToolBox
from data_prep import get_dataset

from pathlib import Path
from sklearn.clusters import MiniBatchKMeans
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt


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


image_root = '/home/martin/Desktop/data/darknet_data/openimgs_extra_v2'
model = CVAE.from_file('models/oi_cvae_6/oi_cvae_6.state', 'models/oi_cvae_6/oi_cvae_6.weights')
tb = CVAEToolBox(model)
images = ImageFeed(image_root, tb)

batch_size = 200
n_imgs = 10000

kmeans = MiniBatchKMeans(16, batch_size=batch_size)

for i in tqdm(range(n_imgs // batch_size)):
    batch = []
    for _ in range(batch_size):
        tensor, _ = next(images)
        latent = tb.to_latent(tensor).numpy()[0]

        print(latent.shape)

        batch.append(latent)

    kmeans.partial_fit(np.array(batch))

center_images = centroids_to_images(kmeans, tb)


for i, image in enumerate(center_images):
    plt.subplot(4, 4, 1 + i)
    plt.imshow(image)
    plt.axis('off')

plt.show()


