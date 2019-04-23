import tensorflow as tf
tf.enable_eager_execution()

from cvae import CVAE
from data_prep import get_dataset

from matplotlib import pyplot as plt
from time import time

arch_def = {
    'input': (128, 128, 3),
    'latent': 128,
    'encode': [(32, 3, (2, 2)),   # out: 64, 64 32
               (64, 3, (2, 2)),   # out: 32, 32, 64
               (128, 3, (2, 2)),  # out: 16, 16, 128
               (256, 3, (2, 2)),  # out: 8, 8, 256
               (512, 3, (2, 2)),  # out: 4, 4, 512
               ],
    'decode': None,
    'name': 'face_cvae_1'
}

image_root = '/home/martin/dataset/croped_faces'
batch_size = 16
epochs = 100
ds = get_dataset(image_root, batch_size)

model = CVAE(arch_def)
model.load_weights('caches/face_cvae_1_0.cache')
loss = model.train_for_n_iterations(ds, 250000, cache_every_n=50000)
model.save_model()

plt.plot(loss)
plt.show()

