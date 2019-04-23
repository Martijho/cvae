import tensorflow as tf
tf.enable_eager_execution()

from cvae import CVAE
from data_prep import get_dataset

from matplotlib import pyplot as plt
from time import time

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

arch_def = {
    'input': (512, 512, 3),
    'latent': 512,
    'encode': [(32, 3, (2, 2)),    # out: 256, 256 32
               (64, 3, (2, 2)),    # out: 128, 128, 64
               (128, 3, (2, 2)),   # out: 64, 64, 128
               (256, 3, (2, 2)),   # out: 32, 32, 256
               (512, 3, (2, 2)),   # out: 16, 3162, 512
               (1024, 3, (2, 2)),  # out: 8, 8, 1024
               (1024, 3, (2, 2)),  # out: 4, 4, 1024
               ],
    'decode': None,  # Mirror enconding for reconstruction
    'name': 'face_cvae_1'
}

image_root = '/home/martin/Desktop/data/darknet_data/openimgs_extra_v2'
batch_size = 8
epochs = 2
steps = epochs * 800000 // batch_size
ds = get_dataset(image_root, batch_size, arch_def['input'])

model = CVAE(arch_def)
#model.load_weights('caches/face_cvae_1_0.cache')
loss = model.train_for_n_iterations(ds, steps, cache_every_n=10000)
model.save_model()

plt.plot(loss)
plt.show()

