import tensorflow as tf
tf.enable_eager_execution()

from cvae import CVAE, CVAEToolBox
from data_prep import get_dataset

from matplotlib import pyplot as plt
from tqdm import tqdm
from pathlib import Path
from time import time

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

arch_def = {
    'input': (128, 128, 3),
    'latent': 128,
    'encode': [(32, 3, (2, 2)),    # out: 64, 64 32
               (64, 3, (2, 2)),    # out: 32, 32, 64
               (128, 3, (2, 2)),   # out: 16, 16, 128
               (256, 3, (2, 2)),   # out: 8, 8, 256
               ],
    'decode': None,  # Mirror enconding for reconstruction
    'name': 'face_cvae_3'
}

model = CVAE(arch_def)
model.load_weights('models/face_cvae_3/face_cvae_3.weights')

test_root = '/home/martin/dataset/face_test'
test_imgs = [str(p) for p in Path(test_root).glob('*.jpg')]
batch_size = 8
epochs = 16
loss = []
tb = CVAEToolBox(model)

for i in range(len(test_imgs[:5])):
    inp, outp = tb.load_and_reconstruct_image(test_imgs[i])
    plt.subplot(2, 5, 1 + i)
    plt.imshow(inp)
    plt.title('Input')
    plt.axis('off')

    plt.subplot(2, 5, 6 + i)
    plt.imshow(outp)
    plt.title('Output')
    plt.axis('off')

plt.show()
out = model.decode(tb.to_latent(tb.to_tensor(test_imgs[0])))
print(out)
