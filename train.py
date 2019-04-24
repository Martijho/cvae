import tensorflow as tf
tf.enable_eager_execution()

from cvae import CVAE, CVAEToolBox
from data_prep import get_dataset

from matplotlib import pyplot as plt
from tqdm import tqdm
from pathlib import Path
import cv2
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
Path('models/{}'.format(arch_def['name'])).mkdir(parents=True, exist_ok=True)
Path('caches/{}'.format(arch_def['name'])).mkdir(parents=True, exist_ok=True)
Path('output/{}'.format(arch_def['name'])).mkdir(parents=True, exist_ok=True)

model = CVAE(arch_def)
image_root = '/home/martin/dataset/cropped_faces'
test_root = '/home/martin/dataset/face_test'
batch_size = 16
epochs = 180
loss = []
tb = CVAEToolBox(model)

ds = get_dataset(image_root, batch_size, arch_def['input'])
test_imgs = [str(p) for p in Path(test_root).glob('*.jpg')]

for epoch in tqdm(range(epochs), desc='Epoch: ', leave=False):
    l = model.train_for_n_iterations(ds, 16000 // batch_size, cache_every_n=5000)
    model.save_model()
    loss += l

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
    plt.savefig('output/{}/plot_{}.png'.format(arch_def['name'], epoch))
    plt.clf()
    cv2.imwrite('output/{}/{}_{}.jpg'.format(arch_def['name'], arch_def['name'], epoch),
                cv2.cvtColor(outp, cv2.COLOR_RGB2BGR))

plt.plot(loss)
plt.show()

