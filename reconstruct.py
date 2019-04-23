import tensorflow as tf
tf.enable_eager_execution()

from cvae import CVAE, CVAEToolBox
from data_prep import get_load_and_preprocess_func

from matplotlib import pyplot as plt
import numpy as np
from time import time

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

weights_path = 'caches/face_cvae_1_30000.cache'
arch_def_path = None

if arch_def_path is None:
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
    model = CVAE(arch_def)
    model.load_weights(weights_path)
else:
    model = CVAE.from_file(arch_def_path, weights_path)

load_and_preprocess = get_load_and_preprocess_func(model.arch_def['input'])

image_paths = [
    '/home/martin/Desktop/data/validation_set/Bakeriet_4-Action_sequence_00336.jpg',
    '/home/martin/Desktop/data/validation_set/Stand_up_small_group_2.jpg'
]

toolbox = CVAEToolBox(model)

a = toolbox.to_tensor(image_paths[0], with_batch_dim=True, preprocess=True)
b = toolbox.to_tensor(image_paths[1], with_batch_dim=True, preprocess=True)

images = toolbox.interpolate_between_images(a, b, steps=12)
a_img, a_latent = toolbox.load_and_reconstruct_image(image_paths[0])
b_img, b_latent = toolbox.load_and_reconstruct_image(image_paths[1])

print(len(images))
images = [a_img, a_latent] + images + [b_latent, b_img]

for i, image in enumerate(images):
    plt.subplot(4, 4, 1 + i)
    plt.imshow(image)
    plt.axis('off')
plt.show()

'''
for image_path in image_paths:
        
    img = toolbox.load_and_preprocess(image_path)
    print(type(img))

    image, output = toolbox.load_and_reconstruct_image(image_path)

    plt.subplot(121)
    plt.title('Input')
    plt.imshow(image)
    plt.subplot(122)
    plt.title('Output')
    plt.imshow(output)
    plt.show()
'''