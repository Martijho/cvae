import tensorflow as tf
tf.enable_eager_execution()

from cvae import CVAE
from data_prep import get_dataset
from training_util import init_model_dirs, run_train_loop

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

arch_def = {
    'input': (512, 512, 3),
    'latent': 2048,
    'encode': [(32, 3, (2, 2)),     # out: 256, 256, 32
               (64, 3, (2, 2)),     # out: 128, 128, 64
               (128, 3, (2, 2)),    # out: 64, 64, 128
               (256, 3, (2, 2)),    # out: 32, 32, 256
               (512, 3, (2, 2)),    # out: 16, 16, 512
               (1024, 3, (2, 2)),   # out: 8, 8, 1024
               (2048, 3, (2, 2))],  # out: 4, 4, 2048
    'decode': None,  # Mirror enconding for reconstruction
    'name': 'oi_cvae_7'
}

model = CVAE(arch_def, loss='kl_mse', learning_rate=0.0001)
#model.load_weights('models/oi_cvae_4/oi_cvae_4.weights')

init_model_dirs(model.arch_def['name'])

image_root = '/home/martin/Desktop/data/darknet_data/openimgs_extra_v2'
test_root = '/home/martin/Desktop/data/validation_set'
batch_size = 8
total_steps = 15000
epochs = 2000
steps_pr_epoch = 100

train_data = get_dataset(image_root, batch_size, model.arch_def['input'])
eval_data = get_dataset(test_root, 1, model.arch_def['input'])
run_train_loop(model,
               train_data,
               epochs,
               steps_pr_epoch,
               testset=eval_data,
               eval_every_epoch=True,
               eval_steps=100,
               save_test_images=2,
               save_interpolation_image=True)


