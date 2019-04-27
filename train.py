import tensorflow as tf
tf.enable_eager_execution()

from cvae import CVAE
from data_prep import get_dataset
from training_util import init_model_dirs, run_train_loop

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

arch_def = {
    'input': (128, 128, 3),
    'latent': 512,
    'encode': [(16, 3, (2, 2)),     # out: 64, 64, 8
               (32, 3, (2, 2)),     # out: 32, 32, 16
               (64, 3, (2, 2)),     # out: 16, 16, 32
               (128, 3, (2, 2)),    # out: 8, 8, 64
               (256, 3, (2, 2)),    # out: 4, 4, 128
               (512, 3, (2, 2)),    # out: 2, 2, 256
               (1024, 3, (2, 2))],  # out: 1, 1, 512

    'decode': None,  # Mirror enconding for reconstruction
    'name': 'face_cvae_first_stage'
}

model = CVAE(arch_def, loss='kl_mse', learning_rate=0.0001)
init_model_dirs(model.arch_def['name'])

image_root = '/home/martin/dataset/cropped_faces'
test_root = '/home/martin/dataset/face_test'
batch_size = 4
epochs = 1500
steps_pr_epoch = 100

train_data = get_dataset(image_root, batch_size, model.arch_def['input'])
eval_data = get_dataset(test_root, 1, model.arch_def['input'])
run_train_loop(model,
               train_data,
               epochs,
               steps_pr_epoch,
               increase_beta_at=[500, 800, 1000, 1200, 1400],
               testset=eval_data,
               eval_every_epoch=True,
               eval_steps=100,
               save_test_images=2,
               save_interpolation_image=True)


arch_def['name'] = 'face_cvae_second_stage'
init_model_dirs(model.arch_def['name'])
model = CVAE(arch_def, loss='kl_mse', learning_rate=0.00001)
model.encoding_trainable(False)
run_train_loop(model,
               train_data,
               epochs,
               steps_pr_epoch,
               increase_beta_at=None,
               testset=eval_data,
               eval_every_epoch=True,
               eval_steps=100,
               save_test_images=2,
               save_interpolation_image=True)
