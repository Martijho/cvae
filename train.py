from cvae import CVAE
from data_prep import get_dataset
from training_util import init_model_dirs, run_train_loop

import tensorflow as tf
tf.enable_eager_execution()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

arch_def = {
    'input': (512, 512, 3),
    'latent': 512,
    'encode': [(8, 3, (2, 2)),      # out: 256, 256, 8
               (16, 3, (2, 2)),     # out: 128, 128, 16
               (32, 3, (2, 2)),     # out: 64, 64, 32
               (64, 3, (2, 2)),     # out: 32, 32, 64
               (128, 3, (2, 2)),    # out: 16, 16, 128
               (256, 3, (2, 2)),    # out: 8, 8, 256
               (512, 3, (2, 2)),    # out: 4, 4, 512
               (1024, 3, (2, 2))],  # out: 2, 2, 1024
    'decode': None,  # Mirror enconding for reconstruction
    'name': 'oi_cvae_1'
}
init_model_dirs(arch_def['name'])

model = CVAE(arch_def)
image_root = '/home/martin/Desktop/data/darknet_data/openimgs_extra_v2'
test_root = '/home/martin/Desktop/data/validation_set'
batch_size = 16
epochs = 180
steps_pr_epoch = 16000 // batch_size

train_data = get_dataset(image_root, batch_size, arch_def['input'])
eval_data = get_dataset(test_root, 1, arch_def['input'])
run_train_loop(model,
               train_data,
               epochs,
               steps_pr_epoch,
               cache_every_n=steps_pr_epoch+5,
               test_images=eval_data,
               eval_every_epoch=True,
               eval_steps=100,
               save_test_images=2,
               save_interpolation_image=True)

'''
for epoch in tqdm(range(epochs), desc='Epoch: ', leave=False):
    l = model.train_for_n_iterations(ds, 16000 // batch_size, cache_every_n=10000)
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
'''

