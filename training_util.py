from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot as plt
import cv2
import numpy as np

from cvae import CVAEToolBox

def init_model_dirs(model_name):
    Path('models/{}'.format(model_name)).mkdir(parents=True, exist_ok=True)
    Path('caches/{}'.format(model_name)).mkdir(parents=True, exist_ok=True)
    Path('output/{}'.format(model_name)).mkdir(parents=True, exist_ok=True)


def run_train_loop(model, dataset, epochs, steps_pr_epoch,
                   beta_factor=None,
                   cache_every_n=10000, testset=None,
                   eval_every_epoch=True, eval_steps=None,
                   save_test_images=2, save_interpolation_image=True):
    '''
    Runs a training session for the given input
    :param model: Model to train
    :param dataset: Data to train model on
    :param epochs: Number of epochs to train
    :param steps_pr_epoch: Number of steps pr epoch
    :param beta_factor: if not None, sets beta as trained_steps / beta_factor
    :param cache_every_n: Caches training state every n training steps (within each epoch. if cache_every_n is
    smaller than steps_pr_epoch, the only states that are saved are those inbetween each epoch)
    :param testset: Images to test on
    :param eval_every_epoch: Run evaluation after each epoch. If True, test_imgs and eval_steps must be set
    :param eval_steps: Number of images to evaluate on
    :param save_test_images: Number of images to reconstruct and save during evaluation. Used to track training
    reconstruction progress
    :param save_interpolation_image: Creates a interpolation between two images each epoch. Used to track
    development of latent space during training
    '''

    if eval_every_epoch:
        assert testset is not None
        assert eval_steps is not None
        if save_interpolation_image:
            assert save_test_images >= 2

    toolbox = CVAEToolBox(model)
    train_loss = []
    val_loss = []

    if testset is not None:
        static_test_images = [next(testset) for _ in range(save_test_images)] if eval_every_epoch else []

    epoch_start = model.trained_steps // steps_pr_epoch

    for epoch in tqdm(range(epoch_start, epochs), desc='Epoch: ', leave=False):
        l = model.train_for_n_iterations(dataset, steps_pr_epoch,
                                         cache_every_n=cache_every_n,
                                         beta_factor=beta_factor)
        model.save_model()
        train_loss += l

        if eval_every_epoch:
            vl = [model.loss_func(next(testset)) for _ in range(eval_steps)]
            val_loss.append(sum(vl)/len(vl))

            for i, x in enumerate(static_test_images):
                img = x[0].numpy()
                reconstructed = toolbox.from_latent(toolbox.to_latent(x))

                demo_img = np.hstack(((img * 255).astype(np.uint8), reconstructed))
                cv2.imwrite('output/{}/img_{}_epoch_{}.jpg'.format(model.model_name, i, epoch),
                            cv2.cvtColor(demo_img, cv2.COLOR_RGB2BGR))

            if save_interpolation_image:
                a = static_test_images[0]
                b = static_test_images[1]

                images = toolbox.interpolate_between_images(a, b, steps=5)
                transition = np.hstack((images[0], images[1]))
                for img in images[2:]:
                    transition = np.hstack((transition, img))

                cv2.imwrite('output/{}/interpolation_epoch_{}.jpg'.format(model.model_name, epoch),
                            cv2.cvtColor(transition, cv2.COLOR_RGB2BGR))

        plt.plot(train_loss)
        if eval_every_epoch:
            plt.plot([(i+1)*len(l) for i in range(epoch+1)], val_loss)
            plt.legend(['training loss', 'validation loss'])
        plt.savefig('output/{}/train_metrics.png'.format(model.model_name))
        plt.clf()
