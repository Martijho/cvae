import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Conv2D, Flatten, Dense, Reshape, Conv2DTranspose
import numpy as np
from tqdm.autonotebook import tqdm
import pickle

from typing import Union

from data_prep import get_load_and_preprocess_func, get_preprocess_func, get_load_func

'''
tf implementation of a Convolutional Variational Autoencoder. 
Source: https://www.tensorflow.org/alpha/tutorials/generative/cvae
'''

EXAMPLE_ARCH_DEF = {
    'input': (28, 28, 1),
    'latent': 12,
    'encode': [(32, 3, (2, 2)),
               (64, 3, (2, 2))],
    'decode': None,
    'name': 'MODEL_NAME'
}


class CVAE(tf.keras.Model):
    def __init__(self, arch_def, loss='kl_mse', learning_rate=0.001, initial_beta=0.0):
        super(CVAE, self).__init__()
        self.latent_dim = arch_def['latent']
        self.model_name = arch_def['name']
        self.arch_def = arch_def

        encode_layers, decode_layers = CVAE.arch_def_parser(arch_def)
        self.inference_net = tf.keras.Sequential(encode_layers)
        self.generative_net = tf.keras.Sequential(decode_layers)
        self.optimizer = tf.train.AdamOptimizer(learning_rate)

        if loss == 'cross_entropy':
            self.loss_func = self.compute_loss
        elif loss == 'kl_mse':
            self.loss_func = self.compute_KL_MSE_loss
        else:
            raise NotImplementedError('loss function not recognized')
        self._loss_arg = loss
        self.trained_steps = 0
        self.beta = initial_beta

    def increase_beta(self, increment=0.25):
        self.beta += increment

    def sample(self, eps=None, n=10):
        if eps is None:
            #eps = tf.random.normal(shape=(n, self.latent_dim))  # tf 1.13
            eps = tf.random_normal(shape=(n, self.latent_dim))  # tf 1.10
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        #eps = tf.random.normal(shape=mean.shape)  # tf 1.13
        eps = tf.random_normal(shape=mean.shape)  # tf 1.10
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=True):
        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs

        return logits

    @staticmethod
    def arch_def_parser(arch_def):
        '''
        Parsing the arch_def dictionary and generates layers in encoder and decoder
        :param arch_def: Dict of architecture definitions
        :return: (encoding layers, decoding layers)
        '''
        # TODO: reshape input to arch_def['input']
        encode = [InputLayer(input_shape=arch_def['input'])]
        encode += [Conv2D(filters=f, kernel_size=k, strides=s, activation='relu', padding='same',)
                   for f, k, s in arch_def['encode']]
        encode.append(Flatten())
        encode.append(Dense(2*arch_def['latent']))

        fsd_h, fsd_w = arch_def['input'][:2]
        for f, k, (s1, s2) in arch_def['encode']:
            fsd_h = fsd_h // s1
            fsd_w = fsd_w // s2
        first_spatial_dim = (fsd_h, fsd_w, arch_def['encode'][0][0])

        decode_def = arch_def['decode'] if arch_def['decode'] else list(reversed(arch_def['encode']))
        decode = [
            InputLayer(input_shape=(arch_def['latent'],)),
            Dense(units=np.prod(first_spatial_dim), activation=tf.nn.relu),
            Reshape(target_shape=first_spatial_dim),
        ]
        decode += [Conv2DTranspose(filters=f, kernel_size=k, strides=s, padding='SAME', activation='relu')
                   for f, k, s in decode_def]
        decode.append(Conv2DTranspose(filters=arch_def['input'][-1], kernel_size=3,
                                      strides=(1, 1), padding='SAME'))

        return encode, decode

    @staticmethod
    def log_normal_pdf(sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)

    def compute_loss(model, x):
        mean, logvar = model.encode(x)
        z = model.reparameterize(mean, logvar)
        x_logit = model.decode(z)

        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = CVAE.log_normal_pdf(z, 0., 0.)
        logqz_x = CVAE.log_normal_pdf(z, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)

    def compute_KL_MSE_loss(model, x):
        ''' From hackathon repo
        reconstruction_loss = mse(K.flatten(inputs), K.flatten(outputs))
        reconstruction_loss *= input_shape[0] * input_shape[0]
        kl_loss = 1. + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        vae.add_loss(vae_loss)
        '''

        mean, logvar = model.encode(x)
        z = model.reparameterize(mean, logvar)
        x_logit = model.decode(z)

        reconstruction_loss = tf.reshape(x, [-1])-tf.reshape(x_logit, [-1])
        reconstruction_loss = reconstruction_loss * reconstruction_loss
        reconstruction_loss = tf.reduce_sum(reconstruction_loss)/tf.cast(tf.reduce_prod(x.shape), tf.float32)
        reconstruction_loss = reconstruction_loss * model.arch_def['input'][0] * model.arch_def['input'][1]

        kl_loss = 1. + logvar - mean*mean - tf.exp(logvar)
        kl_loss = tf.reduce_sum(kl_loss, axis=-1)
        kl_loss = -0.5*kl_loss

        loss = tf.reduce_mean(reconstruction_loss + kl_loss*model.beta)
        # print('KL:', kl_loss)
        # print('Rec:', reconstruction_loss)
        return loss

    def compute_gradients(model, x):
        with tf.GradientTape() as tape:
            loss = model.loss_func(x)  # model.compute_loss(x)
        return tape.gradient(loss, model.trainable_variables), loss

    def apply_gradients(self, gradients, variables):
        self.optimizer.apply_gradients(zip(gradients, variables))

    def train_on_batch(model, batch):
        '''
        Runs one forwards pass of batch, computes gradients and loss and applies gradients to network
        :param batch: Model input
        :param beta_factor: either None or a int > 1. Beta is calculated as trained_steps / beta_factor.
        Recommended to use beta_factor > 0.05*Total_training_steps to limit beta at 20.
        NB: Beta is only used by kl_mse loss
        :return: calculated loss
        '''
        #model.beta = 1 if not beta_factor else model.trained_steps / beta_factor
        gradients, loss = model.compute_gradients(batch)
        model.apply_gradients(gradients, model.trainable_variables)
        model.trained_steps += 1
        return loss

    def train_for_n_iterations(model, dataset, steps, cache_every_n=-1):
        loss = []
        pbar = tqdm(range(steps))
        for step in pbar:
            X = next(dataset)
            l = model.train_on_batch(X)
            loss.append(l)
            pbar.set_description("Loss {0:.2f}".format(l).ljust(15))

            if cache_every_n > 0 and step % cache_every_n == 0:
                model.save_weights(f'caches/{model.model_name}/{model.model_name}_{step}.cache')

        return loss

    def save_model(self):
        self.save_weights(f'models/{self.model_name}/{self.model_name}.weights')
        state = {'arch_def': self.arch_def,
                 'beta': self.beta,
                 'trained_steps': self.trained_steps,
                 'loss': self._loss_arg}
        with open(f'models/{self.model_name}/{self.model_name}.state', 'wb') as writer:
            pickle.dump(state, writer, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_file(cls, state_path, weights_path):
        with open(state_path, 'rb') as handle:
            state = pickle.load(handle)
        cvae = CVAE(state['arch_def'], loss=state['loss'], initial_beta=state['beta'])
        cvae.trained_steps = state['trained_steps']
        cvae.load_weights(weights_path)
        return cvae

    def encoding_trainable(self, trainable):
        self.inference_net.trainable = trainable

    def decoding_trainable(self, trainable):
        self.generative_net.trainable = trainable

    def recreate_optimizer(self, loss):
        self.optimizer = tf.train.AdamOptimizer(loss)

class CVAEToolBox:
    def __init__(self, model: CVAE):
        self.model = model
        self.load_and_preprocess = get_load_and_preprocess_func(model.arch_def['input'], flip=False)
        self.load_from_file = get_load_func()
        self.preprocess = get_preprocess_func(model.arch_def['input'], flip=False)

    def to_tensor(self, input_: Union[np.ndarray, str], with_batch_dim=True, preprocess=True):
        if type(input_) == str:
            tensor = self.load_from_file(input_)
        else:
            tensor = tf.convert_to_tensor(input_, dtype=tf.float32)

        if preprocess:
            tensor = self.preprocess(tensor)

        if with_batch_dim:
            tensor = tf.expand_dims(tensor, 0)
        return tensor

    def load_and_reconstruct_image(self, path):
        image = self.load_and_preprocess(path)
        x = tf.expand_dims(image, 0)
        output = self.from_latent(self.to_latent(x))
        return image, output

    def to_latent(self, x):
        mean, logvar = self.model.encode(x)
        z = self.model.reparameterize(mean, logvar)
        return z

    def from_latent(self, z):
        output = self.model.decode(z)
        output = output[0].numpy()
        #output = output / output.max()
        output = (output * 255).astype(np.uint8)
        return output

    def interpolate_between_images(self, a, b, steps=10):
        assert steps > 1
        latent_a = self.to_latent(a)
        latent_b = self.to_latent(b)

        difference = latent_b - latent_a #- latent_b
        delta = difference / steps

        latent_steps = [latent_a] + [latent_a + i*delta for i in range(1, steps-1)] + [latent_b]
        images = [self.from_latent(latent) for latent in latent_steps]
        return images

    def get_gen_with_diff_function(self, original, translation):
        diff = self.to_latent(translation) - self.to_latent(original)

        def add_diff_to_image(image):
            latent = self.to_latent(image) + diff
            return self.from_latent(latent)

        return add_diff_to_image
