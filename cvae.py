import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Conv2D, Flatten, Dense, Reshape, Conv2DTranspose
import numpy as np
from tqdm.autonotebook import tqdm
import pickle

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
    def __init__(self, arch_def):
        super(CVAE, self).__init__()
        self.latent_dim = arch_def['latent']
        self.model_name = arch_def['name']
        self.arch_def = arch_def

        encode_layers, decode_layers = CVAE.arch_def_parser(arch_def)
        self.inference_net = tf.keras.Sequential(encode_layers)
        self.generative_net = tf.keras.Sequential(decode_layers)
        self.optimizer = tf.train.AdamOptimizer(1e-4)

    def sample(self, eps=None, n=10):
        if eps is None:
            eps = tf.random.normal(shape=(n, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
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
        encode += [Conv2D(filters=f, kernel_size=k, strides=s, activation='relu') for f, k, s in arch_def['encode']]
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
        decode += [Conv2DTranspose(filters=f, kernel_size=k, strides=s,
                                   padding='SAME', activation='relu') for f, k, s in decode_def]
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

    def compute_gradients(model, x):
        with tf.GradientTape() as tape:
            loss = model.compute_loss(x)
        return tape.gradient(loss, model.trainable_variables), loss

    def apply_gradients(self, gradients, variables):
        self.optimizer.apply_gradients(zip(gradients, variables))

    def train_for_n_iterations(model, dataset, steps, cache_every_n=10000):
        loss = []
        data_iterator = dataset.__iter__()
        pbar = tqdm(range(steps))
        for step in pbar:
            X = next(data_iterator)
            gradients, l = model.compute_gradients(X)
            model.apply_gradients(gradients, model.trainable_variables)
            loss.append(l)
            pbar.set_description("Loss {0:.2f}".format(l).ljust(15))

            if step % cache_every_n == 0:
                model.save_weights(f'caches/{model.model_name}_{step}.cache')

        return loss

    def save_model(self):
        self.save_weights(f'models/{self.model_name}.weights')
        with open(f'models/{self.model_name}.def', 'wb') as writer:
            pickle.dump(self.arch_def, writer, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_file(cls, arch_def_path, weights_path):
        with open(arch_def_path, 'rb') as handle:
            arch_def = pickle.load(handle)
        cvae = CVAE(arch_def)
        cvae.load_weights(weights_path)
        return cvae