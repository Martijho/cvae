from keras.models import Model
from keras.layers import Input, Conv2D, Dense, Flatten, UpSampling2D, Reshape, Layer
from keras import backend as K
import numpy as np


class Reparameterize(Layer):
    def __init__(self, **kwargs):
        self.mean_start = None
        self.mean_stop = None
        self.logvar_start = None
        self.logvar_stop = None
        super(Reparameterize, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        assert input_shape[1] % 2 == 0, 'Reparameterize needs a even number of inputs'

        self.mean_start = K.constant([0, 0], dtype='int32')
        self.mean_stop = K.constant([-1, (input_shape[1]//2)], dtype='int32')
        self.logvar_start = K.constant([0, input_shape[1]//2], dtype='int32')
        self.logvar_stop = K.constant([-1, -1], dtype='int32')

        super(Reparameterize, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        eps = K.random_normal(shape=self.mean_stop)
        mean = K.slice(x, self.mean_start, self.mean_stop)
        logvar = K.slice(x, self.logvar_start, self.logvar_stop)
        return eps*K.exp(logvar * .5) + mean

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1]//2)


class CVAE:
    def __init__(self, arch_def, loss='kl_mse', learning_rate=0.001, initial_beta=0.0):
        super(CVAE, self).__init__()
        self.latent_dim = arch_def['latent']
        self.model_name = arch_def['name']
        self.arch_def = arch_def

        encode_layers, decode_layers = CVAE.arch_def_parser(arch_def)
        encode_in, z = encode_layers[0], encode_layers[0]
        decode_in, x_ = decode_layers[0], decode_layers[0]
        for layer in encode_layers[1:]:
            z = layer(z)
        for layer in decode_layers[1:]:
            x_ = layer(x_)

        self.inference_net = Model(inputs=encode_in, outputs=z)
        self.generative_net = Model(inputs=decode_in, outputs=x_)

        self.cvae = Model(inputs=self.inference_net, outputs=self.generative_net(self.inference_net))
        self.cvae.compile(loss=self.kl_mse_loss, optimizer='adam')

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
        encode = [Input(input_shape=arch_def['input'])]
        encode += [Conv2D(filters=f, kernel_size=k, strides=s, activation='relu', padding='same')
                   for f, k, s in arch_def['encode']]
        encode.append(Flatten())
        encode.append(Dense(2*arch_def['latent']))
        encode.append(Reparameterize(name='reparameterize'))

        fsd_h, fsd_w = arch_def['input'][:2]
        for f, k, (s1, s2) in arch_def['encode']:
            fsd_h = fsd_h // s1
            fsd_w = fsd_w // s2
        first_spatial_dim = (fsd_h, fsd_w, arch_def['encode'][0][0])

        decode_def = arch_def['decode'] if arch_def['decode'] else list(reversed(arch_def['encode']))
        decode = [
            Input(input_shape=(arch_def['latent'],)),
            Dense(units=np.prod(first_spatial_dim), activation='relu'),
            Reshape(target_shape=first_spatial_dim),
        ]
        for f, k, s in decode_def:
            if s == (2, 2):
                decode.append(Conv2D(filters=f, kernel_size=k, padding='SAME', activation='relu'))
                decode.append(UpSampling2D(s))

        decode.append(Conv2D(filters=arch_def['input'][-1], kernel_size=3, padding='same'))


        return encode, decode

    @staticmethod
    def kl_loss(x, y):
        pass
    @staticmethod
    def reconstruction_loss(x, y):
        pass

    @classmethod
    def cvae_loss(cls, x, y):
        recon = cls.reconstruction_loss(x, y)
        kl = cls.kl_loss(x, y)


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

        loss = tf.reduce_mean(reconstruction_loss*(1-model.beta) + kl_loss*model.beta)
        # print('KL:', kl_loss)
        # print('Rec:', reconstruction_loss)
        return loss

    def compute_gradients(model, x):
        with tf.GradientTape() as tape:
            loss = model.loss_func(x)  # model.compute_loss(x)
        return tape.gradient(loss, model.trainable_variables), loss

    def apply_gradients(self, gradients, variables):
        self.optimizer.apply_gradients(zip(gradients, variables))

    def train_on_batch(model, batch, beta_factor: Union[None, int]=None):
        '''
        Runs one forwards pass of batch, computes gradients and loss and applies gradients to network
        :param batch: Model input
        :param beta_factor: either None or a int > 1. Beta is calculated as trained_steps / beta_factor.
        Recommended to use beta_factor > 0.05*Total_training_steps to limit beta at 20.
        NB: Beta is only used by kl_mse loss
        :return: calculated loss
        '''
        model.beta = 1 if not beta_factor else model.trained_steps / beta_factor
        gradients, loss = model.compute_gradients(batch)
        model.apply_gradients(gradients, model.trainable_variables)
        model.trained_steps += 1
        return loss

    def train_for_n_iterations(model, dataset, steps, cache_every_n=-1, beta_factor=None):
        loss = []
        pbar = tqdm(range(steps))
        for step in pbar:
            X = next(dataset)
            l = model.train_on_batch(X, beta_factor=beta_factor)
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
