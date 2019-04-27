# Convolutional Variational Autoencoder
This repo contains a CVAE object which simplify training and using a convolutional variational autoencoder. 

The cvae structure is defined through a architecture-definition dictionary (arch_def). 
```python
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
```
* ```'input'```: input image size
* ```'latent'```: size of latent space
* ```'encode'```: list of tuples of conv arguments for the encoding module. (channels, kernel-size, strides)
* ```'decode'```: same as encode, but for decoding module. if None, the encoding-layer arguments is mirrored for decing
* ```'name'```: model name. used when storing weights and arch_def

#### Provided functionality: 
* CVAE (cvae.py): autoencoder-object with methods for 
    * sampling latent space
    * encoding image as latent vector
    * decodes latent vector to image
    * training 
    * save/load functions.
    * reinitializing optimizer with new learning rate
    * Locking weights in encode or decode module
* CVAEToolBox (cvae.py): util-object with some usefull functionality like 
    * to/from_latent space
    * to-tensor(path/np.ndarray)
    * interpolating between images
    * function that encodes image, adds some latent translation and decodes result
* Data preperation (data_prep.py):
    * get_dataset: takes list of image paths, batch-size and model-input-size and returns tf.data.Dataset for training
    * get_load_func: returns a function that takes image path and returns image-tensor
    * get_preprocess_func: returns a function that preprocesses a image-tensor
    * get_load_and_preprocess_func: returns a function that takes image path and returns a preprocessed image tensor

#### Training structure
The problem with training a cvae on a combination of reconstruction and KL-loss is that a continous and disentangled latent
space is not a task that is well alligned with reconstruction. The training is therefore split in two stages: 
###### Stage One: 
The cvae is first trained on only reconstruction loss for a defined number of epochs before KL loss is gradually introduced. 
Over 100 epochs, the KL loss is scaled linearly from 0 to 1, and then incremented by 1 at set epoch-numbers. This is done 
to not massivly change gradients from one epoch to another, but gradually shift the networks attention from reconstruction to 
good feature representation in the cvae bottle neck. 
###### Stage Two: 
When the joined KL-reconstruction loss has stabilized, stage two is started by locking the weights in the encoding module. 
This stage is all about training the decoding module on reconstruction loss only. This way, we hopefully achieve the 
best of both worlds with an encoding module, specialized on creating a good latent representation, and a decoding module 
that only cares about reconstruction

#### Work in progress
* Pure keras implementation for more effective training 
* Clustering scripts 