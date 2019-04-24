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
