#chupinhei de: https://github.com/roatienza/Deep-Learning-Experiments/blob/master/Experiments/Tensorflow/GAN/dcgan_mnist.py
# e de https://stackoverflow.com/questions/30230592/loading-all-images-using-imread-from-a-given-folder/30230738
# e de https://stackoverflow.com/questions/43391205/add-padding-to-images-to-get-them-into-the-same-shape
# fonte das imagens: https://veekun.com/dex/downloads
# com dicas de https://github.com/eriklindernoren/Keras-GAN/blob/master/wgan/wgan.py
import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
import cv2
from IPython import display
from keras import backend as K



def pad_images_to_same_size(images):
    """
    :param images: sequence of images
    :return: list of images padded so that all images have same width and height (max width and height are used)
    """
    width_max = 0
    height_max = 0
    for img in images:
        h, w = img.shape[:2]
        width_max = max(width_max, w)
        height_max = max(height_max, h)

    images_padded = []
    for img in images:
        h, w = img.shape[:2]
        diff_vert = height_max - h
        pad_top = diff_vert//2
        pad_bottom = diff_vert - pad_top
        diff_hori = width_max - w
        pad_left = diff_hori//2
        pad_right = diff_hori - pad_left
        #img_padded = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[255,255,255])
        img_padded = cv2.resize(img,(64, 64))
        # assert img_padded.shape[:2] == (height_max, width_max)
        img_padded = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
        images_padded.append(img_padded)
        images_padded.append(cv2.flip(img_padded,1))
    plt.imshow(images_padded[10])
    plt.show()
    return images_padded


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

pokemao = load_images_from_folder("pikaplain")


pokemao = pad_images_to_same_size(pokemao)

train_images = np.asarray(pokemao)
train_images = train_images.reshape(train_images.shape[0], 64, 64, 3).astype('float16')
train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]

BUFFER_SIZE = 32
BATCH_SIZE = 16

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(4*4*256*3, use_bias=False, input_shape=(256,)))
    #model.add(layers.BatchNormalization())
    #model.add(layers.LeakyReLU())
    #model.add(layers.Dropout(0.3))

    model.add(layers.Reshape((4, 4, 256*3)))
    assert model.output_shape == (None, 4, 4, 256*3) # Note: None is the batch size

    #model.add(layers.UpSampling2D(interpolation='bilinear'))
    model.add(layers.Conv2D(128*3,(2,2), strides = (1,1), padding = "same"))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.UpSampling2D(interpolation='nearest'))
    model.add(layers.Conv2D(64*3, (2, 2), strides=(1, 1), padding = "same"))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.UpSampling2D(interpolation='nearest'))
    model.add(layers.Conv2D(32*3, (2, 2), strides=(1, 1), padding = "same"))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.UpSampling2D(interpolation='nearest'))
    model.add(layers.Conv2D(16*3, (2, 2), strides=(1, 1), padding = "same"))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU())
    model.add(layers.UpSampling2D(interpolation='nearest'))
    model.add(layers.Conv2D(8 * 3, (2, 2), strides=(1, 1), padding="same"))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2D(3, (2, 2), strides=(1, 1), padding = "same", activation='tanh'))
    #model.add(layers.Cropping2D(((8,7),(8,7))))
    print(model.output_shape)
    assert model.output_shape == (None, 64, 64, 3)

    return model




def make_generator_modelS():
    model = tf.keras.Sequential()
    model.add(layers.Dense(4*4*256*3, use_bias=False, input_shape=(256,)))
    #model.add(layers.BatchNormalization())
    #model.add(layers.LeakyReLU())
    #model.add(layers.Dropout(0.3))

    model.add(layers.Reshape((4, 4, 256*3)))
    #assert model.output_shape == (None, 4, 4, 256) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128*3, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    print(model.output_shape)

    #assert model.output_shape == (None, 8, 8, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))


    #model.add(layers.UpSampling2D(size = (3,3)))
    #model.add(layers.Conv2D(64, (5, 5), strides=(1, 1)))
    model.add(layers.Conv2DTranspose(64*3, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    print(model.output_shape)

    #assert model.output_shape == (None, 16, 16, 64)
    #model.add(layers.BatchNormalization())
    #model.add(layers.LeakyReLU())
    #model.add(layers.Dropout(0.3))
    model.add(layers.Conv2DTranspose(32*3, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    #model.add(layers.Cropping2D(((6,5),(6,5))))
    print(model.output_shape)

    assert model.output_shape == (None, 64, 64, 3)

    return model


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[64, 64, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1024))
    model.add(layers.LeakyReLU())
    model.add(layers.Dense(1))



    return model


generator = make_generator_model()
discriminator = make_discriminator_model()

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss


# Define the loss functions for the generator.
def generator_loss(fake_img):
    return -tf.reduce_mean(fake_img)


def odiscriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output)*0.9, real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def ogenerator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


generator_optimizer = tf.keras.optimizers.RMSprop(lr=0.00005)
discriminator_optimizer = tf.keras.optimizers.RMSprop(lr=0.00005)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

EPOCHS = 100000
noise_dim = 256
num_examples_to_generate = 16

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

#clip_value = 0.01

def train(dataset, epochs):
  clip_value = 0.05
  for epoch in range(epochs):
    start = time.time()
    for n in range(5):
        for image_batch in dataset:
          train_step(image_batch)

        #clip weights
        for l in discriminator.layers:
            weights = l.get_weights()
            weights = [np.clip(w, -clip_value, clip_value) for w in weights]
            l.set_weights(weights)

    # Produce images for the GIF as we go
    if epoch % 50 == 0:
        display.clear_output(wait=True)
        generate_and_save_images(generator,
                             epoch + 1,
                             seed)

    # Save the model every 15 epochs
    #if (epoch + 1) % 15 == 0:
    #  checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  display.clear_output(wait=True)
  generate_and_save_images(generator,
                           epochs,
                           seed)

def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(16,16))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)

      plott = predictions[i, :, :, :] * 127.5 + 127.5
      #plott = cv2.cvtColor(np.uint8(plott), cv2.COLOR_BGR2RGB)

      #print(str(np.shape(plott)))
      plt.imshow(np.uint8(plott))
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  #plt.show()

train(train_dataset, EPOCHS)







