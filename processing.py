import tensorflow as tf

# Get the dataset for this project
def get_dataset():

    (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
    
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')

    train_images = preproccess(train_images)
    test_images = preproccess(test_images)
    return train_images, test_images

# Normalizing the images to the range of [0., 1.]
def preproccess(images):

    # Normalize
    images = images/ 255.

    # Binarization
    images[images >= .5] = 1.
    images[images < .5] = 0.

    return images