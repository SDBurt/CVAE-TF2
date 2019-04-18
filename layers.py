import tensorflow as tf

# Resnet Identity Block (input size = output size)
class IdentityBlock(tf.keras.layers.Layer):
    def __init__(self, filters=64):
        super(IdentityBlock, self).__init__()
        self.identity = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters, kernel_size=(3,3), strides=(1,1), padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters, kernel_size=(3,3), strides=(1,1), padding="same"),
            tf.keras.layers.BatchNormalization()
        ])
        self.relu = tf.keras.layers.ReLU()

    def call(self, x_in):
        x_out = self.identity(x_in)
        return self.relu(x_out + x_in)


# Resnet Conv Block (input size != output size)
class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters=64):     
        super(ConvBlock, self).__init__()
        self.identity = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters, 3, 1, padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters, 3, 1, padding="same"),
            tf.keras.layers.BatchNormalization()
        ])
        self.shortcut = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters, 1, 1, padding="same"),
            tf.keras.layers.BatchNormalization(),
        ])
        self.relu = tf.keras.layers.ReLU()

    def call(self, x_in):
        x_out = self.identity(x_in)
        shortcut = self.shortcut(x_in)
        return self.relu(x_out + shortcut)
