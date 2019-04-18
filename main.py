import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, time, glob, PIL, imageio, argparse
from pathlib import Path
from datetime import datetime
from processing import get_dataset
from models import CVAE
from tqdm import trange

parser = argparse.ArgumentParser()

# Model
parser.add_argument('--latent_dim', type=int, default='50')
parser.add_argument('--learning_rate', type=float, default='1e-4')
parser.add_argument('--num_examples_to_generate', type=int, default='16')

# Training
parser.add_argument('--train_buf', type=int, default='60000')
parser.add_argument('--batch_size', type=int, default='100')
parser.add_argument('--epochs', type=int, default='50')

# Data
parser.add_argument('--save_dir', type=str, default='./saves/')
parser.add_argument('--log_dir', type=str, default='./logs/')
parser.add_argument("--log_freq", type=int, default=10)
parser.add_argument('--extension', type=str, default=None)

# Testing / Evaluation
parser.add_argument('--test_buf', type=int, default='10000')
parser.add_argument('--image_dir', type=str, default='./images/')
parser.add_argument('--make_gif', type=bool, default=False)

cfg = parser.parse_args()

# Convolutional Variable Auto Encode
class ConvolutionalVariationalAutoencoder(object):

    def __init__(self):
        super(ConvolutionalVariationalAutoencoder, self).__init__()

        train_images, test_images = get_dataset()

        self.train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(cfg.train_buf).batch(cfg.batch_size)
        self.test_dataset = tf.data.Dataset.from_tensor_slices(test_images).shuffle(cfg.test_buf).batch(cfg.batch_size)
        self.optimizer = tf.keras.optimizers.Adam(lr=cfg.learning_rate)

        self.global_step = 0

        self.model = CVAE(cfg.latent_dim)
        self.build_writers()

    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)

    def build_writers(self):

        if not Path(cfg.save_dir).is_dir():
            os.mkdir(cfg.save_dir)

        if not Path(cfg.image_dir).is_dir():
            os.mkdir(cfg.image_dir)

        if cfg.extension is None:
            cfg.extension = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

        self.log_path = cfg.log_dir + cfg.extension
        self.writer = tf.summary.create_file_writer(self.log_path)
        self.writer.set_as_default()

        self.save_path = cfg.save_dir + cfg.extension
        self.ckpt_prefix = self.save_path + '/ckpt'
        # self.saver = tf.train.Checkpoint(generator=self.generator, gen_optim=self.gen_optim, discriminator=self.discriminator, disc_optim=self.disc_optim, global_step=self.global_step, epoch=self.epoch)
        
        tf.summary.experimental.set_step(0)

    def logger(self, loss):
        if self.global_step % cfg.log_freq == 0:
            tf.summary.scalar('loss', loss, step=self.global_step)


    def log_img(self, name, img):
        if self.global_step % (cfg.log_freq*4) == 0:
            tf.summary.image(name, img, step=self.global_step, max_outputs=6)


    def generate_and_save_images(self, model, epoch, test_input):
        predictions = model.sample(test_input)

        fig = plt.figure(figsize=(4,4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow(predictions[i, :, :, 0], cmap='gray')
            plt.axis('off')

        # tight_layout minimizes the overlap between 2 sub-plots
        # holy hack is bad
        plt.savefig('{}image_at_epoch_{:04d}.png'.format(cfg.image_dir,epoch))
        
        img = np.asarray(PIL.Image.open('{}image_at_epoch_{:04d}.png'.format(cfg.image_dir,epoch)).convert('RGB'))
        
        self.log_img("predictions", np.array([img]))

    def make_gif(self):
        anim_file = 'evaluation/cvae.gif'

        with imageio.get_writer(anim_file, mode='I') as writer:
            filenames = glob.glob(cfg.image_dir+'image*.png')
            filenames = sorted(filenames)
            last = -1
            for i,filename in enumerate(filenames):
                frame = 2*(i**0.5)
                if round(frame) > round(last):
                    last = frame
                else:
                    continue
                image = imageio.imread(filename)
                writer.append_data(image)
            image = imageio.imread(filename)
            writer.append_data(image)

    def train(self):

        # keeping the random vector constant for generation (prediction) so
        # it will be easier to see the improvement.
        random_vector_for_generation = tf.random.normal(shape=[cfg.num_examples_to_generate, cfg.latent_dim])

        print("Training Begins")

        for epoch in trange(1, cfg.epochs + 1):
    
            for train_x in self.train_dataset:

                with tf.GradientTape() as tape:

                    mean, logvar = self.model.encode(train_x)
                    z = self.model.reparameterize(mean, logvar)
                    x_logit = self.model.decode(z)

                    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=train_x)
                    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
                    logpz = self.log_normal_pdf(z, 0., 0.)
                    logqz_x = self.log_normal_pdf(z, mean, logvar)

                    loss = -tf.reduce_mean(logpx_z + logpz - logqz_x)

                self.logger(loss)

                gradients = tape.gradient(loss, self.model.trainable_variables)
                gradient_and_vars = zip(gradients, self.model.trainable_variables)

                self.optimizer.apply_gradients(gradient_and_vars)
                self.global_step += 1
                
            self.generate_and_save_images(self.model, epoch, random_vector_for_generation)

        if cfg.make_gif == True:
            self.make_gif()



def main():
    
    tf.config.gpu.set_per_process_memory_fraction(0.4)

    cvae = ConvolutionalVariationalAutoencoder()
    cvae.train()


if __name__ == '__main__':
    main()

