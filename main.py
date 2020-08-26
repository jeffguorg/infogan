import tensorflow.compat.v1 as tf
import numpy as np
import argparse
from PIL import Image
from models import Discriminator, Decoder
from loader import DataManager


def preprocess(image):
    image = image / 255.0
    image = (image - 0.5) / 0.5
    return image


def create_dataset(w, h, c):
    input_tensor = tf.placeholder(tf.float32, [None, w, h, c])
    dataset = tf.data.Dataset.from_tensor_slices(input_tensor)
    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(10000)
    dataset = dataset.batch(args.batch_size).repeat(10)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    iterator = dataset.make_initializable_iterator()
    next_batch = iterator.get_next()
    return input_tensor, next_batch, iterator.initializer

def eval(gan, dm):
    num = 10
    i=2
    u1 = np.random.uniform(-1.0, 1.0, [1, 62])
    u2 = np.random.uniform(-1., 1., [1, 1])
    u3 = np.random.uniform(-1., 1., [1, 1])
    z1 = np.concatenate([u1.repeat(num, axis=0),
                         np.eye(10),
                         u2.repeat(num, axis=0),
                         u3.repeat(num, axis=0)], axis=-1)

    z2 = np.concatenate([u1.repeat(num, axis=0),
                         np.eye(10)[i:i+1].repeat(num, axis=0),
                         np.linspace(-1.0, 1.0, num).reshape([-1, 1]),
                         u3.repeat(num, axis=0)], axis=-1)

    z3 = np.concatenate([u1.repeat(num, axis=0),
                         np.eye(10)[i:i+1].repeat(num, axis=0),
                         u2.repeat(num, axis=0),
                         np.linspace(-1.0, 1.0, num).reshape([-1, 1])], axis=-1)
    z4 = np.concatenate([np.random.uniform(-1.0, 1.0, [num, 62]),
                         np.eye(10)[i:i+1].repeat(num, axis=0),
                         u2.repeat(num, axis=0),
                         u3.repeat(num, axis=0)], axis=-1)
    batch, _ = dm.sample(num)
    real_code, fake_img = gan.get_code_img((batch / 255.0 - 0.5) / 0.5, np.concatenate([z1, z2, z3, z4]))
    idx = np.squeeze(np.argmax(real_code[:, :10], axis=-1))
    z_recon = np.concatenate([np.random.uniform(-1.0, 1.0, [num, 62]), np.eye(10)[idx], real_code[:, 11:12], real_code[:, 13:14]], axis=-1)
    ss = tf.get_default_session()
    recon = ss.run(gan.fake_img, feed_dict={gan.input_z: z_recon})
    recon = ((np.squeeze(recon) * 0.5 + 0.5) * 255).astype("uint8")
    i1, i2, i3, i4 = np.split(((np.squeeze(fake_img) * 0.5 + 0.5) * 255).astype("uint8"), 4)

    samples = gan.sample(100)
    samples = ((np.squeeze(samples) * 0.5 + 0.5) * 255).astype("uint8")
    samples = np.concatenate([np.concatenate(list(samples[i*10:(i+1)*10]), axis=1) for i in range(10)], axis=0)
    image = Image.fromarray(samples)
    image.save("1.png")

    x = np.concatenate(list(i1[:10]), axis=1)
    img = Image.fromarray(x)
    img.save("eval/fake/z1.png")
    x = np.concatenate(list(i2[:10]), axis=1)
    img = Image.fromarray(x)
    img.save("eval/fake/z2.png")
    x = np.concatenate(list(i3[:10]), axis=1)
    img = Image.fromarray(x)
    img.save("eval/fake/z3.png")
    x = np.concatenate(list(i4[:10]), axis=1)
    img = Image.fromarray(x)
    img.save("eval/fake/z4.png")
    for i, (x, y) in enumerate(zip(np.squeeze(batch[:10]).astype("uint8"), recon[:10])):
        arr = np.concatenate([x, y], axis=1)
        img = Image.fromarray(arr)
        img.save("eval/reconstruct/{}.png".format(i))


class InfoGAN:
    def __init__(self, x, training=True):
        self.categories = 10
        self.cont_dim = 1
        self.regression_dim = 10 + 2 + 2
        self.prior_z = tf.random_uniform([args.batch_size, 62], minval=-1.0, maxval=1.0)
        self.prior_c1 = tf.one_hot(tf.squeeze(tf.random.categorical(tf.log(tf.ones([1, self.categories]) * 0.1), args.batch_size)), self.categories)

        self.prior_c2 = tf.random.uniform([args.batch_size, self.cont_dim], minval=-1.0, maxval=1.0)
        self.prior_c3 = tf.random.uniform([args.batch_size, self.cont_dim], minval=-1.0, maxval=1.0)
        cat_prior = tf.concat([self.prior_z, self.prior_c1, self.prior_c2, self.prior_c3], axis=-1)
        self.discriminator = Discriminator()
        self.decoder = Decoder(62 + 10 + 2)
        self.train_disc_op, self.train_dec_op = self.build_train_op(cat_prior, x, training)
        self.build_inference()

    def build_train_op(self, prior, x, training=True):

        self.fake_sample = fake_img = self.decoder.build_deconvnet(prior, training=training, device_id=0)
        fake_val, fake_enc = self.discriminator.build_convnet(fake_img, self.regression_dim, training=training, device_id=0)

        real_val, _ = self.discriminator.build_convnet(x, reuse=True, training=training, device_id=0)
        logits_c1 = fake_enc[:, :self.categories]
        mean_c2, log_std_c2 = tf.split(fake_enc[:, self.categories:self.categories+self.cont_dim*2], 2, axis=-1)
        mean_c3, log_std_c3 = tf.split(fake_enc[:, self.categories+self.cont_dim*2:self.categories+self.cont_dim*4], 2, axis=-1)
        # dist_c2 = tf.distributions.Normal(mean_c2, tf.exp(log_std_c2))
        # dist_c3 = tf.distributions.Normal(mean_c3, tf.exp(log_std_c3))
        # dist_c2 = tf.distributions.Normal(mean_c2, tf.ones_like(log_std_c2))
        deg = tf.nn.sigmoid(mean_c2) * np.pi
        mean_c3 = tf.nn.tanh(mean_c3)
        # dist_c3 = tf.distributions.Normal(mean_c3, tf.ones_like(log_std_c3) * 0.4)

        self.a = cond_ent1 = -tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(self.prior_c1, logits_c1))

        # self.b = cond_ent2 = tf.reduce_mean(dist_c2.log_prob(self.prior_c2))
        self.b = cond_ent2 = -tf.reduce_mean((tf.cos(deg) - self.prior_c2)**2)
        # self.c = cond_ent3 = tf.reduce_mean(dist_c3.log_prob(self.prior_c3))
        self.c = cond_ent3 = -tf.reduce_mean((mean_c3 - self.prior_c3)**2)

        self.mutual_info = mutual_info = cond_ent1 + cond_ent2 + cond_ent3
        disc_update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="enc_conv_GPU_0")
        dec_update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="dec_deconv_GPU_0")
        self.decoder_loss = decoder_loss = -tf.reduce_mean(tf.log(fake_val + 1e-8))
        self.t1 = t1 = tf.reduce_mean(tf.log(real_val + 1e-8))
        self.t2 = t2 = tf.reduce_mean(tf.log(1. - fake_val + 1e-8))

        self.discriminator_loss = discriminator_loss = - t1 - t2

        disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="enc_conv_GPU_0")
        dec_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="dec_deconv_GPU_0")
        with tf.control_dependencies(disc_update_op):
            train_disc_op = tf.train.AdamOptimizer(2e-4, 0.5).minimize(discriminator_loss - mutual_info, var_list=disc_vars)
        with tf.control_dependencies(dec_update_op):
            train_dec_op = tf.train.AdamOptimizer(1e-3, 0.5).minimize(decoder_loss - mutual_info, var_list=dec_vars)
        return train_disc_op, train_dec_op

    def train_step(self):
        session = tf.get_default_session()
        _, disc_loss, t1, t2, mi,  = session.run([self.train_disc_op, self.discriminator_loss, self.t1, self.t2, self.mutual_info])
        _, dec_loss, a, b, c = session.run([self.train_dec_op, self.decoder_loss, self.a, self.b, self.c])
        return [disc_loss, t1, t2, dec_loss, mi, a, b, c]

    def sample(self, num):
        session = tf.get_default_session()
        z = np.concatenate([np.random.uniform(-1.0, 1.0, [num, 62]),
                            np.eye(10)[np.random.choice(10, num)],
                            np.random.uniform(-1.0, 1.0, [num, 2])], axis=-1)
        return session.run(self.fake_img, feed_dict={self.input_z: z})

    def build_inference(self):
        self.input_img = tf.placeholder(tf.float32, [None, 28, 28, 1])
        _, self.real_code = self.discriminator.build_convnet(self.input_img, 14, reuse=True, training=False, device_id=0)

        self.input_z = tf.placeholder(tf.float32, [None, 64+10])
        self.fake_img = self.decoder.build_deconvnet(self.input_z, reuse=True, training=False, device_id=0)

    def get_code_img(self, x, z):
        session = tf.get_default_session()
        feed_dict = {self.input_img: x, self.input_z: z}
        real_code, fake_img = session.run([self.real_code, self.fake_img], feed_dict=feed_dict)
        return real_code, fake_img



tf.disable_v2_behavior()
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64)
parser.add_argument("--epochs", default=12)
parser.add_argument("--eval", default=False)
args = parser.parse_args()


dm = DataManager("datasets")
dm.load_MNIST(NHWC=True)
iterations = dm.train_input.shape[0] // args.batch_size + 1
input_tensor, next_batch, initializer = create_dataset(dm.w, dm.h, dm.c)

if args.eval:
    gan = InfoGAN(next_batch, training=False)
else:
    gan = InfoGAN(next_batch)
saver = tf.train.Saver()
ss = tf.InteractiveSession()

if args.eval:
    saver.restore(ss, "trained_model/infogan.ckpt")
    eval(gan, dm)
else:
    log_file = open("train.log", "w")
    ss.run(tf.global_variables_initializer())
    for ep in range(args.epochs):
        if ep % 10 == 0:
            ss.run(initializer, feed_dict={input_tensor: dm.train_input})
        for i in range(iterations):
            result = gan.train_step()
            log_info = "[ep {}/{}] [it {}/{}] disc/dec_loss: {:.4f}({:.4f}, {:.4f})/{:.4f} mutual_info: {:.4f} = {:.4f}, {:.4f}, {:.4f}".format(ep+1, args.epochs, i+1, iterations, *result)
            print(log_info)
            log_file.write(log_info + "\n")
        log_file.flush()
        # if (ep+1) % 10 == 0:

    log_file.close()
    saver.save(ss, "trained_model/infogan.ckpt")
    print("model saved.")
    print("evaluating...")
    eval(gan, dm)
    print("done.")




