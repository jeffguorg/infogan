import tensorflow.compat.v1 as tf


class Discriminator:
    def __init__(self, num_channels=64):
        self.num_channels = num_channels

    def build_convnet(self, input, regression_dim=None, reuse=False, training=True, device_id=None):
        assert device_id is not None, "need to specify a device."
        with tf.variable_scope("enc_conv_GPU_{}".format(device_id), reuse=reuse):

            # x = tf.layers.conv2d(input, self.num_channels, 4, 2, "same", use_bias=False)  # 28
            # x = tf.layers.batch_normalization(x, training=training)
            # x = tf.nn.leaky_relu(x)  # 14
            # 
            # x = tf.layers.conv2d(x, self.num_channels * 2, 4, 2, "same", use_bias=False)
            # x = tf.layers.batch_normalization(x, training=training)
            # x = tf.nn.leaky_relu(x)  # 7
            # 
            # x = tf.layers.conv2d(x, self.num_channels * 4, 4, 2, "same", use_bias=False)
            # x = tf.layers.batch_normalization(x, training=training)
            # x = tf.nn.leaky_relu(x)  # 4
            # 
            # x = tf.layers.conv2d(x, self.num_channels * 8, 4, 2, "same", use_bias=False)
            # x = tf.layers.batch_normalization(x, training=training)
            # x = tf.nn.leaky_relu(x)  # 2
            # 
            # x = tf.layers.average_pooling2d(x, 2, 1)
            # x = tf.reshape(x, [-1, self.num_channels * 8])
            # 
            # x = tf.layers.dense(x, 128, use_bias=False)
            # x = tf.layers.batch_normalization(x, training=training)
            # x = tf.nn.leaky_relu(x)
            # discriminator_out = tf.layers.dense(x, 1, activation=tf.nn.sigmoid)
            # 
            # encoder_out = None
            # if regression_dim is not None:
            #     x = tf.layers.dense(x, 64, use_bias=False)
            #     x = tf.layers.batch_normalization(x, training=training)
            #     x = tf.nn.leaky_relu(x)
            #     encoder_out = tf.layers.dense(x, regression_dim)

            x = tf.layers.conv2d(input, self.num_channels, 4, 2, "same")
            x = tf.nn.leaky_relu(x)

            x = tf.layers.conv2d(x, 2*self.num_channels, 4, 2, "same", use_bias=False)
            x = tf.layers.batch_normalization(x, training=training)
            x = tf.nn.leaky_relu(x)
            x = tf.reshape(x, [-1, 7*7*2*self.num_channels])
            x = tf.layers.dense(x, 1024, use_bias=False)
            x = tf.layers.batch_normalization(x, training=training)
            x = tf.nn.leaky_relu(x)
            discriminator_out = tf.layers.dense(x, 1, activation=tf.nn.sigmoid)
            encoder_out = None
            if regression_dim is not None:
                x = tf.layers.dense(x, 128, use_bias=False)
                x = tf.layers.batch_normalization(x, training=training)
                x = tf.nn.leaky_relu(x)
                encoder_out = tf.layers.dense(x, regression_dim)

        return discriminator_out, encoder_out


class Decoder:
    def __init__(self, embedding_dim):
        self.embedding_dim = embedding_dim

    def build_deconvnet(self, input, reuse=False, training=True, device_id=None):
        assert device_id is not None, "need to specify a device."
        with tf.variable_scope("dec_deconv_GPU_{}".format(device_id), reuse=reuse):

            # x = tf.layers.dense(input, 128, use_bias=False)
            # x = tf.layers.batch_normalization(x, training=training)
            # x = tf.nn.relu(x)
            # x = tf.layers.dense(x, 128, use_bias=False)
            # x = tf.layers.batch_normalization(x, training=training)
            # x = tf.nn.relu(x)
            # x = tf.reshape(x, [-1, 1, 1, 128])
            # x = tf.layers.conv2d_transpose(x, 256, 3, 1, "valid", use_bias=False)
            # x = tf.layers.batch_normalization(x, training=training)
            # x = tf.nn.relu(x)
            # x = tf.layers.conv2d_transpose(x, 128, 3, 2, "valid", use_bias=False)
            # x = tf.layers.batch_normalization(x, training=training)
            # x = tf.nn.relu(x)
            # x = tf.layers.conv2d_transpose(x, 64, 4, 2, "same", use_bias=False)
            # x = tf.layers.batch_normalization(x, training=training)
            # x = tf.nn.relu(x)
            # x = tf.layers.conv2d_transpose(x, 1, 4, 2, "same", use_bias=True, activation=tf.nn.tanh)
            # x = tf.layers.batch_normalization(x, training=training)
            # x = tf.nn.relu(x)
            # x = tf.layers.conv2d(x, 1, 1, 1, "same", activation=tf.nn.tanh)
            x = tf.layers.dense(input, 1024, use_bias=False)
            x = tf.layers.batch_normalization(x, training=training,)
            x = tf.nn.relu(x)
            x = tf.layers.dense(x, 7*7*128, use_bias=False)
            x = tf.layers.batch_normalization(x, training=training)
            x = tf.nn.relu(x)
            x = tf.reshape(x, [-1, 7, 7, 128])
            x = tf.layers.conv2d_transpose(x, 64, 4, 2, "same", use_bias=False)
            x = tf.layers.batch_normalization(x, training=training)
            x = tf.nn.relu(x)
            x = tf.layers.conv2d_transpose(x, 1, 4, 2, "same", activation=tf.nn.tanh)
        return x

if __name__=="__main__":
    import numpy as np
    tf.disable_v2_behavior()
    with tf.variable_scope("test"):
        x = tf.random_normal([10, 5], name="randn")
        x = tf.layers.batch_normalization(x)
        x = tf.layers.batch_normalization(x)
    ss=tf.Session()
    ss.run(tf.global_variables_initializer())
    c=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="test")
    # with tf.variable_scope("test", reuse=True):
    #     v=tf.get_variable("batch_norm/moving_mean")
    # print(ss.run(v))
    print(c)
