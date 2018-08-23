import tensorflow as tf
import sys

#The VGGNET-16 Neural Network 
# https://arxiv.org/pdf/1409.1556.pdf : Very Deep Convolutional Networks for Large-Scale Image Recognition
                                        # K. Simonyan, A. Zisserman
                                        # arXiv:1409.1556

# SELU: https://arxiv.org/pdf/1706.02515.pdf
#     - https://github.com/bioinf-jku/SNNs/blob/master/Keras-CNN/MNIST-Conv-SELU.py

class VGG16_selu(object):
    '''
    Initializing this class creates a model of VGG16

    Keyword arguments
    scope_name   -- Scope to place this graph under
    image_width  -- int of input img (default = 224)
    image_height -- int of input img (default = 224)
    image_depth  -- int number of channles of input img (default = 1)
    num_labels   -- number of labels involved
    phase        -- 0 = inference / test, 1 == training. This affects how Batch Normalization works
    '''

    def __init__(self, scope_name, image_width = 224, image_height = 224, image_depth = 3, num_labels = 17, phase=0):
        self.scope_name    = scope_name
        self.image_width   = image_width
        self.image_height  = image_height
        self.image_depth   = image_depth
        self.num_labels    = num_labels
        self.phase         = phase

        with tf.variable_scope(self.scope_name):

            self.w1 = tf.get_variable(name="w1", shape=[3, 3, image_depth, 64], initializer=tf.keras.initializers.lecun_normal())
            self.b1 = tf.get_variable(name="b1", shape=[64], initializer=tf.zeros_initializer())
            self.w2 = tf.get_variable(name="w2", shape=[3, 3, 64, 64], initializer=tf.keras.initializers.lecun_normal())
            self.b2 = tf.get_variable(name="b2", shape=[64], initializer=tf.ones_initializer())

            self.w3 = tf.get_variable(name="w3", shape=[3, 3, 64, 128], initializer=tf.keras.initializers.lecun_normal())
            self.b3 = tf.get_variable(name="b3", shape=[128], initializer=tf.ones_initializer())
            self.w4 = tf.get_variable(name="w4", shape=[3, 3, 128, 128], initializer=tf.keras.initializers.lecun_normal())
            self.b4 = tf.get_variable(name="b4", shape=[128], initializer=tf.ones_initializer())

            self.w5 = tf.get_variable(name="w5", shape=[3, 3, 128, 256], initializer=tf.keras.initializers.lecun_normal())
            self.b5 = tf.get_variable(name="b5", shape=[256], initializer=tf.ones_initializer())
            self.w6 = tf.get_variable(name="w6", shape=[3, 3, 256, 256], initializer=tf.keras.initializers.lecun_normal())
            self.b6 = tf.get_variable(name="b6", shape=[256], initializer=tf.ones_initializer())
            self.w7 = tf.get_variable(name="w7", shape=[3, 3, 256, 256], initializer=tf.keras.initializers.lecun_normal())
            self.b7 = tf.get_variable(name="b7", shape=[256], initializer=tf.ones_initializer())

            self.w8 = tf.get_variable(name="w8", shape=[3, 3, 256, 512], initializer=tf.keras.initializers.lecun_normal())
            self.b8 = tf.get_variable(name="b8", shape=[512], initializer=tf.ones_initializer())
            self.w9 = tf.get_variable(name="w9", shape=[3, 3, 512, 512], initializer=tf.keras.initializers.lecun_normal())
            self.b9 = tf.get_variable(name="b9", shape=[512], initializer=tf.ones_initializer())
            self.w10 = tf.get_variable(name="w10", shape=[3, 3, 512, 512], initializer=tf.keras.initializers.lecun_normal())
            self.b10 = tf.get_variable(name="b10", shape=[512], initializer=tf.ones_initializer()) 

            self.w11 = tf.get_variable(name="w11", shape=[3, 3, 512, 512], initializer=tf.keras.initializers.lecun_normal())
            self.b11 = tf.get_variable(name="b11", shape=[512], initializer=tf.ones_initializer())
            self.w12 = tf.get_variable(name="w12", shape=[3, 3, 512, 512], initializer=tf.keras.initializers.lecun_normal())
            self.b12 = tf.get_variable(name="b12", shape=[512], initializer=tf.ones_initializer())
            self.w13 = tf.get_variable(name="w13", shape=[3, 3, 512, 512], initializer=tf.keras.initializers.lecun_normal())
            self.b13 = tf.get_variable(name="b13", shape=[512], initializer=tf.ones_initializer()) 
    
            self.no_pooling_layers = 5

            self.w14 = tf.get_variable(name="w14", shape=[(image_width // (2**self.no_pooling_layers))*(image_height // (2**self.no_pooling_layers))*512 , 4096], initializer=tf.keras.initializers.lecun_normal())
            self.b14 = tf.get_variable(name="b14", shape=[4096], initializer=tf.ones_initializer()) 

            self.w15 = tf.get_variable(name="w15", shape=[4096, 1000], initializer=tf.keras.initializers.lecun_normal())
            self.b15 = tf.get_variable(name="b15", shape=[1000], initializer=tf.ones_initializer()) 
            
            self.w16 = tf.get_variable(name="w16", shape=[1000, num_labels], initializer=tf.keras.initializers.lecun_normal())
            self.b16 = tf.get_variable(name="b16", shape=[num_labels], initializer=tf.ones_initializer()) 

            # LAYERS ====================================================================================
            # Placeholders
            self.tf_data = tf.placeholder(name="img_placeholder", dtype=tf.float32, shape=(None, None, None, None))
            self.input   = self.input_pipe(batch_tensor=self.tf_data, parallel_iterations=5, img_height=self.image_height, img_width=self.image_width)
            self.dropout_prob = tf.placeholder_with_default(1.0, shape=())
            self.epsilon = 1e-3  

            self.layer1_relu  = self.conv_layer(input=self.input, weight=self.w1, bias=self.b1)

            self.layer2_relu  = self.conv_layer(input=self.layer1_relu, weight=self.w2, bias=self.b2)
            self.layer2_pool  = tf.nn.max_pool(self.layer2_relu, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

            self.layer3_relu  = self.conv_layer(input=self.layer2_pool, weight=self.w3, bias=self.b3)

            self.layer4_relu  = self.conv_layer(input=self.layer3_relu, weight=self.w4, bias=self.b4)
            self.layer4_pool  = tf.nn.max_pool(self.layer4_relu, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

            self.layer5_relu  = self.conv_layer(input=self.layer4_pool, weight=self.w5, bias=self.b5)

            self.layer6_relu  = self.conv_layer(input=self.layer5_relu, weight=self.w6, bias=self.b6)

            self.layer7_relu  = self.conv_layer(input=self.layer6_relu, weight=self.w7, bias=self.b7)
            self.layer7_pool  = tf.nn.max_pool(self.layer7_relu, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

            self.layer8_relu  = self.conv_layer(input=self.layer7_pool, weight=self.w8, bias=self.b8)

            self.layer9_relu  = self.conv_layer(input=self.layer8_relu, weight=self.w9, bias=self.b9)

            self.layer10_relu = self.conv_layer(input=self.layer9_relu, weight=self.w10, bias=self.b10)
            self.layer10_pool = tf.nn.max_pool(self.layer10_relu, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

            self.layer11_relu = self.conv_layer(input=self.layer10_pool, weight=self.w11, bias=self.b11)

            self.layer12_relu = self.conv_layer(input=self.layer11_relu, weight=self.w12, bias=self.b12)

            self.layer13_relu = self.conv_layer(input=self.layer12_relu, weight=self.w13, bias=self.b13)
            self.layer13_pool = tf.nn.max_pool(self.layer13_relu, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
            
            # FULLY CONNECTED LAYERS
            self.flat_layer   = tf.contrib.layers.flatten(self.layer13_pool)

            self.layer14_relu = self.feed_forward_layer(input=self.flat_layer, weight=self.w14, bias=self.b14)
            self.layer14_drop = tf.contrib.nn.alpha_dropout(x=self.layer14_relu, keep_prob=self.dropout_prob)

            self.layer15_relu = self.feed_forward_layer(input=self.layer14_drop, weight=self.w15, bias=self.b15)

            self.logits       = tf.matmul(self.layer15_relu, self.w16) + self.b16

            self.softmax      = tf.nn.softmax(self.logits)

            self.argmax       = tf.argmax(self.softmax, 1)


    def input_pipe(self, batch_tensor, img_height, img_width, parallel_iterations=1):
        '''
        Takes in a batch of images / placecholder, then resizes image to img_height * img_width

        Warning: I cant figure out how to auto detect the number of channels the images have, 
                - because when the graph is constructed no images has gone in yet
                - As such, this model will ONLY TAKE IN IMAGES WITH 3 CHANNELS
        '''

        grayscale = tf.image.rgb_to_grayscale(batch_tensor)
        processed = tf.image.resize_images(grayscale, [img_height, img_width], tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return processed



    def conv_layer(self, input, weight, bias):
        '''
        Creates a layer of 1 convolution, ReLU, and batch normalization (optional)
        :param input: Input tensor or op
        :param weight: weight tensor
        :param bias: bias tensor
        '''

        conv = tf.nn.conv2d(input, weight, [1, 1, 1, 1], padding='SAME')
        out = tf.nn.selu(conv + bias)

        return out


    def feed_forward_layer(self, input, weight, bias):
        '''
        Creates a layer of matmul, ReLU, and batch normalization (optional)
        :param input: Input tensor or op
        :param weight: weight tensor
        :param bias: bias tensor
        '''

        matmul = tf.matmul(input, weight) + bias
        out = tf.nn.selu(matmul)

        return out


    def training_step(self, learning_rate=0.01, learning_rate_type="STATIC", optimizer_type="SGD", **kwargs):
        '''
        Creates training ops

        :param learning_rate: Learning rate for optimizers
        :param learning_rate_type: Learning rate schedule, 'STATIC' for constant, 'EXPONENTIAL_DECAY' for exponential decay...
        :param optimizer_type: see method itself for details
        :param **kwargs: used to pass in other stuff that may be needed for optimizers or learning rate schedules
        '''

        with tf.variable_scope(self.scope_name):

            # Placeholder
            self.tf_labels = tf.placeholder(name="label_placeholder", dtype=tf.float32, shape = (None, self.num_labels))

            # Label Argmax
            self.label_argmax = tf.argmax(self.tf_labels, 1)

            # Global Step
            self.global_step = tf.Variable(0, trainable=False, name='global_step')

            # Loss
            self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.tf_labels))

            # Learning Rate Schedules. if not, basic rate is used
            if learning_rate_type == "EXPONENTIAL_DECAY":
                self.learning_rate = tf.train.exponential_decay(learning_rate=self.learning_rate, global_step=self.global_step, decay_steps=kwargs["decay_steps"], decay_rate=kwargs["decay_rate"], staircase=False)
            else:
                self.learning_rate = learning_rate

            # Optimizer
            if optimizer_type == "SGD":
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

            elif optimizer_type == "ADAM":
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

            elif optimizer_type == "RMSPROP":
                self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate,
                                                            decay=0.9, 
                                                            momentum=0.0, 
                                                            epsilon=1e-10, 
                                                            use_locking=False, 
                                                            centered=False)

            elif optimizer_type == "MOMENTUM":
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate,
                                                            momentum=0.05,
                                                            use_locking=False,
                                                            use_nesterov=False)

            self.optmizer_op = self.optimizer.minimize(loss=self.loss_op, global_step=self.global_step)

            return self.loss_op, self.optmizer_op, self.global_step, self.label_argmax
    

    def get_variables(self):
        return {'w1' : self.w1,  'w2' : self.w2,  'w3' : self.w3,  'w4' : self.w4,  \
                'w5' : self.w5,  'w6' : self.w6,  'w7' : self.w7,  'w8' : self.w8,  \
                'w9' : self.w9,  'w10': self.w10, 'w11': self.w11, 'w12': self.w12, \
                'w13': self.w13, 'w14': self.w14, 'w15': self.w15, 'w16': self.w16, \
                'b1' : self.b1,  'b2' : self.b2,  'b3' : self.b3,  'b4' : self.b4,  \
                'b5' : self.b5,  'b6' : self.b6,  'b7' : self.b7,  'b8' : self.b8,  \
                'b9' : self.b9,  'b10': self.b10, 'b11': self.b11, 'b12': self.b12, \
                'b13': self.b13, 'b14': self.b14, 'b15': self.b15, 'b16': self.b16  }

    def get_imgnet_pretrained_var_map(self):
        return {'vgg_16/conv1/conv1_1/biases': self.b1,
                'vgg_16/conv1/conv1_1/weights': self.w1,
                'vgg_16/conv1/conv1_2/biases': self.b2,
                'vgg_16/conv1/conv1_2/weights': self.w2,
                'vgg_16/conv2/conv2_1/biases': self.b3,
                'vgg_16/conv2/conv2_1/weights': self.w3,
                'vgg_16/conv2/conv2_2/biases': self.b4,
                'vgg_16/conv2/conv2_2/weights': self.w4,
                'vgg_16/conv3/conv3_1/biases': self.b5,
                'vgg_16/conv3/conv3_1/weights': self.w5,
                'vgg_16/conv3/conv3_2/biases': self.b6,
                'vgg_16/conv3/conv3_2/weights': self.w6,
                'vgg_16/conv3/conv3_3/biases': self.b7,
                'vgg_16/conv3/conv3_3/weights': self.w7,
                'vgg_16/conv4/conv4_1/biases': self.b8,
                'vgg_16/conv4/conv4_1/weights': self.w8,
                'vgg_16/conv4/conv4_2/biases':self.b9,
                'vgg_16/conv4/conv4_2/weights': self.w9,
                'vgg_16/conv4/conv4_3/biases': self.b10,
                'vgg_16/conv4/conv4_3/weights': self.w10,
                'vgg_16/conv5/conv5_1/biases': self.b11,
                'vgg_16/conv5/conv5_1/weights': self.w11,
                'vgg_16/conv5/conv5_2/biases': self.b12,
                'vgg_16/conv5/conv5_2/weights': self.w12,
                'vgg_16/conv5/conv5_3/biases': self.b13,
                'vgg_16/conv5/conv5_3/weights': self.w13}

