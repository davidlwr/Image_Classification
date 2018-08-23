import tensorflow as tf

# https://arxiv.org/pdf/1612.01452.pdf

# https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md
    # Regarding DB before or after activation. 
    # Regarding type of activation: ReLu, PReLU, RReLU...etc

# ReLU intitialized HeNormal: https://arxiv.org/pdf/1502.01852.pdf

class Alexnet_bn(object):

    def __init__(self, scope_name, image_width = 224, image_height = 224, image_depth = 1, num_labels = 17, learning_rate = 0.05, phase=0):
        self.scope_name    = scope_name
        self.learning_rate = learning_rate
        self.image_width   = image_width
        self.image_height  = image_height
        self.image_depth   = image_depth
        self.num_labels    = num_labels
        self.phase         = phase # 0=inference, 1=train

        EPSILON = 1e-3      # For batch normalization ops

        ALEX_FILTER_DEPTH_1, ALEX_FILTER_DEPTH_2, ALEX_FILTER_DEPTH_3 = 96, 256, 384
        ALEX_FILTER_SIZE_1, ALEX_FILTER_SIZE_2, ALEX_FILTER_SIZE_3, ALEX_FILTER_SIZE_4 = 11, 5, 3, 3
        ALEX_NUM_HIDDEN_1, ALEX_NUM_HIDDEN_2 = 4096, 4096

        with tf.variable_scope(self.scope_name):

            # WEIGHTS AND BIASES ==========================================================================
            self.w1 = tf.get_variable(name="w1", shape=[ALEX_FILTER_SIZE_1, ALEX_FILTER_SIZE_1, image_depth, ALEX_FILTER_DEPTH_1], initializer=tf.keras.initializers.he_normal())
            self.b1 = tf.get_variable(name="b1", shape=[ALEX_FILTER_DEPTH_1], initializer=tf.initializers.zeros())

            self.w2 = tf.get_variable(name="w2", shape=[ALEX_FILTER_SIZE_2, ALEX_FILTER_SIZE_2, ALEX_FILTER_DEPTH_1, ALEX_FILTER_DEPTH_2], initializer=tf.keras.initializers.he_normal())
            self.b2 = tf.get_variable(name="b2", shape=[ALEX_FILTER_DEPTH_2], initializer=tf.ones_initializer())

            self.w3 = tf.get_variable(name="w3", shape=[ALEX_FILTER_SIZE_3, ALEX_FILTER_SIZE_3, ALEX_FILTER_DEPTH_2, ALEX_FILTER_DEPTH_3], initializer=tf.keras.initializers.he_normal())
            self.b3 = tf.get_variable(name="b3", shape=[ALEX_FILTER_DEPTH_3], initializer=tf.initializers.zeros())

            self.w4 = tf.get_variable(name="w4", shape=[ALEX_FILTER_SIZE_4, ALEX_FILTER_SIZE_4, ALEX_FILTER_DEPTH_3, ALEX_FILTER_DEPTH_3], initializer=tf.keras.initializers.he_normal())
            self.b4 = tf.get_variable(name="b4", shape=[ALEX_FILTER_DEPTH_3], initializer=tf.ones_initializer())
            
            self.w5 = tf.get_variable(name="w5", shape=[ALEX_FILTER_SIZE_4, ALEX_FILTER_SIZE_4, ALEX_FILTER_DEPTH_3, ALEX_FILTER_DEPTH_3], initializer=tf.keras.initializers.he_normal())
            self.b5 = tf.get_variable(name="b5", shape=[ALEX_FILTER_DEPTH_3], initializer=tf.initializers.zeros())
               
            self.pool_reductions = 3
            self.conv_reductions = 2
            self.no_reductions = self.pool_reductions + self.conv_reductions
            self.w6 = tf.get_variable(name="w6", shape=[(self.image_width // 2**self.no_reductions)*(self.image_height // 2**self.no_reductions)*ALEX_FILTER_DEPTH_3, ALEX_NUM_HIDDEN_1], initializer=tf.keras.initializers.he_normal())
            self.b6 = tf.get_variable(name="b6", shape=[ALEX_NUM_HIDDEN_1], initializer=tf.ones_initializer())

            self.w7 = tf.get_variable(name="w7", shape=[ALEX_NUM_HIDDEN_1, ALEX_NUM_HIDDEN_2], initializer=tf.keras.initializers.he_normal())
            self.b7 = tf.get_variable(name="b7", shape=[ALEX_NUM_HIDDEN_2], initializer=tf.ones_initializer())
            
            self.w8 = tf.get_variable(name="w8", shape=[ALEX_NUM_HIDDEN_2, self.num_labels], initializer=tf.keras.initializers.he_normal())
            self.b8 = tf.get_variable(name="b8", shape=[num_labels], initializer=tf.ones_initializer())

            # LAYERS ====================================================================================
            # Placeholders
            self.tf_data = tf.placeholder(tf.float32, shape=(None, self.image_width, self.image_height, self.image_depth))
            self.tf_labels = tf.placeholder(tf.float32, shape = (None, self.num_labels))

            # Layer 1
            self.layer1_conv = tf.nn.conv2d(self.tf_data, self.w1, [1, 4, 4, 1], padding='SAME')
            self.layer1_relu = tf.nn.relu(self.layer1_conv + self.b1)
            self.layer1_mean, self.layer1_var = tf.nn.moments(self.layer1_relu, [0,1,2])
            self.layer1_bn   = tf.nn.batch_normalization(x=self.layer1_relu, mean=self.layer1_mean, variance=self.layer1_var, offset=None, scale=None, variance_epsilon=EPSILON)
            self.layer1_pool = tf.nn.max_pool(self.layer1_bn, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')
            
            # Layer 2
            self.layer2_conv = tf.nn.conv2d(self.layer1_pool, self.w2, [1, 1, 1, 1], padding='SAME')
            self.layer2_relu = tf.nn.relu(self.layer2_conv + self.b2)
            self.layer2_mean, self.layer2_var = tf.nn.moments(self.layer2_relu, [0,1,2])
            self.layer2_bn   = tf.nn.batch_normalization(x=self.layer2_relu, mean=self.layer2_mean, variance=self.layer2_var, offset=None, scale=None, variance_epsilon=EPSILON)
            self.layer2_pool = tf.nn.max_pool(self.layer2_bn, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')
            
            # Layer 3
            self.layer3_conv = tf.nn.conv2d(self.layer2_pool, self.w3, [1, 1, 1, 1], padding='SAME')
            self.layer3_relu = tf.nn.relu(self.layer3_conv + self.b3)
            self.layer3_mean, self.layer3_var = tf.nn.moments(self.layer3_relu, [0,1,2])
            self.layer3_bn   = tf.nn.batch_normalization(x=self.layer3_relu, mean=self.layer3_mean, variance=self.layer3_var, offset=None, scale=None, variance_epsilon=EPSILON)
            
            # Layer 4
            self.layer4_conv = tf.nn.conv2d(self.layer3_bn, self.w4, [1, 1, 1, 1], padding='SAME')
            self.layer4_relu = tf.nn.relu(self.layer4_conv + self.b4)
            self.layer4_mean, self.layer4_var = tf.nn.moments(self.layer4_relu, [0,1,2])
            self.layer4_bn   = tf.nn.batch_normalization(x=self.layer4_relu, mean=self.layer4_mean, variance=self.layer4_var, offset=None, scale=None, variance_epsilon=EPSILON)
            
            # Layer 5
            self.layer5_conv = tf.nn.conv2d(self.layer4_bn, self.w5, [1, 1, 1, 1], padding='SAME')
            self.layer5_relu = tf.nn.relu(self.layer5_conv + self.b5)
            self.layer5_mean, self.layer5_var = tf.nn.moments(self.layer5_relu, [0,1,2])
            self.layer5_bn   = tf.nn.batch_normalization(x=self.layer5_relu, mean=self.layer5_mean, variance=self.layer5_var, offset=None, scale=None, variance_epsilon=EPSILON)
            self.layer5_pool = tf.nn.max_pool(self.layer5_bn, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')
            
            # FULLY CONNECTED LAYERS (FF)
            # Layer 6
            self.flat_layer = tf.contrib.layers.flatten(self.layer5_pool)
            self.layer6_fccd = tf.matmul(self.flat_layer, self.w6) + self.b6
            self.layer6_relu = tf.nn.relu(self.layer6_fccd)
            self.layer6_mean, self.layer6_var = tf.nn.moments(self.layer6_relu, [0])
            self.layer6_bn   = tf.nn.batch_normalization(x=self.layer6_relu, mean=self.layer6_mean, variance=self.layer6_var, offset=None, scale=None, variance_epsilon=EPSILON)
            self.layer6_drop = tf.layers.dropout(inputs=self.layer6_bn, rate=0.5, training=self.phase)

            # Layer 7 w7
            self.layer7_fccd = tf.matmul(self.layer6_drop, self.w7) + self.b6
            self.layer7_relu = tf.nn.relu(self.layer7_fccd)
            self.layer7_mean, self.layer7_var = tf.nn.moments(self.layer7_relu, [0])
            self.layer7_bn   = tf.nn.batch_normalization(x=self.layer7_relu, mean=self.layer7_mean, variance=self.layer7_var, offset=None, scale=None, variance_epsilon=EPSILON)
            
            # Layer 8
            self.logits = tf.matmul(self.layer7_bn, self.w8) + self.b8

            # Prob
            self.prediction_op = tf.nn.softmax(self.logits)
            self.argmax = tf.argmax(self.prediction_op, 1)

            # Loss
            self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.tf_labels))

            # self.optimizer_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_op)
            self.optimizer_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss_op)

    def get_flattened_op(self):
        return self.flat_layer

    def get_inference_op(self):
        return self.logits

    def get_prediction_op(self):
        return self.prediction_op

    def get_loss_op(self):
        return self.loss_op


    def get_optimizer_op(self):
        return self.optimizer_op


    def get_variables(self):
        return {'w1': self.w1, 'w2': self.w2, 'w3': self.w3, 'w4': self.w4, \
                'w5': self.w5, 'w6': self.w6, 'w7': self.w7, 'w8': self.w8, \
                'b1': self.b1, 'b2': self.b2, 'b3': self.b3, 'b4': self.b4, \
                'b5': self.b5, 'b6': self.b6, 'b7': self.b7, 'b8': self.b8  }


    def get_saver(self):
        return tf.train.Saver({'w1': self.w1, 'w2': self.w2, 'w3': self.w3, 'w4': self.w4, \
                               'w5': self.w5, 'w6': self.w6, 'w7': self.w7, 'w8': self.w8, \
                               'b1': self.b1, 'b2': self.b2, 'b3': self.b3, 'b4': self.b4, \
                               'b5': self.b5, 'b6': self.b6, 'b7': self.b7, 'b8': self.b8  })

