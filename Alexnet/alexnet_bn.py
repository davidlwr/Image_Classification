import tensorflow as tf

# https://arxiv.org/pdf/1612.01452.pdf

class Alexnet_bn(object):

    def __init__(self, scope_name, image_width = 224, image_height = 224, image_depth = 1, num_labels = 17, learning_rate = 0.05):
        self.scope_name    = scope_name
        self.learning_rate = learning_rate
        self.image_width   = image_width
        self.image_height  = image_height
        self.image_depth   = image_depth
        self.num_labels    = num_labels

        EPSILON = 1e-3      # For batch normalization ops

        with tf.variable_scope(self.scope_name):

            # WEIGHTS AND BIASES ==========================================================================
            self.w1 = tf.Variable(name="w1", initial_value=tf.truncated_normal([11, 11, image_depth, 96], stddev=0.1))
            self.b1 = tf.Variable(name="b1", initial_value=tf.zeros([96]))

            self.w2 = tf.Variable(name="w2", initial_value=tf.truncated_normal([5, 5, 96, 256], stddev=0.1))
            self.b2 = tf.Variable(name="w2", initial_value=tf.constant(1.0, shape=[256]))

            self.w3 = tf.Variable(name="w3", initial_value=tf.truncated_normal([3, 3, 256, 384], stddev=0.1))
            self.b3 = tf.Variable(name="b3", initial_value=tf.zeros([384]))

            self.w4 = tf.Variable(name="w4", initial_value=tf.truncated_normal([3, 3, 384, 384], stddev=0.1))
            self.b4 = tf.Variable(name="b4", initial_value=tf.constant(1.0, shape=[384]))
            
            # self.w5 = tf.Variable(name="w5", initial_value=tf.truncated_normal([3, 3, 384, 384], stddev=0.1))  // pretty sure this is wrong
            self.w5 = tf.Variable(name="w5", initial_value=tf.truncated_normal([3, 3, 384, 256], stddev=0.1))
            self.b5 = tf.Variable(name="b5", initial_value=tf.zeros([256]))
               
            self.pool_reductions = 3
            self.conv_reductions = 2
            self.no_reductions = self.pool_reductions + self.conv_reductions
            self.w6 = tf.Variable(name="w6", initial_value=tf.truncated_normal([(self.image_width // 2**self.no_reductions)*(self.image_height // 2**self.no_reductions)*256, 4096], stddev=0.1))
            self.b6 = tf.Variable(name="b6", initial_value=tf.constant(1.0, shape = [4096]))

            self.w7 = tf.Variable(name="w7", initial_value=tf.truncated_normal([4096, 4096], stddev=0.1))
            self.b7 = tf.Variable(name="b7", initial_value=tf.constant(1.0, shape = [4096]))
            
            self.w8 = tf.Variable(name="w8", initial_value=tf.truncated_normal([4096, self.num_labels], stddev=0.1))
            self.b8 = tf.Variable(name="b8", initial_value=tf.constant(1.0, shape = [num_labels]))


            # LAYERS ====================================================================================
            # batch normalization layers added as per article: "Imagenet pretrained models with batch normalization. Simon, Rodner..."
            # Placeholders
            self.tf_data = tf.placeholder(tf.float32, shape=(None, self.image_width, self.image_height, self.image_depth))
            self.tf_labels = tf.placeholder(tf.float32, shape = (None, self.num_labels))

            # Layer 1
            self.layer1_conv = tf.nn.conv2d(self.tf_data, self.w1, [1, 4, 4, 1], padding='SAME')
            self.layer1_mean, self.layer1_var = tf.nn.moments(self.layer1_conv, [0,1,2])
            self.layer1_bn   = tf.nn.batch_normalization(x=self.layer1_conv, mean=self.layer1_mean, variance=self.layer1_var, offset=None, scale=None, variance_epsilon=EPSILON)
            self.layer1_relu = tf.nn.relu(self.layer1_bn + self.b1)
            self.layer1_pool = tf.nn.max_pool(self.layer1_relu, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')
            
            # Layer 2
            self.layer2_conv = tf.nn.conv2d(self.layer1_pool, self.w2, [1, 1, 1, 1], padding='SAME')
            self.layer2_mean, self.layer2_var = tf.nn.moments(self.layer2_conv, [0,1,2])
            self.layer2_bn   = tf.nn.batch_normalization(x=self.layer2_conv, mean=self.layer2_mean, variance=self.layer2_var, offset=None, scale=None, variance_epsilon=EPSILON)
            self.layer2_relu = tf.nn.relu(self.layer2_bn + self.b2)
            self.layer2_pool = tf.nn.max_pool(self.layer2_relu, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')
            
            # Layer 3
            self.layer3_conv = tf.nn.conv2d(self.layer2_pool, self.w3, [1, 1, 1, 1], padding='SAME')
            self.layer3_mean, self.layer3_var = tf.nn.moments(self.layer3_conv, [0,1,2])
            self.layer3_bn   = tf.nn.batch_normalization(x=self.layer3_conv, mean=self.layer3_mean, variance=self.layer3_var, offset=None, scale=None, variance_epsilon=EPSILON)
            self.layer3_relu = tf.nn.relu(self.layer3_bn + self.b3)
            
            # Layer 4
            self.layer4_conv = tf.nn.conv2d(self.layer3_relu, self.w4, [1, 1, 1, 1], padding='SAME')
            self.layer4_mean, self.layer4_var = tf.nn.moments(self.layer4_conv, [0,1,2])
            self.layer4_bn   = tf.nn.batch_normalization(x=self.layer4_conv, mean=self.layer4_mean, variance=self.layer4_var, offset=None, scale=None, variance_epsilon=EPSILON)
            self.layer4_relu = tf.nn.relu(self.layer4_bn + self.b4)
            
            # Layer 5
            self.layer5_conv = tf.nn.conv2d(self.layer4_relu, self.w5, [1, 1, 1, 1], padding='SAME')
            self.layer5_mean, self.layer5_var = tf.nn.moments(self.layer5_conv, [0,1,2])
            self.layer5_bn   = tf.nn.batch_normalization(x=self.layer5_conv, mean=self.layer5_mean, variance=self.layer5_var, offset=None, scale=None, variance_epsilon=EPSILON)
            self.layer5_relu = tf.nn.relu(self.layer5_bn + self.b5)
            self.layer5_pool = tf.nn.max_pool(self.layer5_relu, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')
            
            # FULLY CONNECTED LAYERS (FF)
            # Layer 6
            self.flat_layer = tf.layers.flatten(self.layer5_pool)
            self.layer6_fccd = tf.matmul(self.flat_layer, self.w6) + self.b6
            self.layer6_mean, self.layer6_var = tf.nn.moments(self.layer6_fccd, [0])
            self.layer6_bn   = tf.nn.batch_normalization(x=self.layer6_fccd, mean=self.layer6_mean, variance=self.layer6_var, offset=None, scale=None, variance_epsilon=EPSILON)
            # self.layer6_tanh = tf.tanh(self.layer6_bn)
            self.layer6_relu = tf.nn.relu(self.layer6_bn)
            
            # Layer 7 w7
            self.layer7_fccd = tf.matmul(self.layer6_relu, self.w7) + self.b6
            self.layer7_mean, self.layer7_var = tf.nn.moments(self.layer7_fccd, [0])
            self.layer7_bn   = tf.nn.batch_normalization(x=self.layer7_fccd, mean=self.layer7_mean, variance=self.layer7_var, offset=None, scale=None, variance_epsilon=EPSILON)
            # self.layer7_tanh = tf.tanh(self.layer7_bn)
            self.layer7_relu = tf.nn.relu(self.layer7_bn)
            
            # Layer 8
            self.logits = tf.matmul(self.layer7_relu, self.w8) + self.b8

            # Prob
            self.prediction_op = tf.nn.softmax(self.logits)
            self.argmax = tf.argmax(self.prediction_op, 1)

            # Loss
            self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.tf_labels))

            #self.optimizer_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_op)
            self.optimizer_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss_op)

    def get_flattened_op(self):
        return self.flat_layer

    def get_inference_op(self):
        return self.logits


    def get_loss_op(self):
        return self.loss_op


    def get_optimizer_op(self):
        return self.optimizer_op

    def get_prediction_op(self):
        return self.prediction_op


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