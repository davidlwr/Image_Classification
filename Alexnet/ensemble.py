import tensorflow as tf
# from alexnet_bn import Alexnet_bn
from alexnet_bn_after_relu import Alexnet_bn
import sys

class Ensemble(object):

    def __init__(self, scope_name, image_width = 224, image_height = 224, image_depth = 1, num_labels = 17, learning_rate = 0.05, phase=0):

        self.scope_name = scope_name
        self.image_width = image_width
        self.image_height = image_height
        self.image_depth = image_depth
        self.num_labels = num_labels
        self.learning_rate = learning_rate
        self.phase = phase # 0=inference, 1=train

        with tf.variable_scope(self.scope_name):

            # 5 models
            self.holistic_model = Alexnet_bn(scope_name="holistic", image_width = 224, image_height = 224, image_depth = 1, num_labels = 17, phase=phase)
            self.header_model   = Alexnet_bn(scope_name="header", image_width = 224, image_height = 224, image_depth = 1, num_labels = 17, phase=phase)
            self.footer_model   = Alexnet_bn(scope_name="footer", image_width = 224, image_height = 224, image_depth = 1, num_labels = 17, phase=phase)
            self.left_model     = Alexnet_bn(scope_name="left", image_width = 224, image_height = 224, image_depth = 1, num_labels = 17, phase=phase)
            self.right_model    = Alexnet_bn(scope_name="right", image_width = 224, image_height = 224, image_depth = 1, num_labels = 17, phase=phase)	

            # Placeholders
            self.holistc_tf_data    = self.holistic_model.tf_data
            self.header_tf_data     = self.header_model.tf_data
            self.footer_tf_data     = self.footer_model.tf_data
            self.left_tf_data       = self.left_model.tf_data
            self.right_tf_data      = self.right_model.tf_data
            self.tf_labels          = tf.placeholder(tf.float32, shape = (None, self.num_labels))

            # softmax output for all 5 regions
            self.holistic_output = self.holistic_model.get_prediction_op()
            self.header_output   = self.header_model.get_prediction_op()
            self.footer_output   = self.footer_model.get_prediction_op()
            self.left_output     = self.left_model.get_prediction_op()
            self.right_output    = self.right_model.get_prediction_op()
            
            self.concat_output = tf.concat(values=[self.holistic_output, self.header_output, self.footer_output, self.left_output, self.right_output], axis=-1)

            # META CLASSIFIER
            # WEIGHTS
            META_NUM_HIDDEN_1, META_NUM_HIDDEN_2 = 256, 256
            EPSILON = 1e-3      # For batch normalization ops

            self.w1 = tf.Variable(name="w1", initial_value=tf.truncated_normal([num_labels * 5, META_NUM_HIDDEN_1], stddev=0.1))
            self.b1 = tf.Variable(name="b1", initial_value=tf.constant(1.0, shape = [META_NUM_HIDDEN_1]))

            self.w2 = tf.Variable(name="w2", initial_value=tf.truncated_normal([META_NUM_HIDDEN_1, META_NUM_HIDDEN_2], stddev=0.1))
            self.b2 = tf.Variable(name="b2", initial_value=tf.constant(1.0, shape = [META_NUM_HIDDEN_2]))

            self.w3 = tf.Variable(name="w3", initial_value=tf.truncated_normal([META_NUM_HIDDEN_2, self.num_labels], stddev=0.1))
            self.b3 = tf.Variable(name="b3", initial_value=tf.constant(1.0, shape = [self.num_labels]))

            # LAYERS
            # Layer 1
            self.meta1_fccd = tf.matmul(self.concat_output, self.w1) + self.b1
            self.meta1_relu = tf.nn.relu(self.meta1_fccd)
            self.meta1_mean, self.meta1_var = tf.nn.moments(self.meta1_relu, [0])
            self.meta1_bn   = tf.nn.batch_normalization(x=self.meta1_relu, mean=self.meta1_mean, variance=self.meta1_var, offset=None, scale=None, variance_epsilon=EPSILON)
            self.meta1_drop = tf.layers.dropout(inputs=self.meta1_bn, rate=0.5, training=self.phase)

            # Layer 2
            self.meta2_fccd = tf.matmul(self.meta1_drop, self.w2) + self.b2
            self.meta2_relu = tf.nn.relu(self.meta2_fccd)
            self.meta2_mean, self.meta2_var = tf.nn.moments(self.meta2_relu, [0])
            self.meta2_bn   = tf.nn.batch_normalization(x=self.meta2_relu, mean=self.meta2_mean, variance=self.meta2_var, offset=None, scale=None, variance_epsilon=EPSILON)

            # Layer 3
            self.logits = tf.matmul(self.meta2_bn, self.w3) + self.b3
            self.prediction_op = tf.nn.softmax(self.logits)
            self.argmax = tf.argmax(self.prediction_op, 1)

            # Training ops
            self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.tf_labels))

            # Optimizer + Global Step
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self.learning_rate = tf.train.exponential_decay(learning_rate=self.learning_rate, global_step=self.global_step, decay_steps=4000, decay_rate=0.95, staircase=False)
            
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            self.gradients = self.optimizer.compute_gradients(self.loss_op)
            self.capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in self.gradients if grad is not None]
            self.optimizer_op = self.optimizer.apply_gradients(self.capped_gradients, global_step=self.global_step)

            # For sklearn summary metrics to work
            self.label_argmax = tf.argmax(self.tf_labels, 1)


    def get_variables(self):
        return {'w1': self.w1, 'w2': self.w2, 'w3': self.w3, \
                'b1': self.b1, 'b2': self.b2, 'b3': self.b3  }


    def get_saver(self):
        return tf.train.Saver({'w1': self.w1, 'w2': self.w2, 'w3': self.w3, \
                               'b1': self.b1, 'b2': self.b2, 'b3': self.b3  })


'''
ONE SHOT PREDICTOR

Run file directly like this >> python3 vggnet16.py {checkpoint_meta_path} {img_path} 
        >> python3 metaclassifier.py ./restore/epoch_33/alexnet ./img.png
'''
if __name__ == '__main__':

    checkpoint_path = sys.argv[1]
    img_path        = sys.argv[2]

    # Process Image to 224 x 224
    img_str = tf.read_file(img_path)
    original = tf.image.decode_png(img_str, channels=3)
    original = tf.image.rgb_to_grayscale(original)

    # Resize to 796 x 612 then remove pixels from the sides to 780 x 600: 8 px from top and bottom sides, and 6 px from left and right sides
    # Intuition: meaningful document structures are found in the center of the image due to page margins, remove 2% from side enhances features in the final image
    img_resize1 = tf.image.resize_images(original, [780 + 16, 600 + 12], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    holistic = tf.slice(img_resize1, begin=[8, 6, 0], size=[780, 600, -1])

    # Slice regions
    header  = tf.slice(img_resize1, begin=[0, 0, 0], size=[256, -1, -1])
    footer  = tf.slice(img_resize1, begin=[524, 0, 0], size=[-1, -1, -1])
    left    = tf.slice(img_resize1, begin=[191, 0, 0], size=[400, 300, -1])
    right   = tf.slice(img_resize1, begin=[191, 300, 0], size=[400, 300, -1])

    # Resize to 224 x 224 px
    holistic = tf.image.resize_images(holistic, [224, 224], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    header = tf.image.resize_images(header, [224, 224], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    footer = tf.image.resize_images(footer, [224, 224], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    left = tf.image.resize_images(left, [224, 224], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    right = tf.image.resize_images(right, [224, 224], tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    sess = tf.Session()
    saver = tf.train.import_meta_graph(checkpoint_path + '.meta')
    saver.restore(sess, checkpoint_path)
    graph = tf.get_default_graph()       

    # VERSION 2 ============================================================================================
    model = MetaClassifier(scope_name="meta_classifier", image_width = 224, image_height = 224, image_depth = 1, num_labels = 17, learning_rate=0.05)
    variables  = model.get_variables()
    logits_op  = model.logits
    softmax_op = model.prediction_op
    argmax     = model.argmax

    # Placeholders
    holistc_tf_data    = model.holistc_tf_data
    header_tf_data     = model.header_tf_data
    footer_tf_data     = model.footer_tf_data
    left_tf_data       = model.left_tf_data
    right_tf_data      = model.right_tf_data
    tf_labels          = model.tf_labels

    session = tf.Session()
    with session as sess:

        # load meta model
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint_path)

        # Create feed dict with batch, Create ops to run, Run session
        ho, he, fo, le, ri = sess.run([holistic, header, footer, left, right])
        feed_dict = {holistc_tf_data: ho, header_tf_data: he, footer_tf_data: fo, left_tf_data: le, right_tf_data: ri}
        ops       = [logits_op, softmax_op, argmax]
        logits_out, softmax_out, argmax_out = session.run(ops, feed_dict=feed_dict)

        # Print output
        print("logits : {}".format(logits_out))
        print("softmax: {}".format(softmax_out))
        print("argmax : {}".format(argmax_out))


    

