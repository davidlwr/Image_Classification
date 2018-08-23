import tensorflow as tf
from VGG16 import VGG16
import sys


class VGG16_ensemble(object):

    def __init__(self, scope_name, image_width = 224, image_height = 224, image_depth = 1, num_labels = 17, learning_rate = 0.05, bn_option = 0, phase=0):

        self.scope_name    = scope_name
        self.image_width   = image_width
        self.image_height  = image_height
        self.image_depth   = image_depth
        self.num_labels    = num_labels
        self.learning_rate = learning_rate
        self.bn_option     = bn_option      # 0=no bn, 1=bn before ReLU, 2=bn after ReLU
        self.phase         = phase          # 0 = test / inference, 1 = training

        with tf.variable_scope(self.scope_name):

            self.holistic_model = VGG16(scope_name="holistic", image_width = self.image_width, image_height = self.image_height, image_depth = self.image_depth, num_labels = self.num_labels, bn_option=self.bn_option, phase=self.phase)
            self.header_model   = VGG16(scope_name="header", image_width = self.image_width, image_height = self.image_height, image_depth = self.image_depth, num_labels = self.num_labels, bn_option=self.bn_option, phase=self.phase)
            self.footer_model   = VGG16(scope_name="footer", image_width = self.image_width, image_height = self.image_height, image_depth = self.image_depth, num_labels = self.num_labels, bn_option=self.bn_option, phase=self.phase)
            self.left_model     = VGG16(scope_name="left", image_width = self.image_width, image_height = self.image_height, image_depth = self.image_depth, num_labels = self.num_labels, bn_option=self.bn_option, phase=self.phase)
            self.right_model    = VGG16(scope_name="right", image_width = self.image_width, image_height = self.image_height, image_depth = self.image_depth, num_labels = self.num_labels, bn_option=self.bn_option, phase=self.phase)	

            # Placeholders
            self.holistc_tf_data    = self.holistic_model.tf_data
            self.header_tf_data     = self.header_model.tf_data
            self.footer_tf_data     = self.footer_model.tf_data
            self.left_tf_data       = self.left_model.tf_data
            self.right_tf_data      = self.right_model.tf_data

            # softmax output for all 5 regions
            self.holistic_output = self.holistic_model.softmax
            self.header_output   = self.header_model.softmax
            self.footer_output   = self.footer_model.softmax
            self.left_output     = self.left_model.softmax
            self.right_output    = self.right_model.softmax
            self.tf_labels       = tf.placeholder(tf.float32, shape = (None, self.num_labels))

            # concatenate all softmax layers [batch_size, num_labels * 5]
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

            # Layer 2
            self.meta2_fccd = tf.matmul(self.meta1_bn, self.w2) + self.b2
            self.meta2_relu = tf.nn.relu(self.meta2_fccd)
            self.meta2_mean, self.meta2_var = tf.nn.moments(self.meta2_relu, [0])
            self.meta2_bn   = tf.nn.batch_normalization(x=self.meta2_relu, mean=self.meta2_mean, variance=self.meta2_var, offset=None, scale=None, variance_epsilon=EPSILON)
            
            # Layer 3
            self.logits       = tf.matmul(self.meta2_bn, self.w3) + self.b3

            self.softmax      = tf.nn.softmax(self.logits)

            self.argmax       = tf.argmax(self.softmax, 1)

            # Training ops
            self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.tf_labels))

            # self.optimizer_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_op)
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

            # Gradient Clipping
            self.gradients = self.optimizer.compute_gradients(self.loss_op)
            self.capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in self.gradients if grad is not None]
            self.train_op = self.optimizer.apply_gradients(self.capped_gradients)

            # Label Argmax
            self.label_argmax = tf.argmax(self.tf_labels, 1)

            # Global Step
            self.global_step = tf.Variable(0, trainable=False, name='global_step')


    def get_variables(self):
        return {'w1': self.w1, 'w2': self.w2, 'w3': self.w3, \
                'b1': self.b1, 'b2': self.b2, 'b3': self.b3  }




'''
ONE SHOT PREDICTOR

Run file directly like this >> python3 vggnet16.py {checkpoint_meta_path} {img_path} 
     metaclassifier.py ./restore/epoch_33/alexnet ./img.png
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
    holistic = [tf.image.resize_images(holistic, [224, 224], tf.image.ResizeMethod.NEAREST_NEIGHBOR)]
    header = tf.image.resize_images(header, [224, 224], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    footer = tf.image.resize_images(footer, [224, 224], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    left = tf.image.resize_images(left, [224, 224], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    right = tf.image.resize_images(right, [224, 224], tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    sess = tf.Session()
    saver = tf.train.import_meta_graph(checkpoint_path + '.meta')
    saver.restore(sess, checkpoint_path)
    graph = tf.get_default_graph()       

    # OP NAME
    ph_holistic = graph.get_operation_by_name("ensemble/holistic/Placeholder").outputs[0]
    ph_header   = graph.get_operation_by_name("ensemble/header/Placeholder").outputs[0]
    ph_footer   = graph.get_operation_by_name("ensemble/footer/Placeholder").outputs[0]
    ph_left     = graph.get_operation_by_name("ensemble/left/Placeholder").outputs[0]
    ph_right    = graph.get_operation_by_name("ensemble/right/Placeholder").outputs[0]
    softmax     = graph.get_operation_by_name("ensemble/Softmax").outputs[0]
    feed_dict = {ph_holistic: holistic, ph_header: header, ph_footer: footer, ph_left: left, ph_right: right}
    softmax_out = sess.run(softmax, feed_dict)

    print("Calculated softmax: ")
    print(softmax_out)