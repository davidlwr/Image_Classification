import tensorflow as tf
import numpy as np
import os
import argparse
from datetime import datetime
from vgg16_ensemble import VGG16_ensemble
import sys
sys.path.insert(0, '..')
from utils import utils
from utils import data_loader

# General params
train_csv             = "../data/overview_training(PNG).csv"      # CSV that holds paths to training images, format: [rel_img_path], [label]
test_csv              = "../data/overview_testing(PNG).csv"       # CSV that holds paths tp testing images, format:  [rel_img_path], [label]
checkpoint_path       = "../checkpoint"                            # Base path to checkpoint file saved every epoch.i.e. "./checkpoint/all_regions/epoch_1/alexnet"
tensorboard_path      = "../tensorboard"
display_step          = 1
log_postfix           = "all_regions"

# Layer params
bn_option = 0

# Learning params
learning_rate = 0.05    # 0.05 for alexnet with batch normalization, 0.001 for normal alexnet recommended by Abhishek
batch_size    = 10      # Previously 100, batch normalization should work better with higher batch_sizes?
epoch         = 50              
learning_rate_type = "STATIC"  # See vggnet16.py training_step()
optimizer_type     = "SGD"     # See vggnet16.py training_step()

#General
image_width = 224
image_height = 224
image_depth = 3
num_labels = 17


# Arg Parse stuff
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--restore", "-r", choices=["True", "False"], \
                        default="False", help="True: restore entire model (sub-models included), or False: restore all from pretrained holistic model")
arg_parser.add_argument("--restorepath", "-rp", help="Path to the checkpoint file to restore model from")
arg_parser.add_argument("--save_meta_graph", "-smg", default="False", help="True: meta graph is saved along with variables. False: Only variables are saved")
args                    = arg_parser.parse_args()
restore_entire_model    = args.restore
restore_path            = args.restorepath
save_meta_graph         = True if args.save_meta_graph == "True" else False

# Graph
graph = tf.Graph()
with graph.as_default():
    model = VGG16_ensemble(scope_name="ensemble", image_width=image_width, image_height=image_height, image_depth=image_depth, num_labels=num_labels, learning_rate=learning_rate, bn_option=bn_option, phase=1)
    
    # Placeholders
    holistc_tf_data    = model.holistc_tf_data
    header_tf_data     = model.header_tf_data
    footer_tf_data     = model.footer_tf_data
    left_tf_data       = model.left_tf_data
    right_tf_data      = model.right_tf_data
    tf_labels          = model.tf_labels

    # ensemble specific vars
    ensemble_var_dict = model.get_variables()

    # Ops
    logits_op    = model.logits
    softmax_op   = model.softmax
    argmax_op    = model.argmax
    loss_op      = model.loss_op
    train_op     = model.train_op
    global_step  = model.global_step
    label_argmax = model.label_argmax

    # loss_op, optimizer_op, global_step, label_argmax   = model.training_step(learning_rate=learning_rate, learning_rate_type=learning_rate_type, optimizer_type=optimizer_type)

session = tf.Session(graph=graph)
with session as sess:

    # Initialize all vars
    sess.run(tf.global_variables_initializer())

    # Restore models
    saver = tf.train.Saver(max_to_keep=1)

    if restore_entire_model == "False":
        print("Restoring sub-models from pre-trained holistic checkpoint...")
        tf.train.Saver(var_list=model.holistic_model.get_variables()).restore(sess, restore_path)
        tf.train.Saver(var_list=model.header_model.get_variables()).restore(sess, restore_path)
        tf.train.Saver(var_list=model.footer_model.get_variables()).restore(sess, restore_path)
        tf.train.Saver(var_list=model.left_model.get_variables()).restore(sess, restore_path)
        tf.train.Saver(var_list=model.right_model.get_variables()).restore(sess, restore_path)


        
    else:
        print("restoring entire model checkpoint")
        saver.restore(sess, restore_path)
    print("Sub-models restored")

    # Get datasets 
    train_total_count, train_dataset = data_loader.get_region_dataset_training(path_label_csv=train_csv, 
                                                                                batch_size=batch_size,
                                                                                num_labels=num_labels,
                                                                                img_width= image_width, 
                                                                                img_height=image_height)

    test_total_count, test_dataset = data_loader.get_region_dataset_training(path_label_csv=test_csv, 
                                                                            batch_size=batch_size,
                                                                            num_labels=num_labels,
                                                                            img_width= image_width, 
                                                                            img_height=image_height)

    # One shot iterators
    train_iterator   = train_dataset.make_initializable_iterator()
    test_iterator    = test_dataset.make_initializable_iterator()

    next_train_batch = train_iterator.get_next()
    next_test_batch  = test_iterator.get_next()

    session.run(train_iterator.initializer)   
    session.run(test_iterator.initializer)


    # Run training
    utils.log('Training step start. Initialized with learning_rate: {}, Type: {}, Optimizer: {}'.format(str(learning_rate), learning_rate_type, optimizer_type), postfix=log_postfix)
    for e in range(epoch):
        utils.log("==================================== EPOCH {} START ========================================================================".format(str(e)), postfix=log_postfix)

        # Iterate through training set
        for i in range(int(train_total_count / batch_size)):

            # Get next training batch
            holistic_batch, header_batch, footer_batch, left_batch, right_batch, label_batch = session.run(next_train_batch)

            # Create feed dict with batch, Create ops to run, Run session
            feed_dict = {holistc_tf_data: holistic_batch, header_tf_data: header_batch,    \
                            footer_tf_data : footer_batch  , left_tf_data  : left_batch,   \
                            right_tf_data  : right_batch   , tf_labels: label_batch        }

            # Run optimizer, update, metric ops
            ops = [train_op, argmax_op, global_step, loss_op, label_argmax]
            _, argmax, step, t_loss, t_labels = session.run(ops, feed_dict=feed_dict)
        
            if i % display_step == 0:
                # Get Metrics
                t_accuracy, t_precision, t_recall, t_f1 = utils.summary_stats(argmax=argmax, labels=t_labels)

                print("{}   TRAINING SUMMARY at step {:04d} = | loss: {:06.4f} | accuracy: {:02.4f} | precision: {:02.4f} | recall: {:02.4f} | f1: {:02.4f}" \
                            .format(datetime.now(), step, t_loss, t_accuracy, t_precision, t_recall, t_f1), end='\r', flush=True)


        # Validation steps
        agg_labels  = []
        agg_argmax  = []
        agg_loss  = 0
        count     = 0
        for k in range(int(test_total_count / batch_size)):
            # Get next test batch
            holistic_batch, header_batch, footer_batch, left_batch, right_batch, label_batch = session.run(next_test_batch)

            # Create feed dict with batch, Create ops to run, Run session
            feed_dict = {holistc_tf_data: holistic_batch, header_tf_data: header_batch,  \
                        footer_tf_data  : footer_batch  , left_tf_data  : left_batch,    \
                        right_tf_data   : right_batch   , tf_labels     : label_batch    }
            ops = [argmax_op, loss_op, label_argmax]
            v_argmax, v_loss, v_labels = session.run(ops, feed_dict=feed_dict)

            # Aggregate stats
            agg_labels  = np.concatenate((agg_labels, v_labels), axis=0)
            agg_argmax  = np.concatenate((agg_argmax, v_argmax), axis=0)
            agg_loss  += v_loss
            count     += 1


        # Calculate batch metrics
        v_accuracy, v_precision, v_recall, v_f1 = utils.summary_stats(argmax=agg_argmax, labels=agg_labels)
        v_loss = agg_loss / count

        # Display summary statistics
        utils.log("{}   TRAINING SUMMARY at step {:04d} = loss: {:06.4f} | accuracy: {:02.4f} | precision: {:02.4f} | recall: {:02.4f} | f1: {:02.4f}" \
                        .format(datetime.now(), step, t_loss, t_accuracy, t_precision, t_recall, t_f1), log_postfix)
        utils.log("{}   TEST SUMMARY     at step {:04d} = loss: {:06.4f} | accuracy: {:02.4f} | precision: {:02.4f} | recall: {:02.4f} | f1: {:02.4f}" \
                        .format(datetime.now(), step, v_loss, v_accuracy, v_precision, v_recall, v_f1), log_postfix)
        utils.log("==================================== EPOCH {} END ==========================================================================".format(str(e)), postfix=log_postfix)

        # save model
        save_path = checkpoint_path + "/epoch_" + str(e) + "/VGG_ensemble.ckpt"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        saver.save(session, save_path, write_meta_graph=save_meta_graph)