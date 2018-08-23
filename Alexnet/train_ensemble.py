import tensorflow as tf
import os
import argparse
from ensemble import Ensemble
from datetime import datetime
import sys
import numpy as np
sys.path.insert(0, '..')
from utils import utils
from utils import data_loader

# python3 train_metaclassifier.py -rp ../../model_tests/alexnet_holistic_bnafrelu_sgd/checkpoint/epoch_49/alexnet.ckpt

# General params
train_csv = "../data/overview_training(PNG).csv"                # CSV that holds paths to training images, format: [rel_img_path], [label]
test_csv  = "../data/overview_testing(PNG).csv"                 # CSV that holds paths tp testing images, format:  [rel_img_path], [label]
ckpt_save_path       = "../checkpoint/"               # Base path to checkpoint file saved every epoch.i.e. "./checkpoint/all_regions/epoch_1/alexnet"
display_step          = 1 

# Learning params
learning_rate = 0.01    # 0.05 for alexnet with batch normalization, 0.001 for normal alexnet recommended by Abhishek
batch_size    = 100     # Previously 100, batch normalization should work better with higher batch_sizes?
epoch         = 40

#General
image_width  = 224
image_height = 224
image_depth  = 1
num_labels   = 17

# Argparse stuff
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--restore", "-r", choices=["True", "False"], \
                        default="False", help="True: restore entire model (sub-models included), or False: restore all from pretrained holistic model")
arg_parser.add_argument("--restorepath", "-rp", help="Path to the checkpoint file to restore model from")
args                    = arg_parser.parse_args()
restore_entire_model    = args.restore
restore_path            = args.restorepath

# Graph
graph = tf.Graph()
with graph.as_default():
    model = Ensemble(scope_name="meta_classifier", \
                     image_width=image_width, image_height=image_height,  \
                     image_depth=image_depth, num_labels=num_labels,      \
                     learning_rate=learning_rate, phase=1)
    
    # Placeholders
    holistc_tf_data     = model.holistc_tf_data
    header_tf_data      = model.header_tf_data
    footer_tf_data      = model.footer_tf_data
    left_tf_data        = model.left_tf_data
    right_tf_data       = model.right_tf_data
    meta_tf_labels      = model.tf_labels

    # meta-classifier vars
    meta_var_dict = model.get_variables()

    # meta-classifier ops
    inference_op = model.logits
    argmax_op = model.argmax
    loss_op = model.loss_op
    optimizer_op = model.optimizer_op
    label_argmax_op = model.label_argmax

session = tf.Session(graph=graph)
with session as sess:

    # Restore only some vars
    names_to_vars = {v.op.name: v for v in tf.all_variables()}
    del names_to_vars["meta_classifier/global_step"]
    saver = tf.train.Saver(var_list=names_to_vars, max_to_keep=1)

    # Initialize all vars
    sess.run(tf.global_variables_initializer())

    if restore_entire_model == "False":
        print("Restoring sub-models from pre-trained holistic checkpoint...")
        model.holistic_model.get_saver().restore(sess, restore_path)
        model.header_model.get_saver().restore(sess, restore_path)
        model.footer_model.get_saver().restore(sess, restore_path)
        model.left_model.get_saver().restore(sess, restore_path)
        model.right_model.get_saver().restore(sess, restore_path)

    else:
        print("restoring entire model checkpoint")
        saver.restore(sess, restore_path)
    print("Sub-models restored")

    # Get datasets 
    train_total_count, train_dataset = data_loader.get_region_dataset_training(path_label_csv=train_csv, 
                                                                                batch_size=batch_size,
                                                                                num_labels=num_labels,
                                                                                img_width= image_width, 
                                                                                img_height=image_height,
                                                                                channels=image_depth)

    test_total_count, test_dataset = data_loader.get_region_dataset_training(path_label_csv=test_csv, 
                                                                            batch_size=batch_size,
                                                                            num_labels=num_labels,
                                                                            img_width= image_width, 
                                                                            img_height=image_height,
                                                                            channels=image_depth)

    # One shot iterators
    train_iterator   = train_dataset.make_initializable_iterator()
    test_iterator    = test_dataset.make_initializable_iterator()

    next_train_batch = train_iterator.get_next()
    next_test_batch  = test_iterator.get_next()

    session.run(train_iterator.initializer)   
    session.run(test_iterator.initializer)

    # Run training
    log_postfix = "all_regions"
    step_count = 0
    utils.log('Training step start. Initialized with learning_rate: ' + str(learning_rate), postfix='all_regions')
    for e in range(epoch):
        utils.log("{} ==================================== EPOCH {} START ====================================".format(datetime.now(), str(e)), postfix=log_postfix)

        # Iterate through training set
        for i in range(int(train_total_count / batch_size)):
            try:
                holistic_batch, header_batch, footer_batch, left_batch, right_batch, label_batch = session.run(next_train_batch)
    
                feed_dict = {holistc_tf_data: holistic_batch, header_tf_data: header_batch, \
                             footer_tf_data : footer_batch  , left_tf_data  : left_batch,   \
                             right_tf_data  : right_batch   , meta_tf_labels: label_batch   }
                _, l, argmax, label_argmax = session.run([optimizer_op, loss_op, argmax_op, label_argmax_op], feed_dict=feed_dict)
                
                step_count += batch_size
            
                if i % display_step == 0:
                     # Get Metrics
                    t_accuracy, t_precision, t_recall, t_f1 = utils.summary_stats(argmax=argmax, labels=label_argmax)
                    print("{}   SUMMARY at step {:04d} = | loss: {:06.4f} | accuracy {:02.4f} | precision: {:02.4f} | recall: {:02.4f} | f1: {:02.4f}" \
                            .format(datetime.now(), step_count, l, t_accuracy, t_precision, t_recall, t_f1), end='\r', flush=True)
            except tf.errors.OutOfRangeError:
                break   # Current epoch

        # Validate against testing set per epoch
        agg_labels  = []
        agg_argmax  = []
        agg_loss  = 0
        count     = 0
        for k in range(int(test_total_count / batch_size)):
            try:
                holistic_batch, header_batch, footer_batch, left_batch, right_batch, label_batch = session.run(next_test_batch)

                feed_dict = {holistc_tf_data: holistic_batch, header_tf_data: header_batch, \
                             footer_tf_data : footer_batch  , left_tf_data  : left_batch,   \
                             right_tf_data  : right_batch   , meta_tf_labels: label_batch   }
                v_argmax, v_label_argmax, v_loss = session.run([argmax_op, label_argmax_op, loss_op], feed_dict=feed_dict)

                agg_labels  = np.concatenate((agg_labels, v_label_argmax), axis=0)
                agg_argmax  = np.concatenate((agg_argmax, v_argmax), axis=0)
                count += 1
                agg_loss += v_loss
            except tf.errors.OutOfRangeError:
                break #end of test dataset

        # Display accuracy
        v_accuracy, v_precision, v_recall, v_f1 = utils.summary_stats(argmax=agg_argmax, labels=agg_labels)
        v_loss = agg_loss / count

        utils.log("{}   SUMMARY at step {:04d} = | loss: {:06.4f} | accuracy {:02.4f} | precision: {:02.4f} | recall: {:02.4f} | f1: {:02.4f}" \
                .format(datetime.now(), step_count, l, t_accuracy, t_precision, t_recall, t_f1), postfix=log_postfix)
        utils.log("{}   VALIDATION at step {:04d} = | loss: {:06.4f} | accuracy {:02.4f} | precision: {:02.4f} | recall: {:02.4f} | f1: {:02.4f}" \
                .format(datetime.now(), step_count, v_loss, v_accuracy, v_precision, v_recall, v_f1), postfix=log_postfix)
        utils.log("==================================== EPOCH {} END ======================================".format(str(e)), postfix=log_postfix)

        # save model
        save_path = ckpt_save_path + "/epoch_" + str(e) + "/alexnet.ckpt"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        saver.save(session, save_path, write_meta_graph=True)