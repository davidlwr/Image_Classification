import tensorflow as tf
import numpy as np
import os
import argparse
from VGG16 import VGG16
from VGG16_concat import VGG16_concat
from VGG16_selu import VGG16_selu
from datetime import datetime
import sys
sys.path.insert(0, '..')
from utils import utils
from utils import data_loader

version = 1
phase   = 1     # 0 = inference/test, 1 = training

# General params
train_csv        = "../data/overview_training(PNG).csv"      # CSV that holds paths to training images, format: [rel_img_path], [label]
test_csv         = "../data/overview_testing(PNG).csv"       # CSV that holds paths tp testing images, format:  [rel_img_path], [label]
checkpoint_path  = "../checkpoint"                            # Base path to checkpoint file saved every epoch.i.e. "./checkpoint/all_regions/epoch_1/alexnet"
tensorboard_path = "../tensorboard"
display_step          = 1

# Learning params
learning_rate = 0.05    # 0.05 for alexnet with batch normalization, 0.001 for normal alexnet recommended by Abhishek
batch_size    = 10      # Previously 100, batch normalization should work better with higher batch_sizes?
epoch         = 10              
learning_rate_type = "EXPONENTIAL_DECAY"  # See vggnet16.py training_step() | "EXPONENTIAL_DECAY" | "STATIC"
optimizer_type     = "SGD"     # See vggnet16.py training_step()
dropout_rate  = 0.2

#General
image_width = 224
image_height = 224
image_depth = 1
num_labels = 17

# Arg Parse stuff
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--restorepath", "-rp", default="",  help="Path to the checkpoint file to restore model from. Note the path must contain the words 'imagenet' if you are restoring imagenet weights")
arg_parser.add_argument("--save_meta_graph", "-smg", default="False", help="True: meta graph is saved along with variables. False: Only variables are saved")
arg_parser.add_argument("--saved_model", "-s", default="", help="Path to create tf.saved_model in, exits after model is created. Does not save if left blank")
args                    = arg_parser.parse_args()
restore_path            = args.restorepath
save_meta_graph         = True if args.save_meta_graph == "True" else False
saved_model_working_dir  = args.saved_model

# Label classes
classes = {0:"memo", 1:"form", 2:"email", 3:"handwritten", 4:"advertisement",\
           5:"scientific report", 6:"scientific publication", 7:"specification", \
           8: "file folder", 9:"news article", 10:"budget", 11:"invoice", \
           12:"presentation", 13:"questionnaire", 14:"resume", 15:"memo", 16:"Tobacco800"}
class_l = list(classes.values())

# Graph
graph = tf.Graph()
with graph.as_default():

    # Normal
    model = VGG16(scope_name="vgg_16", image_width = 224, image_height = 224, image_depth = 3, num_labels = 17, bn_option=2, phase=phase)
    # model = VGG16_concat(scope_name="holistic", image_width = 224, image_height = 224, image_depth = 3, num_labels = 17, bn_option=2, phase=phase)
    # model = VGG16_selu(scope_name="holistic", image_width = 224, image_height = 224, image_depth = 3, num_labels = 17, phase=phase)
    variables                          = model.get_variables()
    logits_op                          = model.logits
    softmax_op                         = model.softmax
    argmax_op                          = model.argmax
    # loss_op, optimizer_op, global_step, label_argmax = model.training_step(learning_rate=learning_rate, learning_rate_type=learning_rate_type, optimizer_type=optimizer_type)
    loss_op, optimizer_op, global_step, label_argmax = model.training_step(learning_rate=0.01, learning_rate_type="STATIC", optimizer_type="RMSPROP")

    # Placeholders
    tf_data       = model.tf_data
    tf_labels     = model.tf_labels
    dropout_prob  = model.dropout_prob

    # Classes
    label_classes = tf.convert_to_tensor(value=class_l)


session = tf.Session(graph=graph)
with session as sess:
    session.run(tf.global_variables_initializer())
    session.run(tf.local_variables_initializer())

    saver = tf.train.Saver(variables, max_to_keep=epoch)

    # Load checkpoint?
    if len(restore_path) > 0:
        print("restoring model from checkpoint...")
        if 'imagenet' in restore_path:
            restorer = tf.train.Saver(var_list=model.get_imgnet_pretrained_var_map())
        else:
            restorer = tf.train.Saver()

        restorer.restore(session, restore_path)
        print("restore done")

    # Create saved model for Tensorflow serving?
    if len(saved_model_working_dir) > 0:
        print("saving tf.saved_model...")
        utils.create_saved_model(session=sess, work_dir=saved_model_working_dir, version=version,     \
                                 prediction_input_tensor=tf_data, prediction_output_tensor=argmax_op, \
                                 classify_input_tensor=tf_data, classify_output_scores=softmax_op, classify_output_classes=label_classes)
        print("saving done. Exiting now...")
        sys.exit(0)

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
    train_iterator = train_dataset.make_initializable_iterator()
    test_iterator = test_dataset.make_initializable_iterator()

    next_train_batch = train_iterator.get_next()
    next_test_batch = test_iterator.get_next()

    session.run(train_iterator.initializer)   
    session.run(test_iterator.initializer)

    # Run training
    log_postfix = 'holistic_bn_relu'
    utils.log('Training step start. Initialized with learning_rate: {}, Type: {}, Optimizer: {}'.format(str(learning_rate), learning_rate_type, optimizer_type), postfix=log_postfix)
    for e in range(epoch):
        utils.log("==================================== EPOCH {} START ========================================================================".format(str(e)), postfix=log_postfix)
        
        # Training steps
        for i in range(int(train_total_count / batch_size)):

            # Get next training batch
            holistic_batch, header_batch, footer_batch, left_batch, right_batch, label_batch = session.run(next_train_batch)
            feed_dict = {tf_data: holistic_batch, tf_labels: label_batch, dropout_prob:dropout_rate}

            # Run optimizer, update, metric ops
            ops = [optimizer_op, argmax_op, global_step, loss_op, label_argmax]
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

            # Run Prediction Ops
            feed_dict = {tf_data: holistic_batch, tf_labels: label_batch, dropout_prob:dropout_rate}
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
        save_path = checkpoint_path + "/epoch_" + str(e) + "/VGG.ckpt"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        saver.save(session, save_path, write_meta_graph=save_meta_graph)

