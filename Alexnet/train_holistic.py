import tensorflow as tf
import os
import argparse
from alexnet_bn import Alexnet_bn
from datetime import datetime
import sys
sys.path.insert(0, '..')
from utils import utils
from utils import data_loader

# General params
train_csv       = "../data/overview_training(PNG).csv"      # CSV that holds paths to training images, format: [rel_img_path], [label]
test_csv        = "../data/overview_testing(PNG).csv"       # CSV that holds paths tp testing images, format:  [rel_img_path], [label]
checkpoint_path = ".../checkpoint"                            # Base path to checkpoint file saved every epoch.i.e. "./checkpoint/all_regions/epoch_1/alexnet"

display_step          = 1

# Learning params
learning_rate = 0.05    # 0.05 for alexnet with batch normalization, 0.001 for normal alexnet recommended by Abhishek
batch_size = 1          # Previously 100, batch normalization should work better with higher batch_sizes?
epoch = 50              

#General
image_width = 224
image_height = 224
image_depth = 1
num_labels = 17

# Argparse stuff
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--restore", "-r", choices=["True", "False"], \
                        default="False", help="True: restore from variable \"restore_path\", or False: train model from scratch")
arg_parser.add_argument("--restorepath", "-rp", help="Path to the checkpoint file to restore model from")
args = arg_parser.parse_args()
restore_entire_model = args.restore
restore_path         = args.restorepath

# Graph
# Note: as a self learning point, DO NOT DECLARE TENSORS, OPS OR ANY TF OBJECT OUTSIDE OF THE GRAPH. THIS CAUSES A MEMORY LEEK
graph = tf.Graph()
with graph.as_default():
    model = Alexnet_bn(scope_name="scope1", learning_rate=learning_rate, image_width = 224, image_height = 224, image_depth = 1, num_labels = 17)
    variables     = model.get_variables()
    inference_op  = model.get_inference_op()
    loss_op       = model.get_loss_op()
    optimizer_op  = model.get_optimizer_op()
    prediction_op = model.get_prediction_op()

    # Placeholders
    tf_data       = model.tf_data
    tf_labels     = model.tf_labels

    # Accuracy Ops
    correct_prediction = tf.equal(tf.argmax(prediction_op, 1), tf.argmax(tf_labels, 1))
    accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


session = tf.Session(graph=graph)
with session as sess:
    session.run(tf.global_variables_initializer())

    saver = tf.train.Saver(variables, max_to_keep=epoch)

    # Load checkpoint?
    if restore_entire_model == "True":
        print("restoring model from checkpoint...")
        saver.restore(session, restore_path)

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
    train_iterator = train_dataset.make_initializable_iterator()
    test_iterator = test_dataset.make_initializable_iterator()

    next_train_batch = train_iterator.get_next()
    next_test_batch = test_iterator.get_next()

    session.run(train_iterator.initializer)   
    session.run(test_iterator.initializer)


    # Run training
    log_postfix = 'holistic'
    step_count = 0
    utils.log('Training step start. Initialized with learning_rate: ' + str(learning_rate), log_postfix)
    for e in range(epoch):
        utils.log("==================================== EPOCH {} START ====================================".format(str(e)), postfix=log_postfix)
        utils.log("{}   Epoch: {}".format(datetime.now(), e), log_postfix)
        
        # Iterate through training set
        for i in range(int(train_total_count / batch_size)):
            try:
                holistic_batch, header_batch, footer_batch, left_batch, right_batch, label_batch = session.run(next_train_batch)

                feed_dict = {tf_data: holistic_batch, tf_labels: label_batch}
                _, l, train_accuracy = session.run([optimizer_op, loss_op, accuracy_op], feed_dict=feed_dict)

                step_count += batch_size

                if i % display_step == 0:
                    print("{}   SUMMARY at step {:04d} : loss_op is {:06.2f}   |   accuracy on training set {:02.2f} ".format(datetime.now(), step_count, l, train_accuracy), end='\r', flush=True)
            except tf.errors.OutOfRangeError:
                break   # Current epoch

        # Validate against testing set per epoch
        all_count    = 0
        all_accuracy = 0
        for k in range(int(test_total_count / batch_size)):
            try:
                holistic_batch, header_batch, footer_batch, left_batch, right_batch, label_batch = session.run(next_test_batch)

                feed_dict = {tf_data: holistic_batch, tf_labels: label_batch}
                batch_accuracy = session.run(accuracy_op, feed_dict=feed_dict)

                all_count += 1
                all_accuracy += batch_accuracy

            except tf.errors.OutOfRangeError:
                break #end of test dataset

        # Display accuracy
        test_accuracy = all_accuracy / all_count
        utils.log("{}   VALIDATION at step {:04d} : loss is {:06.2f}   |   accuracy on training set {:02.2f}   |   accuracy on test set {:02.2f}".format(datetime.now(), step_count, l, train_accuracy, test_accuracy), 'holistic')
        utils.log("==================================== EPOCH {} END ======================================".format(str(e)), postfix=log_postfix)

        # save model
        save_path = checkpoint_path + "/epoch_" + str(e) + "/alexnet.ckpt"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        saver.save(session, save_path, write_meta_graph=False)

