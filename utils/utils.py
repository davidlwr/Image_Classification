import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import random
import csv
import math
import sklearn

def summary_stats(argmax, labels, average='weighted', warn_for=("precision", "recall", "f-score")):
    # labels is set because a batch of predictions may not contain all classes, as such the calculation for averages includes a zero for the missing label. resulting in nan
    precision, recall, f1, _ = sklearn.metrics.precision_recall_fscore_support(y_true=labels, y_pred=argmax, average=average, warn_for=warn_for, labels=np.unique(argmax))
    
    accuracy = sklearn.metrics.accuracy_score(y_true=labels, y_pred=argmax)
    return accuracy, precision, recall, f1


def log(string, postfix=''):
    '''
    Writes and prints to std_out a string to a text file, used for logging purposes

    :param string: Text to write to file
    :param postfix: postifx of logfile to write to. i.e. ".logs/log_{postfix}"

    '''
    log_folder = "../logs"
    log_file = log_folder + "/" + str(postfix) + ".txt"

    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    with open(log_file, 'a+') as file:
        file.write(string + "\n")
        print(string)


def train_summary_tensorboard(save_folder, loss=None, accuracy=None, precision=None, recall=None, f1=None):

    loss_n      = "loss"
    accuracy_n  =  "accuracy"
    precision_n =  "precision"
    recall_n    =  "recall"
    f1_n        =  "f1"

    if not loss == None:
        tf.summary.scalar(loss_n, loss)

    if not accuracy == None:
        tf.summary.scalar(accuracy_n, accuracy)

    if not precision == None:
        tf.summary.scalar(precision_n, precision)

    if not recall == None:
        tf.summary.scalar(recall_n, recall)

    if not f1 == None:
        tf.summary.scalar(f1_n, f1)

    # Merge all summaries
    summary_merge_op = tf.summary.merge_all()

    # File writers for both train and test
    # Create folders if they dont exist
    train_path = save_folder + "/train"
    test_path  = save_folder + "/test"

    if not os.path.exists(train_path):
        os.makedirs(train_path)

    if not os.path.exists(test_path):
        os.makedirs(test_path)

    train_writer = tf.summary.FileWriter(save_folder + '/train', tf.get_default_graph())
    test_writer  = tf.summary.FileWriter(save_folder  + '/test')

    return train_writer, test_writer, summary_merge_op



def inspect_tensors_in_ckpt(ckpt_path):
    '''
    Returns and prints a sorted list where each entry contains the saved variabled name, its shape, and total tensors contained
    i.e. "meta_classifier/header/w1: [11, 11, 1, 96] => 11616"
    '''

    # Open TensorFlow ckpt
    reader = tf.train.NewCheckpointReader(ckpt_path)

    print('\nCount the number of parameters in ckpt file(%s)' % ckpt_path)
    param_map = reader.get_variable_to_shape_map()
    out = []
    total_count = 0
    for k, v in param_map.items():
        if 'Momentum' not in k and 'global_step' not in k:
            temp = np.prod(v)
            total_count += temp
            out.append('%s: %s => %d' % (k, str(v), temp))

    print('Total Param Count: %d' % total_count)
    for entry in sorted(out):
        print(entry)

    return sorted(out)



# See guide here: https://towardsdatascience.com/how-to-deploy-machine-learning-models-with-tensorflow-part-1-make-your-model-ready-for-serving-776a14ec3198
def create_saved_model(session, work_dir, version, prediction_input_tensor=None, prediction_output_tensor=None, classify_input_tensor=None, classify_output_scores=None, classify_output_classes=None):
    '''
    Creates a tf.saved_model in the the dir "./{work_dir}/{version}/" 
    Can either give params either and params for prediction and classify
    At least 1 must be given

    Keyword arguments
    session  -- tf.Session obj with graph
    work_dir -- working directory to save to
    version  -- version of model
    prediction_input_tensor  -- Input tenstor, batch or single image
    prediction_output_tensor -- Output of the model
    classify_input_tensor    -- Input tenstor, batch or single image
    classify_output_scores   -- Output softmax op
    classify_output_classes  -- Classes / labels tensor of strings

    '''

    # Tensorflow Saved Model params
    saved_model_export_path = os.path.join(tf.compat.as_bytes(work_dir),
                                           tf.compat.as_bytes(str(version)))
    builder = tf.saved_model.builder.SavedModelBuilder(saved_model_export_path)

    signature_def_map = {}
    default_set = False

    # Classification endpoint
    if classify_input_tensor != None and classify_output_scores != None and classify_output_classes != None:
        # Make protobufs
        classify_inputs_sig  = tf.saved_model.utils.build_tensor_info(classify_input_tensor)
        classify_classes_sig = tf.saved_model.utils.build_tensor_info(classify_output_classes)
        classify_scores_sig  = tf.saved_model.utils.build_tensor_info(classify_output_scores)

        # Create signatureDef
        classification_signature = (tf.saved_model.signature_def_utils.build_signature_def(inputs = {tf.saved_model.signature_constants.CLASSIFY_INPUTS: classify_inputs_sig},
                                                                                           outputs= {tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES: classify_classes_sig,
                                                                                                     tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES: classify_scores_sig},
                                                                                           method_name = {tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME}))

        # Add to signature def map
        key = ['classify_images' if default_set else tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        signature_def_map[key] = classification_signature
        default_set = True


    # Prediction end point
    if prediction_input_tensor != None and prediction_output_tensor != None:
        # Make protobufs
        prediction_inputs_sig = tf.saved_model.utils.build_tensor_info(prediction_input_tensor)
        prediction_output_sig = tf.saved_model.utils.build_tensor_info(prediction_output_tensor)

        # Create signatureDef
        prediction_signature  = (tf.saved_model.signature_def_utils.build_signature_def(inputs  = {'images': prediction_inputs_sig},
                                                                                        outputs = {'scores': prediction_output_sig},
                                                                                        method_name = tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

        # Add to signature def map
        key = ['predict_images' if default_set else tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        signature_def_map[key] = prediction_signature
        default_set = True


    # Add to builder
    builder.add_meta_graph_and_variables(sess=session,
                                         tags=[tf.saved_model.tag_constants.SERVING],
                                         signature_def_map=signature_def_map)
    # Im not too sure here, but I think the tag simply identifies this meta graph. When loading, you give the tag also to tell tf which meta graph to load

    # Save
    builder.save()      # Saved model creates folders that dont exist, dw



# TEST STUFF
# # Inspect checkpoint tensors directly
# path = ""
# inspect_tensors_in_ckpt(ckpt_path="")
# # Inspect checkpoint meta graph ops directly
# sess = tf.Session()
# saver = tf.train.import_meta_graph('../model_alexnet_chain/restore/epoch_33/alexnet.meta')
# saver.restore(sess, '../model_alexnet_chain/restore/epoch_33/alexnet')
# graph = tf.get_default_graph()       
# for op in graph.get_operations():
#     print(op.name)

# # Holistic
# logs_holistic = [("bn_adam",            "./logs/past_tests/alexnet_bn_adam.txt"),
#                  ("bn_sgd",             "./logs/past_tests/alexnet_bn_sgd.txt"),
#                  ("bn_after_relu_adam", "./logs/past_tests/alexnet_bnafrelu_adam.txt"),
#                  ("bn_after_relu_sgd",      "./logs/past_tests/alexnet_bnafrelu_sgd.txt"),]
# graph_logs(logs=logs_holistic, show_type="train_accuracy", graph_title="Alexnet Holistic")
# graph_logs(logs=logs_holistic, show_type="test_accuracy", graph_title="Alexnet Holistic")
# graph_logs(logs=logs_holistic, show_type="loss", graph_title="Alexnet Holistic")

# # Ensemble
# logs_ensemble = [("bn_after_relu_sgd", "./logs/past_tests/ensemble_bnafrelu_sgd.txt")]
# graph_logs(logs=logs_ensemble, show_type="train_accuracy", graph_title="Alexnet Ensemble")
# graph_logs(logs=logs_ensemble, show_type="test_accuracy", graph_title="Alexnet Ensemble")
# graph_logs(logs=logs_ensemble, show_type="loss", graph_title="Alexnet Ensemble")
