
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import pdb
import argparse
import numpy as np
import tensorflow as tf
#from tensorflow.python import debug as tf_debug

from data_utils import DataProcessor, BatchData, get_vocab
from my_flags import FLAGS
from my_model import MY_Model
from train_test_utils import batch_predict_with_a_model, batch_load_data
from sklearn.model_selection import ParameterSampler, ParameterGrid
from model_utils import decoding_with_lm, match_label_trans
from evaluation import EvalStatistics

seed = 42

np.random.seed(seed)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Rutine for sweeping through hyper-parameters setups for the original sidenet')
  parser.add_argument('gpu',help='gpu id',type=str,default="0")
  parser.add_argument('dataset',help='Dataset to use / mode of FLAGS setup',type=str,default="newsqa")
  parser.add_argument('file_suffix',help='Suffix for exp name',type=str,default="")
  args = parser.parse_args()
  
  FLAGS.data_mode = args.dataset
  FLAGS.gpu_id = args.gpu
  
  FLAGS.force_reading = False
  FLAGS.train_dir = os.path.abspath("./train_dir_" + args.dataset+"_subs")
  FLAGS.train_epoch = 20
  FLAGS.use_subsampled_dataset = True

  #dir ubuntu
  FLAGS.raw_data_dir = "/home/usuario/datasets"

  if args.dataset=='eus':
    FLAGS.max_audio_length = 680 # obtained from sequence lengths histogram
    FLAGS.max_freq_length = 201
  elif args.dataset=='quz':
    FLAGS.max_audio_length = 100 # TBD
    FLAGS.max_freq_length = 100

  # set sentence, doc length to maximum
  output = open("tunning_"+FLAGS.data_mode+"/"+FLAGS.data_mode + "_hp_grid_tuning_%s.txt" % args.file_suffix,'w')

  vocab_dict,inverted_vocab = get_vocab()
  train_data = DataProcessor(vocab_dict,inverted_vocab,data_type="train")
  val_batch = batch_load_data(DataProcessor(vocab_dict,inverted_vocab,data_type="val"))
  
  setup_by_id = {}
  results_by_id_wer = {}
  results_by_id_cer = {}
  setup_id = 0
  best_global_wer = 200
  best_global_cer = 200
  best_setup_id_wer = -1
  best_setup_id_cer = -1

  ## FLAGS.___ = ___ # set as constant so it doesn't clutter output
  #FLAGS.use_conv2d = True
  #FLAGS.use_dropout = False 
  #FLAGS.birnn = False
  #FLAGS.lookahead_conv = False
  #FLAGS.lookahead_width = 2
  #FLAGS.lstm_cell = "lstm"
  #FLAGS.bigram = False
  #FLAGS.do_sortagram = True
  #FLAGS.lang_model = # write binary pathname here

  parameter_grid = {
    "batch_size" : [16,32,64,128],
    "learning_rate" : [1e-4],
    "use_conv2d": [True],
    "conv_layers": [1,2,3],
    "rnn_layers": [1,2,3],
    "size":[100],
    "embedding_size":[204],
    "filter_height":[41],
    "filter_width":[11],
    "dropout": [1.0],
    'max_gradient_norm': [-1,0],
    'beam_size':[0.5,1],
    'alpha':[1.0],
    'beta':[1.0]
  }
  
  ## loop for hyperparams
  param_gen = ParameterGrid(parameter_grid)
  for setup in param_gen:
    setup_time = time.time()
    setup_by_id[setup_id] = setup
    
    FLAGS.batch_size = setup["batch_size"]
    FLAGS.learning_rate = setup["learning_rate"]
    FLAGS.use_conv2d = setup["use_conv2d"]
    FLAGS.conv_layers = setup["conv_layers"]
    FLAGS.rnn_layers = setup["rnn_layers"]
    FLAGS.filter_height = setup["filter_height"]
    FLAGS.filter_width = setup["filter_width"]
    FLAGS.size = setup["size"]
    FLAGS.dropout = setup["dropout"]
    FLAGS.max_gradient_norm = setup["max_gradient_norm"]
    FLAGS.beam_size = setup["beam_size"]
    FLAGS.alpha = setup["alpha"]
    FLAGS.beta = setup["beta"]
    max_fixed = FLAGS.max_audio_length // 2 if FLAGS.bigram else FLAGS.max_audio_length
    
    prev_drpt = FLAGS.use_dropout
    
    # calculate lower boundary of emb_size
    if FLAGS.use_conv2d:
      final_freq = (FLAGS.max_freq_length // 2**(FLAGS.conv_layers)) + 1
      emb_size = max(final_freq,setup["embedding_size"])
      final_output_channel = emb_size // final_freq
      FLAGS.embedding_size = final_output_channel * final_freq
      setup["embedding_size"] = FLAGS.embedding_size # just for printing

    print("Setup ",setup_id,": ",setup)
    output.write("Setup %d: %s\n" % (setup_id,str(setup)))

    best_wer = 200
    best_cer = 200
    best_ep = 0
    best_ep_wer = 0
    with tf.Graph().as_default() and tf.device('/gpu:'+FLAGS.gpu_id):
      config = tf.ConfigProto(allow_soft_placement = True)
      tf.set_random_seed(seed)
      with tf.Session(config = config) as sess:
        model = MY_Model(sess, len(vocab_dict))
        init_epoch = 1
        
        for epoch in range(init_epoch, FLAGS.train_epoch+1):
          ep_time = time.time() # to check duration
          shuffle = (epoch != init_epoch) if FLAGS.do_sortagrad  else True
          if shuffle:
            train_data.shuffle_fileindices()

          total_loss = 0
          # Start Batch Training
          step = 1
          while (step * FLAGS.batch_size) <= len(train_data.fileindices):
            # Get batch data as Numpy Arrays
            batch = train_data.get_batch(((step-1)*FLAGS.batch_size), (step * FLAGS.batch_size))
            dshape = np.array([batch.spect.shape[0],max_fixed],dtype=np.int64)
            sparse_labels = tf.SparseTensorValue(batch.label_indices,batch.label_values,dshape)

            # Run optimizer: optimize policy and reward estimator
            _,ctc_loss = sess.run([model.train_main_network,
                                  model.ctc_loss],
                                  feed_dict={model.spect_placeholder: batch.spect,
                                             model.label_placeholder: sparse_labels,
                                             model.length_placeholder: batch.lengths})
            FLAGS.training = False
            prev_use_dpt = FLAGS.use_dropout
            FLAGS.use_dropout = False

            total_loss += ctc_loss
            # Increase step
            if step%500==0:
              print ("\tStep: ",step)
            step += 1
          #END-WHILE-TRAINING
          total_loss /= step
          FLAGS.use_dropout = False
          # retrieve batch with updated logits in it
          val_batch = batch_predict_with_a_model(val_batch, "validation", model, session=sess)
          
          dshape = np.array([val_batch.spect.shape[0],max_fixed],dtype=np.int64)
          sparse_labels = tf.SparseTensorValue(val_batch.label_indices,val_batch.label_values,dshape)
          # Validation Accuracy and Prediction
          if FLAGS.lang_model == '<none>':
            batch_decoded = sess.run(model.decoding,feed_dict={model.logits_placeholder: val_batch.logits,
                                                               model.length_placeholder: val_batch.lengths})
          else:
            ## decoding with LM, slow
            batch_decoded = decoding_with_lm(batch_logits,batch.lengths,
                                vocab_dict,language_model,sess)

          val_stats = EvalStatistics(inverted_vocab)
          val_stats.calc_statistics(batch_decoded,val_batch.transcripts)

          ctc_loss_val = sess.run(model.ctc_loss_val,
                                  feed_dict={model.logits_placeholder: val_batch.logits,
                                             model.label_placeholder:  sparse_labels,
                                             model.length_placeholder: val_batch.lengths})
          val_wer = val_stats.global_word_error_rate
          val_cer = val_stats.global_letter_error_rate

          print("Epoch %2d || train ctc loss: %.4f || val ctc loss: %.4f || val wer %.3f || val cer %.3f || duration: %3.2f" % 
            (epoch,total_loss,ctc_loss_val,
              val_wer,val_cer,
              time.time()-ep_time))
          output.write("Epoch %2d || train ctc loss: %.4f || val ctc loss: %.4f || val wer %.3f || val cer %.3f || duration: %3.2f\n" % 
            (epoch,total_loss,ctc_loss_val,
              val_wer,val_cer,
              time.time()-ep_time))

          FLAGS.use_dropout = prev_drpt
          if val_wer < best_wer and val_wer!=0.0:
            best_wer = val_wer
            best_ep = epoch
          if val_cer < best_cer and val_cer!=0.0:
            best_cer = val_stats.global_letter_error_rate
            best_ep_cer = epoch
          #break # for time testing
        #END-FOR-EPOCH
        results_by_id[setup_id] = (best_wer,best_ep)
        results_by_id_cer[setup_id] = (best_cer,best_ep_cer)
        if best_wer < best_global_wer:
          best_global_wer = best_wer
          best_setup_id = setup_id
        if best_cer < best_global_cer:
          best_global_cer = best_cer
          best_setup_id_cer = setup_id
      # clear graph
      tf.reset_default_graph()
    #END-GRAPH
    
    print("Best WER result in this setup:",results_by_id[setup_id])
    print("Best CER result in this setup:",results_by_id_cer[setup_id])
    print("Duration: %.4fsec" % (time.time()-setup_time))
    output.write("Best wer result in this setup: %.6f,%d\n" % (best_wer,best_ep))
    output.write("Best cer result in this setup: %.6f,%d\n" % (best_cer,best_ep_cer))
    output.write("Duration: %.4fsec\n" % (time.time()-setup_time))
    setup_id += 1

    #break
  #END-FOR-PARAMS
  
  print("Best wer setup: ",setup_by_id[best_setup_id])
  print("  WER: %.4f | Epoch: %d" % results_by_id[best_setup_id])
  print("Best cer setup: ",setup_by_id_cer[best_setup_id_cer])
  print("  CER: %.4f | Epoch: %d" % results_by_id_cer[best_setup_id_cer])
  output.write("Best WER setup: " + str(setup_by_id[best_setup_id]) + "\n")
  output.write("  WER: %.4f | Epoch: %d\n" % results_by_id[best_setup_id])
  output.write("Best CER setup: " + str(setup_by_id[best_setup_id_cer]) + "\n")
  output.write("  CER: %.4f | Epoch: %d\n" % results_by_id_cer[best_setup_id_cer])
  output.close()
  
