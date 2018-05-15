# coding: utf-8


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import numpy as np
import tensorflow as tf
import time
import sys
import pickle
import argparse
import copy
import morfessor
import my_dropout
import collections


class PTBSmallConfig(object):
  # Global hyperparameters
  batch_size = 20
  max_grad_norm = 5
  lr_decay = 0.90
  learning_rate = 0.7
  init_scale = 0.1
  num_epochs = 70
  max_epoch = 10
  word_vocab_size = 0 # to be determined later

  # LSTM hyperparameters
  num_steps = 35
  hidden_size = 200
  num_layers = 2
  drop_x = 0.0
  drop_i = 0.0
  drop_h = 0.3
  drop_o = 0.3

  # Morpheme embedding hyperparameters
  morph_vocab_size = 0 # to be determined later
  morph_emb_dim = 200
  highway_size = 200
  max_word_len = 0   # to be determined later
    
  # Sampled softmax (SSM) hyperparameters
  ssm = 0            # do not use SSM by default
  num_sampled = 0    # to be determined later
  
  #Reusing options
  reuse_emb = True   # reuse embedding layer
  reuse_hw1 = True   # reuse first highway
  reuse_hw2 = True   # reuse second highway


class PTBMediumConfig(object):
  # Global hyperparameters
  batch_size = 100 #20
  max_grad_norm = 5
  lr_decay = 0.90
  learning_rate = 0.7
  init_scale = 0.05
  num_epochs = 70
  max_epoch = 10
  word_vocab_size = 0 # to be determined later

  # LSTM hyperparameters
  num_steps = 35
  hidden_size = 650
  num_layers = 2
  drop_x = 0.0
  drop_i = 0.0
  drop_h = 0.5
  drop_o = 0.5

  # Morpheme embedding hyperparameters
  morph_vocab_size = 0 # to be determined later
  morph_emb_dim = 650
  highway_size = 650
  max_word_len = 0   # to be determined later
    
  # Sampled softmax (SSM) hyperparameters
  ssm = 0            # do not use SSM by default
  num_sampled = 0    # to be determined later
  
  #Reusing options
  reuse_emb = True   # reuse embedding layer
  reuse_hw1 = True   # reuse first highway
  reuse_hw2 = True   # reuse second highway


class WT2SmallConfig(object):
  # Global hyperparameters
  batch_size = 20
  max_grad_norm = 5
  lr_decay = 0.90
  learning_rate = 0.7
  init_scale = 0.1
  num_epochs = 70
  max_epoch = 10
  word_vocab_size = 0 # to be determined later

  # LSTM hyperparameters
  num_steps = 35
  hidden_size = 200
  num_layers = 2
  drop_x = 0.0
  drop_i = 0.0
  drop_h = 0.2
  drop_o = 0.2

  # Morpheme embedding hyperparameters
  morph_vocab_size = 0 # to be determined later
  morph_emb_dim = 200
  highway_size = 200
  max_word_len = 0   # to be determined later

  # Sampled softmax (SSM) hyperparameters
  ssm = 0            # do not use SSM by default
  num_sampled = 0    # to be determined later
  
  #Reusing options
  reuse_emb = True   # reuse embedding layer
  reuse_hw1 = True   # reuse first highway
  reuse_hw2 = True   # reuse second highway


class WT2MediumConfig(object):
  # Global hyperparameters
  batch_size = 20
  max_grad_norm = 5
  lr_decay = 0.90
  learning_rate = 0.7
  init_scale = 0.05
  num_epochs = 70
  max_epoch = 10
  word_vocab_size = 0 # to be determined later

  # LSTM hyperparameters
  num_steps = 35
  hidden_size = 650
  num_layers = 2
  drop_x = 0.0
  drop_i = 0.0
  drop_h = 0.4
  drop_o = 0.4

  # Morpheme embedding hyperparameters
  morph_vocab_size = 0 # to be determined later
  morph_emb_dim = 650
  highway_size = 650
  max_word_len = 0   # to be determined later

  # Sampled softmax (SSM) hyperparameters
  ssm = 0            # do not use SSM by default
  num_sampled = 0    # to be determined later
  
  #Reusing options
  reuse_emb = True   # reuse embedding layer
  reuse_hw1 = True   # reuse first highway
  reuse_hw2 = True   # reuse second highway


def parse_args():
  '''Parse command line arguments'''
  parser = argparse.ArgumentParser(formatter_class=
                                   argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--config', default='PTBSmall',
                      help='config. Possible options: PTBSmall, PTBMedium, ' \
                           'WT2Small, WT2Medium')
  parser.add_argument('--is_train', default='1', 
                      help='mode. 1 = training, 0 = evaluation')
  parser.add_argument('--data_dir', default='data/ptb', 
                      help='data directory. Should have train.txt/valid.txt' \
                           '/test.txt with input data')
  parser.add_argument('--morph', default='morph/ptb.bin',
                      help='location of a morphological' \
                           ' segmentation model')
  parser.add_argument('--save_dir', default='saves',
                      help='saves directory')
  parser.add_argument('--prefix', default='MorphSum',
                      help='prefix for filenames when saving data and model')
  parser.add_argument('--eos', default='<eos>',
                      help='EOS marker')
  parser.add_argument('--ssm', default='0',
                      help='sampled softmax. 1 = yes, 0 = no')
  parser.add_argument('--verbose', default='1',
                      help='print intermediate results. 1 = yes, 0 = no')
  parser.add_argument('--remb', default='1',
                      help='reuse embedding layer. 1 = yes, 0 = no')
  parser.add_argument('--rhw1', default='1',
                      help='reuse first highway layer. 1 = yes, 0 = no')
  parser.add_argument('--rhw2', default='1',
                      help='reuse second highway layer. 1 = yes, 0 = no')
  parser.add_argument('--wlp', default='100',
                      help='word length percentile.')
  parser.add_argument('--batch', default='20',
                      help='batch size')
  parser.add_argument('--gpu', default='0',
                      help='GPU ID to use.')
  return parser.parse_args()


def read_data(args, config):
  '''read data sets, construct all needed structures and update the config'''
  config.batch_size = int(args.batch)
  if args.ssm == '1': config.ssm = 1
  
  morf_io = morfessor.MorfessorIO()
  morf_model = morf_io.read_binary_model_file(args.morph)

  def my_morphemes(word):
    morphemes, _ = morf_model.viterbi_segment(word)
    return morphemes

  if args.is_train == '1':
    if not os.path.exists(args.save_dir):
      os.makedirs(args.save_dir)
    with open(os.path.join(
        args.save_dir, args.prefix + '-data.pkl'), 'wb') as data_file:
      word_data = open(os.path.join(args.data_dir, 'train.txt'), 'r').read() \
                  .replace('\n', args.eos).split()
      #words = list(set(word_data))
      counter = collections.Counter(word_data)
      count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
      words, _ = list(zip(*count_pairs))
        
      morphemes = set()
      word_lens_in_morph = []

      for word in words:
        morphs = my_morphemes(word)
        word_lens_in_morph.append(len(morphs))
        for morph in morphs:
          morphemes.add(morph)

      morphs_list = list(morphemes)
      pickle.dump(
          (word_data, words, word_lens_in_morph, morphs_list), data_file)

  else:
    with open(os.path.join(
        args.save_dir, args.prefix + '-data.pkl'), 'rb') as data_file:
      word_data, words, word_lens_in_morph, morphs_list = \
          pickle.load(data_file)

  word_data_size, word_vocab_size = len(word_data), len(words)
  print('data has %d words, %d unique' % (word_data_size, word_vocab_size))
  config.word_vocab_size = word_vocab_size
  config.num_sampled = int(word_vocab_size * 0.2)

  word_to_ix = { word:i for i,word in enumerate(words) }
  ix_to_word = { i:word for i,word in enumerate(words) }

  def get_word_raw_data(input_file):
    data = open(input_file, 'r').read().replace('\n', args.eos).split()
    return [word_to_ix[w] for w in data]

  train_raw_data = get_word_raw_data(os.path.join(args.data_dir, 'train.txt'))
  valid_raw_data = get_word_raw_data(os.path.join(args.data_dir, 'valid.txt'))
  test_raw_data = get_word_raw_data(os.path.join(args.data_dir, 'test.txt'))

  morph_vocab_size = len(morphs_list)
  max_word_len = int(np.percentile(word_lens_in_morph, args.wlp))
  config.max_word_len = max_word_len
  print('data has %d unique morphs' % morph_vocab_size)
  print('max word length in morphs is set to', max_word_len)

  # a fake morpheme for zero-padding
  zero_pad_morph = ' '
  morphs_list.insert(0, zero_pad_morph)
  morph_vocab_size += 1
  config.morph_vocab_size = morph_vocab_size

  morph_to_ix = { morph:i for i,morph in enumerate(morphs_list) }
  ix_to_morph = { i:morph for i,morph in enumerate(morphs_list) }

  #word_ix_to_morph_ixs = {}
  word_ix_to_morph_ixs = np.empty([word_vocab_size, max_word_len], 
                                  dtype=np.int32)
  for word in words:
    word_ix = word_to_ix[word]
    word_in_morphs = my_morphemes(word)
    if len(word_in_morphs) > max_word_len:
      del word_in_morphs[max_word_len:]
    else:
      word_in_morphs += [zero_pad_morph] * (max_word_len - len(word_in_morphs))
    word_ix_to_morph_ixs[word_ix] = \
        [morph_to_ix[morph] for morph in word_in_morphs]

  return train_raw_data, valid_raw_data, test_raw_data, word_ix_to_morph_ixs


class batch_producer(object):
  '''Slice the raw data into batches'''
  def __init__(self, raw_data, batch_size, num_steps):
    self.raw_data = raw_data
    self.batch_size = batch_size
    self.num_steps = num_steps
    
    self.batch_len = len(self.raw_data) // self.batch_size
    self.data = np.reshape(self.raw_data[0 : self.batch_size * self.batch_len],
                           (self.batch_size, self.batch_len))
    
    self.epoch_size = (self.batch_len - 1) // self.num_steps
    self.i = 0
  
  def __next__(self):
    if self.i < self.epoch_size:
      # batch_x and batch_y are of shape [batch_size, num_steps]
      batch_x = self.data[::, 
          self.i * self.num_steps : (self.i + 1) * self.num_steps : ]
      batch_y = self.data[::, 
          self.i * self.num_steps + 1 : (self.i + 1) * self.num_steps + 1 : ]
      self.i += 1
      return (batch_x, batch_y)
    else:
      raise StopIteration()

  def __iter__(self):
    return self


class Model:
  '''Morpheme-aware language model'''
  def __init__(self, config, word_ix_to_morph_ixs, need_reuse=False):
    # get hyperparameters
    batch_size = config.batch_size
    num_steps = config.num_steps
    self.max_word_len = max_word_len = config.max_word_len
    self.morph_emb_dim = morph_emb_dim = config.morph_emb_dim
    self.highway_size = highway_size = config.highway_size
    self.init_scale = init_scale = config.init_scale
    num_sampled = config.num_sampled
    morph_vocab_size = config.morph_vocab_size
    hidden_size = config.hidden_size
    num_layers = config.num_layers
    word_vocab_size = config.word_vocab_size
    drop_x = config.drop_x
    drop_i = config.drop_i
    drop_h = config.drop_h
    drop_o = config.drop_o

    # morpheme embedding matrix
    with tf.variable_scope('morph_emb', reuse=need_reuse):
      self.morph_embedding = tf.get_variable("morph_embedding", 
        [morph_vocab_size, morph_emb_dim], dtype=tf.float32)
    
    # placeholders for training data and labels
    self.x = tf.placeholder(tf.int32, [batch_size, num_steps, max_word_len])
    self.y = tf.placeholder(tf.int32, [batch_size, num_steps])
    y_float = tf.cast(self.y, tf.float32)
    
    # we first embed morphemes ...
    words_embedded = tf.nn.embedding_lookup(self.morph_embedding, self.x)
    words_emb_as_list = tf.unstack(words_embedded, axis=1)
    
    words_list = []
    for word_emb in words_emb_as_list:
      #summing up morpheme embeddings
      morph_sum = tf.reduce_sum(word_emb, axis=1)
      words_list.append(morph_sum)
      
    words_packed_reshaped = tf.reshape(tf.stack(words_list, axis=1), 
                                       [-1, morph_emb_dim])
    
    # we project word vectors to match the dimensionality of 
    # the highway layer
    if morph_emb_dim != highway_size:
      with tf.variable_scope('projection', reuse=need_reuse):
        proj_w = tf.get_variable('proj_w', 
          [morph_emb_dim, highway_size], dtype=tf.float32)
      words_packed_reshaped_proj = tf.matmul(words_packed_reshaped, proj_w)
    else:
      words_packed_reshaped_proj = words_packed_reshaped
    
    # we feed the word vector into a stack of two HW layers ...
    with tf.variable_scope('highway1', reuse=need_reuse):
      highw1_output = self.highway_layer(words_packed_reshaped_proj)
    
    with tf.variable_scope('highway2', reuse=need_reuse):
      highw2_output = self.highway_layer(highw1_output)
        
    highw_output_reshaped = tf.reshape(highw2_output, 
                                       [batch_size, num_steps, -1])
    if not need_reuse:
      highw_output_reshaped = tf.nn.dropout(
          highw_output_reshaped, 1-drop_x, [batch_size, num_steps, 1])
    
    # ... and then process it with a stack of two LSTMs
    lstm_input = tf.unstack(highw_output_reshaped, axis=1)
    # basic LSTM cell
    def lstm_cell():
      return tf.nn.rnn_cell.LSTMCell(hidden_size, 
                                                   forget_bias=1.0,
                                                   reuse=need_reuse)
    cells = []
    for i in range(num_layers):
      with tf.variable_scope('layer' + str(i)):
        if not need_reuse:
          if i == 0:
            cells.append(
                my_dropout.MyDropoutWrapper(lstm_cell(), 
                                            input_keep_prob=1-drop_i,
                                            state_keep_prob=1-drop_h,
                                            output_keep_prob=1-drop_o,
                                            variational_recurrent=True,
                                            input_size=highway_size,
                                            dtype=tf.float32))
          else:
            cells.append(
                my_dropout.MyDropoutWrapper(lstm_cell(),
                                            state_keep_prob=1-drop_h,
                                            output_keep_prob=1-drop_o,
                                            variational_recurrent=True,
                                            input_size=hidden_size,
                                            dtype=tf.float32))
        else:
          cells.append(lstm_cell())
    self.cell = tf.nn.rnn_cell.MultiRNNCell(cells)
    
    self.init_state = self.cell.zero_state(batch_size, dtype=tf.float32)
    with tf.variable_scope('lstm_rnn', reuse=need_reuse):
      outputs, self.state = tf.contrib.rnn.static_rnn(
          self.cell, 
          lstm_input, 
          dtype=tf.float32, 
          initial_state=self.init_state)
    output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, hidden_size])
      
    # finally we predict the next word according to a softmax normalization
    if config.reuse_emb:
      self.morph_embedding_out = self.morph_embedding
    with tf.variable_scope('softmax_params', reuse=need_reuse):
      if morph_emb_dim != highway_size:
        proj_w_out = tf.get_variable('proj_w_out', 
          [morph_emb_dim, highway_size], dtype=tf.float32)
      if highway_size != hidden_size:
        proj2_w_out = tf.get_variable('proj2_w_out', 
          [highway_size, hidden_size], dtype=tf.float32)
      if not config.reuse_emb:
        self.morph_embedding_out = tf.get_variable("morph_embedding_out", 
          [morph_vocab_size, morph_emb_dim], dtype=tf.float32)
      biases = tf.get_variable('biases', [word_vocab_size], dtype=tf.float32)
    
    if config.ssm == 1 and not need_reuse:
      def _sum_rows(x):
        """Returns a vector summing up each row of the matrix x."""
        # _sum_rows(x) is equivalent to math_ops.reduce_sum(x, 1) when x is
        # a matrix.  The gradient of _sum_rows(x) is more efficient than
        # reduce_sum(x, 1)'s gradient in today's implementation. Therefore,
        # we use _sum_rows(x) in the nce_loss() computation since the loss
        # is mostly used for training.
        cols = tf.shape(x)[1]
        ones_shape = tf.stack([cols, 1])
        ones = tf.ones(ones_shape, x.dtype)
        return tf.reshape(tf.matmul(x, ones), [-1])
    
      labels = tf.cast(tf.reshape(self.y, [-1, 1]), tf.int64)
      labels_flat = tf.reshape(labels, [-1])

      # Sample the negative labels.
      #   sampled shape: [num_sampled] tensor
      #   true_expected_count shape = [batch_size, 1] tensor
      #   sampled_expected_count shape = [num_sampled] tensor
      sampled_values = tf.nn.log_uniform_candidate_sampler(
          true_classes=labels,
          num_true=1,
          num_sampled=num_sampled,
          unique=True,
          range_max=word_vocab_size)
      sampled, true_expected_count, sampled_expected_count = (
          tf.stop_gradient(s) for s in sampled_values)
      sampled = tf.cast(sampled, tf.int64)
    
      # labels_flat is a [batch_size * num_steps] tensor
      # sampled is a [num_sampled] int tensor
      all_ids = tf.concat([labels_flat, sampled], 0)
    
      words_in_morphs = tf.nn.embedding_lookup(
          word_ix_to_morph_ixs, all_ids)
      words_in_morphs_embedded = tf.nn.embedding_lookup(
          self.morph_embedding_out, words_in_morphs)
      
      all_w_full = tf.reduce_sum(words_in_morphs_embedded, axis=1)
      if morph_emb_dim != highway_size:
        all_w_proj = tf.matmul(all_w_full, proj_w_out)
      else:
        all_w_proj = all_w_full
      with tf.variable_scope('highway1' if config.reuse_hw1 else 'highway1_out', 
                             reuse=config.reuse_hw1 or need_reuse):
        all_w_hw1_output = self.highway_layer(all_w_proj)
      with tf.variable_scope('highway2' if config.reuse_hw2 else 'highway2_out', 
                             reuse=config.reuse_hw2 or need_reuse):
        all_w_hw2_output = self.highway_layer(all_w_hw1_output)
      if highway_size != hidden_size:
        all_w = tf.matmul(all_w_hw2_output, proj2_w_out)
      else:
        all_w = all_w_hw2_output
    
      # true_w shape is [batch_size * num_true, dim]
      true_w = tf.slice(all_w, [0, 0],
                        tf.stack([tf.shape(labels_flat)[0], -1]))
      sampled_w = tf.slice(
          all_w, tf.stack([tf.shape(labels_flat)[0], 0]), [-1, -1])
    
      sampled_logits = tf.matmul(output, sampled_w, transpose_b=True)
    
      all_b = tf.nn.embedding_lookup(
          biases, all_ids)
      # true_b is a [batch_size * num_true] tensor
      # sampled_b is a [num_sampled] float tensor
      true_b = tf.slice(all_b, [0], tf.shape(labels_flat))
      sampled_b = tf.slice(all_b, tf.shape(labels_flat), [-1])
    
      dim = tf.shape(true_w)[1:2]
      new_true_w_shape = tf.concat([[-1, 1], dim], 0)
      row_wise_dots = tf.multiply(
          tf.expand_dims(output, 1),
          tf.reshape(true_w, new_true_w_shape))
      # We want the row-wise dot plus biases which yields a
      # [batch_size, num_true] tensor of true_logits.
      dots_as_matrix = tf.reshape(row_wise_dots,
                                  tf.concat([[-1], dim], 0))
      true_logits = tf.reshape(_sum_rows(dots_as_matrix), [-1, 1])
      true_b = tf.reshape(true_b, [-1, 1])
      true_logits += true_b
      sampled_logits += sampled_b
    
      true_logits -= tf.log(true_expected_count)
      sampled_logits -= tf.log(sampled_expected_count)

      # Construct output logits and labels. The true labels/logits start at col 0.
      out_logits = tf.concat([true_logits, sampled_logits], 1)
      # true_logits is a float tensor, ones_like(true_logits) is a float tensor
      # of ones. We then divide by num_true to ensure the per-example labels sum
      # to 1.0, i.e. form a proper probability distribution.
      out_labels = tf.concat([
          tf.ones_like(true_logits),
          tf.zeros_like(sampled_logits)
      ], 1)
      loss = tf.nn.softmax_cross_entropy_with_logits(
          labels=out_labels, logits=out_logits)
    
    else:
      words_in_morphs_embedded = tf.nn.embedding_lookup(
          self.morph_embedding_out, word_ix_to_morph_ixs)
      weights_full = tf.reduce_sum(words_in_morphs_embedded, axis=1)
      if morph_emb_dim != highway_size:
        weights_proj = tf.matmul(weights_full, proj_w_out)
      else:
        weights_proj = weights_full
      with tf.variable_scope('highway1' if config.reuse_hw1 else 'highway1_out', 
                             reuse=config.reuse_hw1 or need_reuse):
        weights_highw1_output = self.highway_layer(weights_proj)
      with tf.variable_scope('highway2' if config.reuse_hw2 else 'highway2_out', 
                             reuse=config.reuse_hw2 or need_reuse):
        weights_highw2_output = self.highway_layer(weights_highw1_output)
      if highway_size != hidden_size:
        weights = tf.matmul(weights_highw2_output, proj2_w_out)
      else:
        weights = weights_highw2_output

      logits = tf.matmul(output, tf.transpose(weights)) + biases
      loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
              [logits],
              [tf.reshape(self.y, [-1])],
              [tf.ones([batch_size * num_steps], dtype=tf.float32)])
    
    self.cost = tf.reduce_sum(loss) / batch_size
    
  def highway_layer(self, highway_inputs):
    '''Highway layer'''
    transf_weights = tf.get_variable(
        'transf_weights', 
        [self.highway_size, self.highway_size],
        dtype=tf.float32)
    transf_biases = tf.get_variable(
        'transf_biases', 
        [self.highway_size],
        initializer=tf.random_uniform_initializer(-2-0.01, -2+0.01),
        dtype=tf.float32)
    highw_weights = tf.get_variable(
        'highw_weights', 
        [self.highway_size, self.highway_size],
        dtype=tf.float32)
    highw_biases = tf.get_variable(
        'highw_biases', 
        [self.highway_size],
        dtype=tf.float32)
    transf_gate = tf.nn.sigmoid(
        tf.matmul(highway_inputs, transf_weights) + transf_biases)
    highw_output = tf.multiply(
        transf_gate, 
        tf.nn.relu(
            tf.matmul(highway_inputs, highw_weights) + highw_biases)) \
        + tf.multiply(
        tf.ones([self.highway_size], dtype=tf.float32) - transf_gate, 
        highway_inputs)
    return highw_output
    

class Train(Model):
  '''for training we need to compute gradients'''
  def __init__(self, config, word_ix_to_morph_ixs):
    super(Train, self).__init__(config, word_ix_to_morph_ixs)
    self.clear_morph_embedding_padding = tf.scatter_update(
        self.morph_embedding, 
        [0], 
        tf.constant(0.0, shape=[1, config.morph_emb_dim], dtype=tf.float32))
    self.clear_morph_embedding_out_padding = tf.scatter_update(
        self.morph_embedding_out, 
        [0], 
        tf.constant(0.0, shape=[1, config.morph_emb_dim], dtype=tf.float32))

    self.lr = tf.Variable(0.0, trainable=False, dtype=tf.float32)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 
                                      config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self.lr)
    self.train_op = optimizer.apply_gradients(zip(grads, tvars),
      global_step=tf.train.get_or_create_global_step())
    
    self.new_lr = tf.placeholder(tf.float32, shape=[], 
                                 name="new_learning_rate")
    self.lr_update = tf.assign(self.lr, self.new_lr)

  # this will update the learning rate
  def assign_lr(self, session, lr_value):
    session.run(self.lr_update, feed_dict={self.new_lr: lr_value})


def model_size():
  '''finds the total number of trainable variables a.k.a. model size'''
  params = tf.trainable_variables()
  size = 0
  for x in params:
    sz = 1
    for dim in x.get_shape():
      sz *= dim.value
    size += sz
  return size


def run_epoch(sess, model, raw_data, config, is_train=False, lr=None):
  start_time = time.time()
  if is_train: model.assign_lr(sess, lr)

  iters = 0
  costs = 0
  state = sess.run(model.init_state)

  batches = batch_producer(raw_data, config.batch_size, config.num_steps)

  for batch in batches:
    my_x = np.empty(
        [config.batch_size, config.num_steps, config.max_word_len], 
        dtype=np.int32)

    # split words into morphemes
    for t in range(config.num_steps):
      for i in range(config.batch_size):
        my_x[i, t] = word_ix_to_morph_ixs[batch[0][i, t]]

    # run the model on current batch
    if is_train:
      _, c, state = sess.run(
          [model.train_op, model.cost, model.state],
          feed_dict={model.x: my_x, model.y: batch[1], 
          model.init_state: state})
      sess.run(model.clear_morph_embedding_padding)
      sess.run(model.clear_morph_embedding_out_padding)
    else:
      c, state = sess.run([model.cost, model.state], 
          feed_dict={model.x: my_x, model.y: batch[1], 
          model.init_state: state})

    costs += c
    step = iters // config.num_steps
    if is_train and args.verbose == '1' \
        and step % (batches.epoch_size // 10) == 10:
      print('%.3f' % (step * 1.0 / batches.epoch_size), end=' ')
      print('train ppl = %.3f' % np.exp(costs / iters), end=', ')
      print('speed =', 
          round(iters * config.batch_size / (time.time() - start_time)), 
          'wps')
    iters += config.num_steps
  
  return np.exp(costs / iters)


if __name__ == '__main__':
  args = parse_args()
  os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
  if args.config == 'PTBSmall': config = PTBSmallConfig()
  elif args.config == 'PTBMedium': config = PTBMediumConfig()
  elif args.config == 'WT2Small': config = WT2SmallConfig()
  elif args.config == 'WT2Medium': config = WT2MediumConfig()
  else: sys.exit('Invalid config.')
  initializer = tf.random_uniform_initializer(-config.init_scale,
                                              config.init_scale) 
  train_raw_data, valid_raw_data, test_raw_data, word_ix_to_morph_ixs \
      = read_data(args, config)
  
  config.reuse_emb = bool(int(args.remb))
  config.reuse_hw1 = bool(int(args.rhw1))
  config.reuse_hw2 = bool(int(args.rhw2))
  
  with tf.variable_scope('Model', reuse=False, initializer=initializer):
    train = Train(config, word_ix_to_morph_ixs)
  print('Model size is: ', model_size())

  with tf.variable_scope('Model', reuse=True, initializer=initializer):
    valid = Model(config, word_ix_to_morph_ixs, need_reuse=True)

  test_config = copy.deepcopy(config)
  #test_config.batch_size = 1
  test_config.ssm = 0
  #test_config.num_steps = 1
  with tf.variable_scope('Model', reuse=True, initializer=initializer):
    test = Model(test_config, word_ix_to_morph_ixs, need_reuse=True)

  saver = tf.train.Saver()

  
  if args.is_train == '1':
    num_epochs = config.num_epochs
    init = tf.global_variables_initializer()
    learning_rate = config.learning_rate

    with tf.Session() as sess:
      sess.run(init)
      
      saver.save(sess, os.path.join(
          args.save_dir, args.prefix + '-model.ckpt'))  
        
      sess.run(train.clear_morph_embedding_padding)
      sess.run(train.clear_morph_embedding_out_padding)
      prev_valid_ppl = float('inf')
      best_valid_ppl = float('inf')

      for epoch in range(num_epochs):
        train_ppl = run_epoch(
            sess, train, train_raw_data, config, is_train=True, 
            lr=learning_rate)
        print('epoch', epoch + 1, end = ': ')
        print('train ppl = %.3f' % train_ppl, end=', ')
        print('lr = %.3f' % learning_rate, end=', ')

        # Get validation set perplexity
        valid_ppl = run_epoch(
            sess, valid, valid_raw_data, config, is_train=False)
        print('valid ppl = %.3f' % valid_ppl)
        
        # Update the learning rate if necessary
        if epoch + 2 > config.max_epoch: learning_rate *= config.lr_decay
        
        # Save model if it gives better valid ppl
        if valid_ppl < best_valid_ppl:
          save_path = saver.save(sess, os.path.join(
              args.save_dir, args.prefix + '-model.ckpt'))
          print('Valid ppl improved. Model saved in file: %s' % save_path)
          best_valid_ppl = valid_ppl

  # Evaluation of a trained model on test set
  with tf.Session() as sess:
    # Restore variables from disk.
    saver.restore(
        sess, os.path.join(args.save_dir, args.prefix + '-model.ckpt'))
    print('Model restored.')

    # Get test set perplexity
    test_ppl = run_epoch(
        sess, test, test_raw_data, test_config, is_train=False)
    print('Test set perplexity = %.3f' % test_ppl)
