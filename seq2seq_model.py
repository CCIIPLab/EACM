# -*-coding:UTF-8-*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
from six.moves import range    # pylint: disable=redefined-builtin
import tensorflow as tf

import data_utils
import seq2seq
import rnn_cell


class Seq2SeqModel(object):

    def __init__(self,
                 source_vocab_size,
                 target_vocab_size,
                 buckets,
                 hidden_size,
                 num_layers,
                 max_gradient_norm,
                 batch_size,
                 learning_rate,
                 learning_rate_decay_factor,
                 num_samples=-1,
                 embedding_size=200,
                 forward_only=False,
                 beam_search=False,
                 beam_size=10,
                 category=6,
                 use_emb=False,
                 use_autoEM=False,
                 use_imemory=False,
                 use_ememory=False,
                 emotion_size=200,
                 imemory_size=256,
                 dtype=tf.float32):


        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.buckets = buckets
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(
                float(learning_rate), trainable=False, dtype=dtype)
        self.learning_rate_decay_op = self.learning_rate.assign(
                self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)



        # If we use sampled softmax, we need an output projection.
        output_projection = None
        softmax_loss_function = None

        # Sampled softmax only makes sense if we sample less than vocabulary hidden_size.
        if num_samples > 0 and num_samples < self.target_vocab_size:
            w_t = tf.get_variable("proj_w", [self.target_vocab_size, hidden_size], dtype=dtype)
            w = tf.transpose(w_t)
            b = tf.get_variable("proj_b", [self.target_vocab_size], dtype=dtype)
            output_projection = (w, b)

            def sampled_loss(inputs, labels):
                labels = tf.reshape(labels, [-1, 1])
                # We need to compute the sampled_softmax_loss using 32bit floats to
                # avoid numerical instabilities.
                local_w_t = tf.cast(w_t, tf.float32)
                local_b = tf.cast(b, tf.float32)
                local_inputs = tf.cast(inputs, tf.float32)
                return tf.cast(
                        tf.nn.sampled_softmax_loss(local_w_t, local_b, local_inputs, labels,
                                                                        num_samples, self.target_vocab_size), dtype)
            softmax_loss_function = sampled_loss
        else:
            w_t = tf.get_variable("proj_w", [self.target_vocab_size, hidden_size], dtype=dtype)
            w = tf.transpose(w_t)
            b = tf.get_variable("proj_b", [self.target_vocab_size], dtype=dtype)
            output_projection = (w, b)

        # Create the internal multi-layer cell for our RNN.
        gru = tf.nn.rnn_cell.GRUCell(hidden_size)
        encoder_cell = gru
        if num_layers > 1:
            encoder_cell = rnn_cell.MultiRNNCell([gru] * num_layers)
        # Create the internal multi-layer cell for our RNN.
        decoder_cell = encoder_cell
        if use_imemory or use_emb:
            decoder_cell = rnn_cell.MEMGRUCell(hidden_size)
            if num_layers > 1:
                decoder_cell = rnn_cell.MEMMultiRNNCell([decoder_cell]+[gru] * (num_layers-1))

        # The seq2seq function: we use embedding for the input and attention.
        def seq2seq_f(encoder_inputs, decoder_inputs, encoder_emotions, decoder_emotions, do_decode, autoEM_logit):
            return seq2seq.embedding_attention_seq2seq(
                    encoder_inputs,
                    decoder_inputs,
                    encoder_emotions,
                    decoder_emotions,
                    autoEM_logit,
                    encoder_cell,
                    decoder_cell,
                    num_encoder_symbols=source_vocab_size,
                    num_decoder_symbols=target_vocab_size,
                    embedding_size=embedding_size,
                    hidden_size=hidden_size,
                    emotion_category=category,
                    emotion_size=emotion_size,
                    imemory_size=imemory_size,
                    use_emb=use_emb,
                    use_imemory=use_imemory,
                    use_ememory=use_ememory,
                    output_projection=output_projection,
                    initial_state_attention=True,
                    feed_previous=do_decode,
                    dtype=dtype,
                    beam_search=beam_search,
                    beam_size=beam_size)

        # Feeds for inputs.
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        for i in range(buckets[-1][0]):    # Last bucket is the biggest one.
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="encoder{0}".format(i)))
        for i in range(buckets[-1][1] + 1):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="decoder{0}".format(i)))
            self.target_weights.append(tf.placeholder(dtype, shape=[None], name="weight{0}".format(i)))

        # Our targets are decoder inputs shifted by one.
        targets = [self.decoder_inputs[i + 1]
                             for i in range(len(self.decoder_inputs) - 1)]
        
        self.decoder_emotions = tf.placeholder(tf.int32, shape=[None,2], name="decoder_emotion")
        self.encoder_emotions = tf.placeholder(tf.int32, shape=[None,2], name="encoder_emotion")

        if use_autoEM:
            senti_cell = tf.nn.rnn_cell.GRUCell(hidden_size)
            grammar_cell = tf.nn.rnn_cell.GRUCell(hidden_size)
            if num_layers > 1:
                senti_cell = rnn_cell.MultiRNNCell([senti_cell] * num_layers)
                grammar_cell = rnn_cell.MultiRNNCell([grammar_cell] * num_layers)
            self.autoEM_losses, self.pos_logits, self.res_logits, self.res_losses ,self.res_cross_entropy,self.rlabels,self.weight = seq2seq.classify_model_with_buckets(senti_cell,
                        grammar_cell, self.encoder_inputs, self.encoder_emotions, self.decoder_emotions, buckets,
                         hidden_size, embedding_size, category, source_vocab_size)

            self.pos_predics = []
            self.res_predics = []
            for each in self.pos_logits:
                self.pos_predics.append(tf.arg_max(each, 1))
            for each in self.res_logits:
                self.res_predics.append(tf.arg_max(each, 1))


        # Training outputs and losses.
        if forward_only:
            if beam_search:
                self.outputs, self.beam_results, self.beam_symbols, self.beam_parents = seq2seq.decode_model_with_buckets(
                    self.encoder_inputs, self.decoder_inputs, targets,
                    self.target_weights, self.encoder_emotions, self.decoder_emotions, buckets,
                    lambda w, x, y, z,m: seq2seq_f(w, x, y, z, True, m),
                    softmax_loss_function=softmax_loss_function)
            else:
                self.outputs, self.losses, self.ppxes= seq2seq.model_with_buckets(
                        self.encoder_inputs, self.decoder_inputs, targets,
                        self.target_weights, self.encoder_emotions, self.decoder_emotions, buckets,
                        lambda w, x, y, z, m: seq2seq_f(w, x, y, z, True, m),self.res_logits,
                        softmax_loss_function=softmax_loss_function, use_imemory=use_imemory, use_ememory=use_ememory)
        else:
            self.outputs, self.losses, self.ppxes = seq2seq.model_with_buckets(
                    self.encoder_inputs, self.decoder_inputs, targets,
                    self.target_weights, self.encoder_emotions, self.decoder_emotions, buckets,
                    lambda w, x, y, z, m: seq2seq_f(w, x, y, z, False, m),self.res_logits,
                    softmax_loss_function=softmax_loss_function, use_imemory=use_imemory, use_ememory=use_ememory)

        self.total_losses = (1 * np.array(self.autoEM_losses) + 1 * np.array(self.losses)).tolist()
        # self.total_losses = self.autoEM_losses

            # Gradients and SGD update operation for training the model.
        params = tf.trainable_variables()
        if not forward_only:
            self.gradient_norms = []
            self.updates = []
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            for b in range(len(buckets)):  # len(buckets) is 4 on this occasion
                gradients = tf.gradients(self.total_losses[b], params)
                clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
                self.gradient_norms.append(norm)
                self.updates.append(opt.apply_gradients(
                        zip(clipped_gradients, params), global_step=self.global_step))

        self.pretrain_var = []
        self.initial_var = []
        for i in tf.trainable_variables():
            if 'Emotion' not in i.name and 'emotion' not in i.name and 'memory' not in i.name and 'Memory' not in i.name and "classify" not in i.name and "Attention_0" not in i.name:
                self.pretrain_var.append(i)
        for i in tf.all_variables():
            if i not in self.pretrain_var:
                self.initial_var.append(i)
        self.pretrain_saver = tf.train.Saver(self.pretrain_var, write_version=tf.train.SaverDef.V2)
        self.saver = tf.train.Saver(tf.all_variables(), write_version=tf.train.SaverDef.V2, max_to_keep=400)

    def get_accuracy(self, input1, input2):
        total = 0
        correct = 0
        for index, each in enumerate(input1):
            total += 1
            if each == input2[index]:
                correct += 1
        accuracy = float(correct / total)
        return accuracy

    def step(self, session, encoder_inputs, decoder_inputs, target_weights, encoder_emotions, decoder_emotions,
                     bucket_id, forward_only, beam_search, use_autoEM):


        # Check if the sizes match.
        encoder_size, decoder_size = self.buckets[bucket_id]
        if len(encoder_inputs) != encoder_size:
            raise ValueError("Encoder length must be equal to the one in bucket,"
                                             " %d != %d." % (len(encoder_inputs), encoder_size))
        if len(decoder_inputs) != decoder_size:
            raise ValueError("Decoder length must be equal to the one in bucket,"
                                             " %d != %d." % (len(decoder_inputs), decoder_size))
        if len(target_weights) != decoder_size:
            raise ValueError("Weights length must be equal to the one in bucket,"
                                             " %d != %d." % (len(target_weights), decoder_size))


        input_feed = {}
        for l in range(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in range(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]

        # Since our targets are decoder inputs shifted by one, we need one more.
        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)
        input_feed[self.decoder_emotions.name] = decoder_emotions
        input_feed[self.encoder_emotions.name] = encoder_emotions


        # Output feed: depends on whether we do a backward step or not.
        pos_acc = .0
        res_acc = .0
        if not forward_only: # Training
            output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                           self.gradient_norms[bucket_id],  # Gradient norm.
                           self.losses[bucket_id],
                           self.ppxes[bucket_id]]  # Loss for this batch.
            if use_autoEM:
                output_feed.append(self.autoEM_losses[bucket_id])         # Loss for this batch.
        else:
            if beam_search: # Decoding with beam_search
                output_feed = [self.beam_results[bucket_id],
                               self.beam_symbols[bucket_id],
                               self.beam_parents[bucket_id]]
            else: # Decoding with greedy_search
                output_feed = [self.ppxes[bucket_id]]
                if use_autoEM:
                    temp_output_feed = [self.autoEM_losses[bucket_id],
                                        self.pos_predics[bucket_id],
                                        self.res_predics[bucket_id],
                                        self.res_losses[bucket_id],
                                        self.res_cross_entropy[bucket_id], self.rlabels[bucket_id], self.res_logits[bucket_id],self.weight[bucket_id]]    # run the EM model to get the right emotion category
                    temp_outputs = session.run(temp_output_feed, input_feed)
                    pos_acc = self.get_accuracy(temp_outputs[1], [emotion[0] for emotion in encoder_emotions])
                    res_acc = self.get_accuracy(temp_outputs[2], [emotion[0] for emotion in decoder_emotions])


            for l in range(decoder_size):    # Output logits.
                output_feed.append(self.outputs[bucket_id][l])


        outputs = session.run(output_feed, input_feed) # TODO solve the interface problem

        if not forward_only:

            return outputs[1], outputs[3] , outputs[4]   # Gradient norm, ppx, senti_loss, autoEMloss
        else:
            if beam_search:
                return [outputs[0], outputs[1], outputs[2]], None, outputs[3:]  # beam_results,
            else:
                return outputs[0], outputs[1:], pos_acc, res_acc, temp_outputs[2]   # ppx, senti_loss, senti_infer, logits


    def get_batch(self, data, bucket_id, decode = False):

        encoder_size, decoder_size = self.buckets[bucket_id]
        encoder_inputs, decoder_inputs, encoder_emotions, decoder_emotions = [], [], [], []

        # Get a  random batch of encoder and decoder inputs from data,
        # pad them if needed, reverse encoder inputs and add GO to decoder.
        # All the data in each batch belongs to one emotion category.
        # The input and the target are re-indexed, but batch_weights aren't.
        _emotion = np.random.randint(6)
        for idx in range(self.batch_size):
            if decode:
                encoder_input, decoder_input, encoder_emotion, decoder_emotion = data[idx]
            else:
                decoder_emotion = [-1]
                while decoder_emotion[0] != _emotion:
                    encoder_input, decoder_input, encoder_emotion, decoder_emotion = random.choice(data[bucket_id])
            # TODO 为什么要设置一个batch拥有同一个decoder_emotion？
            # Encoder inputs are padded and then reversed.
            encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

            # Decoder inputs get an extra "GO" symbol, and are padded then.
            decoder_pad_size = decoder_size - len(decoder_input) - 1
            decoder_inputs.append([data_utils.GO_ID] + decoder_input +
                                                        [data_utils.PAD_ID] * decoder_pad_size)
            decoder_emotions.append(decoder_emotion)
            encoder_emotions.append(encoder_emotion)
        # Now we create batch-major vectors from the data selected above.
        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []
        batch_decoder_emotions = np.array(decoder_emotions, dtype=np.int32)
        batch_encoder_emotions = np.array(encoder_emotions, dtype=np.int32)

        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in range(encoder_size):
            batch_encoder_inputs.append(
                    np.array([encoder_inputs[batch_idx][length_idx]
                                        for batch_idx in range(self.batch_size)], dtype=np.int32))

        # Batch decoder inputs are re-indexed decoder_inputs, and we use weight musks.
        for length_idx in range(decoder_size):
            batch_decoder_inputs.append(
                    np.array([decoder_inputs[batch_idx][length_idx]
                                        for batch_idx in range(self.batch_size)], dtype=np.int32))

            # Create target_weights to be 0 for targets that are padding.
            batch_weight = np.ones(self.batch_size, dtype=np.float32)
            for batch_idx in range(self.batch_size):
                # We set weight to 0 if the corresponding target is a PAD symbol.
                # The corresponding target is decoder_input shifted by 1 forward.
                if length_idx < decoder_size - 1:
                    target = decoder_inputs[batch_idx][length_idx + 1]
                if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)
        return batch_encoder_inputs, batch_decoder_inputs, batch_weights, batch_encoder_emotions, batch_decoder_emotions
