from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import nltk
import os
import random
import sys
import time
import json
import numpy as np
from six.moves import range    # pylint: disable=redefined-builtin
import tensorflow as tf
import data_utils
import seq2seq_model
import configparser as ConfigParser

config = ConfigParser.RawConfigParser()
config.read('config')

sess_config = tf.ConfigProto() 
sess_config.gpu_options.allow_growth = True

# Network model hyprams
tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,"Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,"Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 128, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("hidden_size", 256, "Size of each model layer.")
tf.app.flags.DEFINE_integer("category", 6, "category of emotions.")
tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers in the model.")

tf.app.flags.DEFINE_integer("beam_size", 20, "Size of beam.")
tf.app.flags.DEFINE_integer("embedding_size", 200, "Size of word embedding.")
tf.app.flags.DEFINE_integer("emotion_size", 200, "Size of emotion embedding.")
tf.app.flags.DEFINE_integer("imemory_size", 256, "Size of imemory.")

tf.app.flags.DEFINE_integer("post_vocab_size", 40000, "post vocabulary size.")
tf.app.flags.DEFINE_integer("response_vocab_size", 40000, "response vocabulary size.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,"Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_boolean("use_fp16", False,"Train using fp16 instead of fp32.")

# Directories
tf.app.flags.DEFINE_string("data_dir", "train_data/", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "train/EACM", "Training directory.")
tf.app.flags.DEFINE_string("pretrain_dir", "pretrain", "Pretraining directory.")
tf.app.flags.DEFINE_integer("pretrain", 1484000, "pretrain model number")
tf.app.flags.DEFINE_integer("load_model", 0, "which model to load.")

# User interfaces
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 1000, "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("use_emb", True, "use embedding model")
tf.app.flags.DEFINE_boolean("use_imemory", False, "use imemory model")
tf.app.flags.DEFINE_boolean("use_ememory", False, "use ememory model")
tf.app.flags.DEFINE_boolean("use_autoEM", True, "use emotion aware setting")
tf.app.flags.DEFINE_boolean("decode", False, "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("beam_search", False, "beam search")


FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(12, 12), (16, 16), (20, 20), (30, 30)]

def read_data(path, max_size=None):
    data_set = [[] for _ in _buckets]
    data = json.load(open(path,'r'))
    size_max = 0
    count = 0
    for pair in data:
        count += 1
        post = pair[0]
        response = pair[1]
        source_ids = [int(x) for x in post[0]]
        target_ids = [int(x) for x in response[0]]
        target_ids.append(data_utils.EOS_ID)
        for bucket_id, (source_size, target_size) in enumerate(_buckets):
            if len(source_ids) < source_size and len(target_ids) < target_size:
                data_set[bucket_id].append([source_ids, target_ids, [int(post[1]), int(post[2])], [int(response[1]), int(response[2])]])
                break
    return data_set

def refine_data(bucket_data):
    new_data = []
    for each_bucket in bucket_data:
        b = []
        for e in range(6):
            b.append([x for x in each_bucket if x[-1][0] == e])
        new_data.append(b)
    return new_data


def create_model(session, forward_only, beam_search):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    model = seq2seq_model.Seq2SeqModel(
            FLAGS.post_vocab_size,
            FLAGS.response_vocab_size,
            _buckets,
            FLAGS.hidden_size,
            FLAGS.num_layers,
            FLAGS.max_gradient_norm,
            FLAGS.batch_size,
            FLAGS.learning_rate,
            FLAGS.learning_rate_decay_factor,
            embedding_size=FLAGS.embedding_size,
            forward_only=forward_only,
            beam_search=beam_search,
            beam_size=FLAGS.beam_size,
            category=FLAGS.category,
            use_emb=FLAGS.use_emb,
            use_autoEM=FLAGS.use_autoEM,
            use_imemory=FLAGS.use_imemory,
            use_ememory=FLAGS.use_ememory,
            emotion_size=FLAGS.emotion_size,
            imemory_size=FLAGS.imemory_size,
            dtype=dtype)
    see_variable = True
    if see_variable == True:
        for i in tf.all_variables():
            print(i.name, i.get_shape())
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    pre_ckpt = tf.train.get_checkpoint_state(FLAGS.pretrain_dir)
    if ckpt:
        if FLAGS.load_model == 0:
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            model.saver.restore(session, ckpt.model_checkpoint_path)
        else:
            path = ckpt.model_checkpoint_path[:ckpt.model_checkpoint_path.find('-')+1]+str(FLAGS.load_model)
            print("Reading model parameters from %s" % path)
            model.saver.restore(session, path)
    else:
        if pre_ckpt:
            session.run(tf.initialize_variables(model.initial_var))
            if FLAGS.pretrain > -1:
                path = pre_ckpt.model_checkpoint_path[:pre_ckpt.model_checkpoint_path.find('-')+1]+str(FLAGS.pretrain)
                print("Reading pretrain model parameters from %s" % path)
                model.pretrain_saver.restore(session, path)
            else:
                print("Reading pretrain model parameters from %s" % pre_ckpt.model_checkpoint_path)
                model.pretrain_saver.restore(session, pre_ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            session.run(tf.initialize_all_variables())
            vec_post, vec_response = data_utils.get_word_embedding(FLAGS.data_dir, FLAGS.post_vocab_size, FLAGS.response_vocab_size)  # 40000*100
            initvec_post = tf.constant(vec_post, dtype=dtype, name='init_wordvector_post')
            initvec_response = tf.constant(vec_response, dtype=dtype, name='init_wordvector_response')
            embedding_post = [x for x in tf.trainable_variables() if x.name == 'embedding_attention_seq2seq/RNN/EmbeddingWrapper/embedding:0'][0]
            embedding_response = [x for x in tf.trainable_variables() if x.name == 'embedding_attention_seq2seq/embedding_attention_decoder/embedding:0'][0]
            session.run(embedding_post.assign(initvec_post))
            session.run(embedding_response.assign(initvec_response))
        if FLAGS.use_ememory:
            vec_ememory = data_utils.get_ememory(FLAGS.data_dir, FLAGS.response_vocab_size)  # 6*40000
            initvec_ememory = tf.constant(vec_ememory, dtype=dtype, name='init_ememory')
            ememory = [x for x in tf.all_variables() if x.name == 'embedding_attention_seq2seq/embedding_attention_decoder/external_memory:0'][0]
            session.run(ememory.assign(initvec_ememory))

        if FLAGS.use_autoEM:
            senti_embedding, grammar_embedding = data_utils.get_pretrained_embedding(FLAGS.data_dir, FLAGS.post_vocab_size)
            initvec_senti = tf.constant(senti_embedding, dtype=dtype, name='initvec_senti')
            initvec_grammar = tf.constant(grammar_embedding, dtype=dtype, name='initvec_grammar')
            senti_tensor = [x for x in tf.trainable_variables() if
                              x.name == 'classify_model_with_buckets/senti_embed:0'][0]
            grammar_tensor = [x for x in tf.trainable_variables() if
                                  x.name == 'classify_model_with_buckets/grammar_embed:0'][0]
            session.run(senti_tensor.assign(initvec_senti))
            session.run(grammar_tensor.assign(initvec_grammar))


    return model


def train():
    print(FLAGS.__flags)
    # Prepare data.
    print("Preparing data in %s" % FLAGS.data_dir)
    train_path, dev_path, test_path, _, _ = data_utils.prepare_data(
            FLAGS.data_dir, FLAGS.post_vocab_size, FLAGS.response_vocab_size)

    with tf.Session(config=sess_config) as sess:

        # Create model.
        print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.hidden_size))
        model = create_model(sess, False, False)

        # Read data into buckets and compute their sizes.
        # Only dev_set will be refined.
        print ("Reading development and training data (limit: %d)."
                     % FLAGS.max_train_data_size)
        test_set= read_data(test_path)
        test_set = refine_data(test_set)
        dev_set = read_data(dev_path)
        dev_set = refine_data(dev_set)
        train_set = read_data(train_path, FLAGS.max_train_data_size)
        train_bucket_sizes = [len(train_set[b]) for b in range(len(_buckets))]
        train_total_size = float(sum(train_bucket_sizes))
        print([len(x) for x in dev_set])
        print([len(x) for x in train_set])
        # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
        # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
        # the size if i-th training bucket, as used later.
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                                                     for i in range(len(train_bucket_sizes))]
        # For example [0.43782635897014194, 0.7202605648906367, 0.8942405246857675, 1.0]
        print(train_buckets_scale)
        
        # This is the training loop.
        step_time, loss ,senti_loss = 0.0, 0.0, 0.0
        current_step = model.global_step.eval()
        epoch_steps = 4400000 / FLAGS.batch_size
        previous_losses = []

        # Summary variables
        sum_train_loss = tf.Variable(0.0,trainable=False,name="sum_train_loss",dtype=tf.float32)
        sum_dev_loss = tf.Variable(0.0, trainable=False, name="sum_dev_loss", dtype=tf.float32)
        sum_test_loss = tf.Variable(0.0, trainable=False, name="sum_test_loss", dtype=tf.float32)
        sum_autoEM_loss = tf.Variable(0.0, trainable=False, name="sum_autoEM_loss", dtype=tf.float32)
        tf.summary.scalar("Train_loss",sum_train_loss)
        tf.summary.scalar("Dev_loss", sum_dev_loss)
        tf.summary.scalar("Test_loss", sum_test_loss)
        tf.summary.scalar("autoEM_loss", sum_autoEM_loss)
        summary_path = FLAGS.train_dir + "/logs/"
        writer = tf.summary.FileWriter(summary_path, sess.graph)

        try:
            while True:
                # Choose a bucket according to data distribution. We pick a random number
                # in [0, 1] and use the corresponding interval in train_buckets_scale.
                random_number_01 = np.random.random_sample()
                bucket_id = min([i for i in range(len(train_buckets_scale))
                                                 if train_buckets_scale[i] > random_number_01])

                # Get a batch and make a step.Batch is formated as a list of arrays.
                start_time = time.time()
                encoder_inputs, decoder_inputs, target_weights, encoder_emotions, decoder_emotions = model.get_batch(
                        train_set, bucket_id)
                _, step_loss, autoEMloss = model.step(sess, encoder_inputs, decoder_inputs,
                                                                         target_weights, encoder_emotions, decoder_emotions, bucket_id, False, False, FLAGS.use_autoEM)
                if current_step % 10 == 0:
                    print('ok_{0}'.format(current_step))

                step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
                loss += step_loss / FLAGS.steps_per_checkpoint
                current_step += 1
                # Once in a while, we save checkpoint, print statistics, and run evals.
                if current_step % FLAGS.steps_per_checkpoint == 0:
                    # Print statistics for the previous epoch.
                    perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                    autoEM_perplexity = math.exp(float(autoEMloss))
                    print ("global step %d (%.2f epoch) learning rate %.4f step-time %.2f perplexity "
                                 "%.2f and autoEM_ppx is %.6f " % (model.global_step.eval(), model.global_step.eval() / float(epoch_steps), model.learning_rate.eval(), step_time, perplexity, autoEM_perplexity))


                    # Decrease learning rate if no improvement was seen over last 3 times.
                    if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                        sess.run(model.learning_rate_decay_op)
                    previous_losses.append(loss)

                    # Save checkpoint and zero timer and loss.
                    if current_step % (FLAGS.steps_per_checkpoint * 1) == 0:
                        checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
                        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                    step_time, loss = 0.0, 0

                    dev_ppx, dev_pos_acc, dev_res_acc= evaluate(sess, model, dev_set, "dev_set")
                    test_ppx, test_pos_acc, test_res_acc = evaluate(sess, model, test_set, "test_set")

                    # Write the summary
                    sess.run(sum_train_loss.assign(perplexity))
                    sess.run(sum_dev_loss.assign(dev_ppx))
                    sess.run(sum_test_loss.assign(test_ppx))
                    sess.run(sum_autoEM_loss.assign(autoEM_perplexity))
                    merged = tf.summary.merge_all()
                    summaries = sess.run(merged)
                    writer.add_summary(summaries, current_step)

        except KeyboardInterrupt as e:
            print(e)
            checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
            model.saver.save(sess, checkpoint_path, global_step=model.global_step)
            print("Exit training with the model saved at step %d" % model.global_step.eval())


def evaluate(sess, model, data_set, name):
    # dev set evaluation
    total_loss = .0
    total_len = .0
    total_pos_acc = .0
    total_res_acc = .0
    for bucket_id in range(len(_buckets)):
        if len(data_set[bucket_id]) == 0:
            print("    eval: empty bucket %d" % (bucket_id))
            continue
        bucket_loss = .0
        bucket_len = .0
        res_bucket_acc = .0
        pos_bucket_acc = .0
        for e in range(6):
            len_data = len(data_set[bucket_id][e])
            for batch in range(0, len_data, FLAGS.batch_size):
                step = min(FLAGS.batch_size, len_data - batch)
                model.batch_size = step
                encoder_inputs, decoder_inputs, target_weights, encoder_emotions, decoder_emotions = model.get_batch(
                    data_set[bucket_id][e][batch:batch + step], bucket_id, decode=True)
                step_loss,  _, pos_acc, res_acc, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                                                               target_weights, encoder_emotions,
                                                                               decoder_emotions, bucket_id, True, False,
                                                                               FLAGS.use_autoEM)
                bucket_loss += step_loss * step
                res_bucket_acc += res_acc * step
                pos_bucket_acc += pos_acc * step
            bucket_len += len_data
        total_loss += bucket_loss
        total_len += bucket_len
        bucket_loss = float(bucket_loss / bucket_len)
        res_bucket_acc = float(res_bucket_acc / bucket_len)
        pos_bucket_acc = float(pos_bucket_acc / bucket_len)
        bucket_ppx = math.exp(bucket_loss) if bucket_loss < 300 else float("inf")
        total_pos_acc += pos_bucket_acc
        total_res_acc += res_bucket_acc
        print("    "+ name +" eval: bucket %d perplexity %.2f and post_acc is %.5f resp_acc is %.5f" % (
        bucket_id, bucket_ppx, pos_bucket_acc, res_bucket_acc))
    total_loss = float(total_loss / total_len)
    total_pos_acc  = float(total_pos_acc / len(_buckets))
    total_res_acc = float(total_res_acc / len(_buckets))
    total_ppx = math.exp(total_loss) if total_loss < 300 else float(
        "inf")
    print("    "+ name +" eval: bucket avg perplexity %.2f and post_acc is %.5f resp_acc is %.5f" % (total_ppx, total_pos_acc, total_res_acc))
    sys.stdout.flush()
    model.batch_size = FLAGS.batch_size
    return total_ppx, total_pos_acc, total_res_acc

def decode():

    # TODO(jiayi): Implement the segmentation pre-processing.

    try:
        from wordsegment import Global
    except:
        Global = None

    def split(sent):
        sent = sent
        if Global == None:
            return sent.split(' ')
        tuples = [(word, pos) for word, pos in Global.GetTokenPos(sent)]
        return [each[0] for each in tuples]

    with tf.Session(config=sess_config) as sess:
        with tf.device("/cpu:0"):
            # Create model and load parameters.
            model = create_model(sess, True, FLAGS.beam_search)
            model.batch_size = 1    # We decode one sentence at a time.
            beam_search = FLAGS.beam_search
            beam_size = FLAGS.beam_size
            num_output = 5

            # Load vocabularies.
            print(config.get('data', "post_vocab_file"))
            post_vocab_path = os.path.join(FLAGS.data_dir, config.get('data', 'post_vocab_file') % FLAGS.post_vocab_size)
            response_vocab_path = os.path.join(FLAGS.data_dir, config.get('data', 'response_vocab_file') % FLAGS.response_vocab_size)
            post_vocab, _ = data_utils.initialize_vocabulary(post_vocab_path)
            _, rev_response_vocab = data_utils.initialize_vocabulary(response_vocab_path)

            # Decode from test.post.
            start_time = time.time()
            read_path = "train/test.post"
            write_path = FLAGS.train_dir + "/results"

            with open(read_path, "r+") as f1:
                test_post = f1.readlines()
                cnt = 0
                flag = 0
                all_responses = []
                check_list = [i for i in range(200)]
                # check_list = random.sample(range(1, len(test_post)-1),100)
                for each_post in test_post:
                    cnt += 1
                    if (cnt in check_list and len(each_post.split(" ")) > 4):
                        flag = 1
                    else:
                        flag = 0

                    each_post = " ".join(split(each_post))
                    token_ids = data_utils.sentence_to_token_ids(each_post, post_vocab)
                    int2emotion = ['null', 'like', 'sad', 'disgust', 'angry', 'happy']
                    for decoder_emotion in range(1, 2):
                        decoder_emotion = [0,0]
                        bucket_id = min([b for b in range(len(_buckets))
                                         if _buckets[b][0] > len(token_ids)])
                        encoder_inputs, decoder_inputs, target_weights, encoder_emotions, decoder_emotions = model.get_batch(
                            [[token_ids, [], [0,0], decoder_emotion]], bucket_id, decode=True)
                        results, output_logits, acc1, accuracy, em_predict = model.step(sess, encoder_inputs, decoder_inputs,
                                                               target_weights, encoder_emotions, decoder_emotions, bucket_id, True, beam_search,FLAGS.use_autoEM)
                        if beam_search:
                            result = results[0]
                            symbol = results[1]
                            parent = results[2]
                            res = []
                            nounk = []
                            for i, (prb, _, prt) in enumerate(result):
                                if len(prb) == 0: continue
                                for j in range(len(prb)):
                                    p = prt[j]
                                    s = -1
                                    output = []
                                    for step in range(i - 1, -1, -1):
                                        s = symbol[step][p]
                                        p = parent[step][p]
                                        output.append(s)
                                    output.reverse()
                                    if data_utils.UNK_ID in output:
                                        res.append([prb[j][0], " ".join(
                                            [tf.compat.as_str(rev_response_vocab[int(x)]) for x in output])])
                                    else:
                                        nounk.append([prb[j][0], " ".join(
                                            [tf.compat.as_str(rev_response_vocab[int(x)]) for x in output])])
                            res.sort(key=lambda x: x[0], reverse=True)
                            nounk.sort(key=lambda x: x[0], reverse=True)
                            if len(nounk) < beam_size:
                                res = nounk + res[:(num_output - len(nounk))]
                            else:
                                res = nounk
                            for i in res[:num_output]:
                                print(int2emotion[decoder_emotion[0]] + ': ' + i[1])
                        else:
                            # This is a greedy decoder - outputs are just argmaxes of output_logits.
                            outputs = [
                                int(np.argmax(np.split(logit, [2, FLAGS.response_vocab_size], axis=1)[1], axis=1) + 2)
                                for logit in output_logits]
                            # If there is an EOS symbol in outputs, cut them at that point.
                            if data_utils.EOS_ID in outputs:
                                outputs = outputs[:outputs.index(data_utils.EOS_ID)]
                            # Print out response sentence corresponding to outputs.
                            cur_response = "".join(
                                [tf.compat.as_str(rev_response_vocab[output]) for output in outputs])
                            all_responses.append(cur_response)
                            if(flag):
                                print('\n' + "Post%d:" % cnt + each_post + 'Response%d : '% cnt + cur_response)
                                print("Predict response emotion category : %s" % int2emotion[em_predict[0]])

            # Save the result at test.response.
            if not os.path.exists(write_path):
                os.makedirs(write_path)
            model_name = FLAGS.train_dir.split("/")[-1]
            write_name = write_path + "/" + model_name + str(model.global_step.eval())
            with open(write_name, 'w+') as f2:
                f2.write("\n".join(all_responses))
            print("Time cost : %f" % (time.time() - start_time))

            # Decode from standard input.
            sys.stdout.write("> ")
            sys.stdout.flush()
            sentence = sys.stdin.readline()
            while sentence:
                sentence = " ".join(split(sentence))
                # Get token-ids for the input sentence.
                token_ids = data_utils.sentence_to_token_ids(sentence, post_vocab)
                int2emotion = ['null', 'like', 'sad', 'disgust', 'angry', 'happy']
                for decoder_emotion in range(1, 2):
                    decoder_emotion = [0, 0]
                    bucket_id = min([b for b in range(len(_buckets))
                                     if _buckets[b][0] > len(token_ids)])
                    encoder_inputs, decoder_inputs, target_weights, encoder_emotions, decoder_emotions = model.get_batch(
                        [[token_ids, [], [0, 0], decoder_emotion]], bucket_id, decode=True)
                    results, output_logits,  acc1, accuracy, em_predict = model.step(sess, encoder_inputs, decoder_inputs,
                                                                          target_weights, encoder_emotions,
                                                                          decoder_emotions, bucket_id, True,
                                                                          beam_search, FLAGS.use_autoEM)
                    if beam_search:
                        result = results[0]
                        symbol = results[1]
                        parent = results[2]
                        res = []
                        nounk = []
                        for i, (prb, _, prt) in enumerate(result):
                            if len(prb) == 0: continue
                            for j in range(len(prb)):
                                p = prt[j]
                                s = -1
                                output = []
                                for step in range(i - 1, -1, -1):
                                    s = symbol[step][p]
                                    p = parent[step][p]
                                    output.append(s)
                                output.reverse()
                                if data_utils.UNK_ID in output:
                                    res.append([prb[j][0], " ".join(
                                        [tf.compat.as_str(rev_response_vocab[int(x)]) for x in output])])
                                else:
                                    nounk.append([prb[j][0], " ".join(
                                        [tf.compat.as_str(rev_response_vocab[int(x)]) for x in output])])
                        res.sort(key=lambda x: x[0], reverse=True)
                        nounk.sort(key=lambda x: x[0], reverse=True)
                        if len(nounk) < beam_size:
                            res = nounk + res[:(num_output - len(nounk))]
                        else:
                            res = nounk
                        for i in res[:num_output]:
                            print(int2emotion[decoder_emotion] + ': ' + i[1])
                    else:
                        # This is a greedy decoder - outputs are just argmaxes of output_logits.
                        outputs = [
                            int(np.argmax(np.split(logit, [2, FLAGS.response_vocab_size], axis=1)[1], axis=1) + 2)
                            for logit in output_logits]
                        # If there is an EOS symbol in outputs, cut them at that point.
                        if data_utils.EOS_ID in outputs:
                            outputs = outputs[:outputs.index(data_utils.EOS_ID)]
                        # Print out response sentence corresponding to outputs.
                        cur_response = "".join(
                            [tf.compat.as_str(rev_response_vocab[output]) for output in outputs])
                        print('\n' + "Post: " + sentence + 'Response: ' + cur_response)
                        print("Predict response emotion category : %s" % int2emotion[em_predict[0]])
                print("> ", end="")
                sys.stdout.flush()
                sentence = sys.stdin.readline()

def main(_):
    if FLAGS.decode:
        decode()
    else:
        train()

if __name__ == "__main__":
    tf.app.run()
