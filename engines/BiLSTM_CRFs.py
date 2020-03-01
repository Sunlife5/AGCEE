# -*- coding: utf-8 -*-
# @Time : 2019/6/2 上午10:55
# @Author : Scofield Phil
# @FileName: BiLSTM_CRFs.py
# @Project: sequence-lableing-vex

import math, os
from engines.utils import metrics, save_csv_, extractEntity
import numpy as np
import tensorflow as tf
import pandas as pd
import time
from engines.try_splice import sen2vec

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



# BiLSTM_CRFs(configs, logger, dataManager).build

class BiLSTM_CRFs(object):
    def __init__(self, configs, logger, dataManager):
        os.environ['CUDA_VISIBLE_DEVICES'] = configs.CUDA_VISIBLE_DEVICES  #0

        self.configs = configs
        self.logger = logger
        self.logdir = configs.log_dir
        self.measuring_metrics = configs.measuring_metrics
        self.dataManager = dataManager

        if configs.mode == "train":
            self.is_training = True
        else:
            self.is_training = False

        #result save
        self.checkpoint_name = configs.checkpoint_name
        self.checkpoints_dir = configs.checkpoints_dir
        self.output_test_file = configs.datasets_fold + "/" + configs.output_test_file
        self.is_output_sentence_entity = configs.is_output_sentence_entity
        self.output_sentence_entity_file = configs.datasets_fold + "/" + configs.output_sentence_entity_file

        self.biderectional = configs.biderectional     # true
        self.cell_type = configs.cell_type            # LSTM
        self.num_layers = configs.encoder_layers   # 神经网络层数  1层

        self.is_crf = configs.use_crf                # true

        self.learning_rate = configs.learning_rate   # 0.001
        self.dropout_rate = configs.dropout          # 0.5
        self.batch_size = configs.batch_size         # 32

        self.emb_dim = configs.embedding_dim         # 200
        self.hidden_dim = configs.hidden_dim         # 200

        if configs.cell_type == 'LSTM':
            if self.biderectional:
                self.cell = tf.nn.rnn_cell.LSTMCell(self.hidden_dim)
                # self.cell = tf.contrib.rnn.IndyLSTMCell(self.hidden_dim)
                # self.cell = tf.contrib.rnn.IndRNNCell(self.hidden_dim)
                # self.cell = tf.contrib.rnn.GLSTMCell(self.hidden_dim)

            else:
                self.cell =  tf.nn.rnn_cell.LSTMCell(2 * self.hidden_dim)
                # self.cell = tf.contrib.rnn.IndyLSTMCell(2 * self.hidden_dim)
                # self.cell = tf.contrib.rnn.IndRNNCell(2 * self.hidden_dim)
                # self.cell = tf.contrib.rnn.GLSTMCell(2 * self.hidden_dim)

        else:
            if self.biderectional:
                # self.cell = tf.nn.rnn_cell.GRUCell(self.hidden_dim)
                self.cell = tf.contrib.rnn.IndyGRUCell(self.hidden_dim)
            else:
                # self.cell = tf.nn.rnn_cell.GRUCell(2 * self.hidden_dim)
                self.cell = tf.contrib.rnn.IndyGRUCell(2 * self.hidden_dim)

        self.is_attention = configs.use_self_attention     # true
        self.attention_dim = configs.attention_dim         # 400

        self.num_epochs = configs.epoch                    # 300
        self.max_time_steps = configs.max_sequence_length  # 时间步长  300

        self.num_tokens = dataManager.max_token_number     # 总词数/token的总数
        self.num_classes = dataManager.max_label_number    # 总标签数/类别数

        self.is_early_stop = configs.is_early_stop         # True
        self.patient = configs.patient                     # 15

        self.max_to_keep = configs.checkpoints_max_to_keep  # 3
        self.print_per_batch = configs.print_per_batch      # 20
        self.best_f1_val = 0

        if configs.optimizer == 'Adagrad':                                                         # 选择优化器
            self.optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        elif configs.optimizer == 'Adadelta':
            self.optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)
        elif configs.optimizer == 'RMSprop':
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        elif configs.optimizer == 'GD':
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        else:
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

        self.initializer = tf.contrib.layers.xavier_initializer()                                  # 初始化权重
        self.global_step = tf.Variable(0, trainable=False, name="global_step", dtype=tf.int32)     # 启动全局变量

        if configs.use_pretrained_embedding:                                                       # 采用预训练的词嵌入
            embedding_matrix = dataManager.getEmbedding(configs.token_emb_dir)                     # token_emb_dir = data/example_datasets3/word.emb
            self.embedding = tf.Variable(embedding_matrix, trainable=False, name="emb", dtype=tf.float32)
        else:                                                                                           # 随机初始化变量
            self.embedding = tf.get_variable("emb", [self.num_tokens, self.emb_dim], trainable=True,
                                             initializer=self.initializer)

        self.build()
        self.logger.info("model initialed...\n")

        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    def build(self):
        # use_splice = 'True'
        self.targets = tf.placeholder(tf.int32, [16, self.max_time_steps])

        # print('run_target')
        self.inputs = tf.placeholder(tf.int32, [16, self.max_time_steps])

        self.sen_vec = tf.placeholder(tf.float32, [9600, 100])
        self.inputs_emb = tf.nn.embedding_lookup(self.embedding, self.inputs)

        self.sen_emb = tf.reshape(self.sen_vec, [-1, self.max_time_steps, 100])


        if self.is_attention == 'True':
            num_units = 100
            # num_heads = 8,
            dropout_rate = 0
            is_training = True
            causality = False
            scope = "multihead_attention"
            reuse = None
            queries = self.inputs_emb
            keys = self.sen_emb

            if num_units is None:
                # print('1111111')
                num_units = queries.get_shape().as_list[-1]  # 取queries的大小并形成列表，取最后一位

            padding_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k),降维，按行求和取绝对值后进入sign
            padding_masks = tf.tile(tf.expand_dims(padding_masks, 2), [1, 1, 300])  # 将padding_masks变成三维，然后复制
            paddings = tf.ones_like(keys) * (-2 ** 32 + 1)  # 创建与keys大小相同且全为（）的矩阵，
            outputs = tf.where(tf.equal(padding_masks, 0), paddings, keys)  # 找出padding和keys中对应padding_masks里非零位置的元素索引

            if causality:
                diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)  创建与outputs相同的矩阵
                # mask 把上三角的值全部设置为0
                tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k) 上三角为0
                masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)，加一维，复制

                paddings = tf.ones_like(masks) * (-2 ** 32 + 1)  # 创建与masks大小相同且权威（）的矩阵
                outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)，找出padding和outputs中对应masks里的元素索引


            multi_outputs = None
            for head in range(0, 3):
                Q = tf.layers.dense(outputs, num_units, activation=tf.nn.relu)  # (N, T_q, C),输出= relu（outputs*权重矩阵+偏置）
                K = tf.layers.dense(outputs, num_units, activation=tf.nn.relu)  # (N, T_k, C),输出= relu（outputs*权重矩阵+偏置）
                V = tf.layers.dense(outputs, num_units, activation=tf.nn.relu)  # (N, T_k, C),输出= relu（outputs*权重矩阵+偏置）

                output = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # 将K转置，Q和K转 矩阵乘法
                output = output / (K.get_shape().as_list()[-1] ** 0.5)  #
                output = tf.matmul(output, V)  # 矩阵乘法
                if multi_outputs is None:
                    multi_outputs = output
                else:
                    multi_outputs = tf.concat((multi_outputs, output), -1)  # 两个矩阵按最后一个维度连接

            outputs = tf.layers.dense(multi_outputs, 300, activation=tf.nn.relu)  # (N, T_k, C)，输出=relu（multi_outputs*权重矩阵）
            outputs = tf.layers.dropout(outputs, rate=dropout_rate,
                                            training=tf.convert_to_tensor(is_training))  # 将数据变成张量后dropout

            # Residual connection
            outputs += queries

            # Normalize
            inputs = outputs
            epsilon = 1e-8
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]

            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta = tf.Variable(tf.zeros(params_shape))
            gamma = tf.Variable(tf.ones(params_shape))
            normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
            outputs = gamma * normalized + beta
        #
        self.inputs_emb = tf.transpose(outputs, [1, 0, 2])
        # self.inputs_emb = tf.transpose(self.inputs_emb, [1, 0, 2])
        self.inputs_emb = tf.reshape(self.inputs_emb, [-1, self.emb_dim])
        self.inputs_emb = tf.split(self.inputs_emb, self.max_time_steps, 0)
        # print(self.inputs_emb)

        # lstm cell
        if self.biderectional:
            lstm_cell_fw = self.cell       # GLSTM
            lstm_cell_bw = self.cell       # GLSTM

            # dropout
            if self.is_training:
                lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_fw, output_keep_prob=(1 - self.dropout_rate))
                lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_bw, output_keep_prob=(1 - self.dropout_rate))

            lstm_cell_fw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_fw] * self.num_layers)
            lstm_cell_bw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_bw] * self.num_layers)

            # get the length of each sample
            self.length = tf.reduce_sum(tf.sign(self.inputs), reduction_indices=1)
            self.length = tf.cast(self.length, tf.int32)


            # forward and backward
            outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(
                lstm_cell_fw,
                lstm_cell_bw,
                self.inputs_emb,
                dtype=tf.float32,
                sequence_length=self.length
                # sequence_length=8
            )

            # print(self.inputs_emb)
            # print(self.length)
        else:
            lstm_cell = self.cell
            if self.is_training:
                lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=(1 - self.dropout_rate))
            lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * self.num_layers)
            self.length = tf.reduce_sum(tf.sign(self.inputs), reduction_indices=1)
            self.length = tf.cast(self.length, tf.int32)

            # print(self.length)
            # print(self.inputs_emb)

            outputs, _ = tf.contrib.rnn.static_rnn(
                lstm_cell,
                self.inputs_emb,
                dtype=tf.float32,
                sequence_length=self.length
            )
        # outputs: list_steps[batch, 2*dim]
        outputs = tf.concat(outputs, 1)
        outputs = tf.reshape(outputs, [self.batch_size, self.max_time_steps, self.hidden_dim * 2])



        # linear
        self.outputs = tf.reshape(outputs, [-1, self.hidden_dim * 2])                           # [9600, 600]
        # print(self.outputs.get_shape().as_list)
        self.softmax_w = tf.get_variable("softmax_w", [self.hidden_dim * 2, self.num_classes],
                                         initializer=self.initializer)                          # [600, 12]
        self.softmax_b = tf.get_variable("softmax_b", [self.num_classes], initializer=self.initializer)    # [12,]
        self.logits = tf.matmul(self.outputs, self.softmax_w) + self.softmax_b

        self.logits = tf.reshape(self.logits, [self.batch_size, self.max_time_steps, self.num_classes])    # [32，300，12]

        if not self.is_crf:
            # softmax
            softmax_out = tf.nn.softmax(self.logits, axis=-1)

            self.batch_pred_sequence = tf.cast(tf.argmax(softmax_out, -1), tf.int32)
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.targets)
            mask = tf.sequence_mask(self.length)

            self.losses = tf.boolean_mask(losses, mask)

            self.loss = tf.reduce_mean(losses)
        else:
            # crf
            self.log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
                self.logits, self.targets, self.length)                                                 # 预测标签，真实标签，序列长度
            self.batch_pred_sequence, self.batch_viterbi_score = tf.contrib.crf.crf_decode(self.logits,
                                                                                           self.transition_params,
                                                                                           self.length)

            self.loss = tf.reduce_mean(-self.log_likelihood)

        self.train_summary = tf.summary.scalar("loss", self.loss)
        self.dev_summary = tf.summary.scalar("loss", self.loss)

        self.opt_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

    def train(self):
        X_train, y_train, X_val, y_val = self.dataManager.getTrainingSet()
        # print(X_val)
        tf.initialize_all_variables().run(session=self.sess)              # 全局变量初始化

        saver = tf.train.Saver(max_to_keep=self.max_to_keep)              # 保存模型参数
        tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(self.logdir + "/training_loss", self.sess.graph)
        dev_writer = tf.summary.FileWriter(self.logdir + "/validating_loss", self.sess.graph)

        num_iterations = int(math.ceil(1.0 * len(X_train) / self.batch_size))
        num_val_iterations = int(math.ceil(1.0 * len(X_val) / self.batch_size))

        cnt = 0
        cnt_dev = 0
        unprogressed = 0
        very_start_time = time.time()
        best_at_epoch = 0
        self.logger.info("\ntraining starting" + ("+" * 20))
        for epoch in range(self.num_epochs):                       # 300个epoch
            start_time = time.time()
            # shuffle train at each epoch   每个epoch再次打乱句子顺序
            sh_index = np.arange(len(X_train))
            np.random.shuffle(sh_index)
            X_train = X_train[sh_index]
            y_train = y_train[sh_index]

            self.logger.info("\ncurrent epoch: %d" % (epoch))
            for iteration in range(num_iterations):                  # 迭代器个数为 句子总量和批次大小的商
                X_train_batch, y_train_batch = self.dataManager.nextBatch(X_train, y_train,
                                                                          start_index=iteration * self.batch_size)

                sen_emb = sen2vec(list(X_train_batch))


                _, loss_train, train_batch_viterbi_sequence, train_summary = \
                    self.sess.run([
                        self.opt_op,
                        self.loss,
                        self.batch_pred_sequence,
                        self.train_summary
                    ],
                        feed_dict={
                            self.inputs: X_train_batch,
                            self.targets: y_train_batch,
                            self.sen_vec: sen_emb
                        })

                if iteration % self.print_per_batch == 0:
                    cnt += 1
                    train_writer.add_summary(train_summary, cnt)

                    measures = metrics(X_train_batch, y_train_batch,
                                       train_batch_viterbi_sequence,
                                       self.measuring_metrics, self.dataManager)

                    res_str = ''
                    for k, v in measures.items():
                        res_str += (k + ": %.3f " % v)
                    self.logger.info("training batch: %5d, loss: %.5f, %s" % (iteration, loss_train, res_str))

            # validation
            loss_vals = list()
            val_results = dict()
            for measu in self.measuring_metrics:
                val_results[measu] = 0

            for iterr in range(num_val_iterations):
                cnt_dev += 1
                X_val_batch, y_val_batch = self.dataManager.nextBatch(X_val, y_val, start_index=iterr * self.batch_size)

                sen_emb = sen2vec(X_val_batch)

                loss_val, val_batch_viterbi_sequence, dev_summary = \
                    self.sess.run([
                        self.loss,
                        self.batch_pred_sequence,
                        self.dev_summary
                    ],
                        feed_dict={
                            self.inputs: X_val_batch,
                            self.targets: y_val_batch,
                            self.sen_vec: sen_emb
                        })

                measures = metrics(X_val_batch, y_val_batch, val_batch_viterbi_sequence,
                                   self.measuring_metrics, self.dataManager)
                dev_writer.add_summary(dev_summary, cnt_dev)

                for k, v in measures.items():
                    val_results[k] += v
                loss_vals.append(loss_val)

            time_span = (time.time() - start_time) / 60
            val_res_str = ''
            dev_f1_avg = 0
            for k, v in val_results.items():
                val_results[k] /= num_val_iterations
                val_res_str += (k + ": %.3f " % val_results[k])
                if k == 'f1': dev_f1_avg = val_results[k]

            self.logger.info("time consumption:%.2f(min),  validation loss: %.5f, %s" %
                             (time_span, np.array(loss_vals).mean(), val_res_str))
            if np.array(dev_f1_avg).mean() > self.best_f1_val:
                unprogressed = 0
                self.best_f1_val = np.array(dev_f1_avg).mean()
                best_at_epoch = epoch
                saver.save(self.sess, self.checkpoints_dir + "/" + self.checkpoint_name, global_step=self.global_step)
                self.logger.info("saved the new best model with f1: %.3f" % (self.best_f1_val))
            else:
                unprogressed += 1

            if self.is_early_stop:
                if unprogressed >= self.patient:
                    self.logger.info("early stopped, no progress obtained within %d epochs" % self.patient)
                    self.logger.info("overall best f1 is %f at %d epoch" % (self.best_f1_val, best_at_epoch))
                    self.logger.info(
                        "total training time consumption: %.3f(min)" % ((time.time() - very_start_time) / 60))
                    self.sess.close()
                    return
        self.logger.info("overall best f1 is %f at %d epoch" % (self.best_f1_val, best_at_epoch))
        self.logger.info("total training time consumption: %.3f(min)" % ((time.time() - very_start_time) / 60))
        self.sess.close()

    def test(self):
        X_test, y_test_psyduo_label, X_test_str = self.dataManager.getTestingSet()

        num_iterations = int(math.ceil(1.0 * len(X_test) / self.batch_size))
        self.logger.info("total number of testing iterations: " + str(num_iterations))

        self.logger.info("loading model parameter\n")
        tf.initialize_all_variables().run(session=self.sess)
        saver = tf.train.Saver()
        saver.restore(self.sess, tf.train.latest_checkpoint(self.checkpoints_dir))

        tokens = []
        labels = []
        entities = []
        entities_types = []
        self.logger.info("\ntesting starting" + ("+" * 20))
        for i in range(num_iterations):
            self.logger.info("batch: " + str(i + 1))
            X_test_batch = X_test[i * self.batch_size: (i + 1) * self.batch_size]
            X_test_str_batch = X_test_str[i * self.batch_size: (i + 1) * self.batch_size]
            y_test_psyduo_label_batch = y_test_psyduo_label[i * self.batch_size: (i + 1) * self.batch_size]

            if i == num_iterations - 1 and len(X_test_batch) < self.batch_size:
                X_test_batch = list(X_test_batch)
                X_test_str_batch = list(X_test_str_batch)
                y_test_psyduo_label_batch = list(y_test_psyduo_label_batch)
                gap = self.batch_size - len(X_test_batch)

                X_test_batch += [[0 for j in range(self.max_time_steps)] for i in range(gap)]
                X_test_str_batch += [['x' for j in range(self.max_time_steps)] for i in
                                     range(gap)]
                y_test_psyduo_label_batch += [[self.dataManager.label2id['O'] for j in range(self.max_time_steps)] for i
                                              in range(gap)]
                X_test_batch = np.array(X_test_batch)
                X_test_str_batch = np.array(X_test_str_batch)
                y_test_psyduo_label_batch = np.array(y_test_psyduo_label_batch)
                results, token, entity, entities_type, _ = self.predictBatch(self.sess, X_test_batch,
                                                                             y_test_psyduo_label_batch,
                                                                             X_test_str_batch)
                results = results[:len(X_test_batch)]
                token = token[:len(X_test_batch)]
                entity = entity[:len(X_test_batch)]
                entities_type = entities_type[:len(X_test_batch)]
            else:
                results, token, entity, entities_type, _ = self.predictBatch(self.sess, X_test_batch,
                                                                             y_test_psyduo_label_batch,
                                                                             X_test_str_batch)

            labels.extend(results)
            tokens.extend(token)
            entities.extend(entity)
            entities_types.extend(entities_type)

        def save_test_out(tokens, labels):
            # transform format
            newtokens, newlabels = [], []
            for to, la in zip(tokens, labels):
                newtokens.extend(to)
                newtokens.append("")
                newlabels.extend(la)
                newlabels.append("")
            # save
            save_csv_(pd.DataFrame({"token": newtokens, "label": newlabels}), self.output_test_file, ["token", "label"],
                      delimiter=self.configs.delimiter)

        save_test_out(tokens, labels)
        self.logger.info("testing results saved.\n")

        if self.is_output_sentence_entity:
            with open(self.output_sentence_entity_file, "w", encoding='utf-8') as outfile:
                for i in range(len(entities)):
                    if self.configs.label_level == 1:
                        outfile.write(' '.join(tokens[i]) + "\n" + "\n".join(entities[i]) + "\n\n")
                    elif self.configs.label_level == 2:
                        outfile.write(' '.join(tokens[i]) + "\n" + "\n".join(
                            [a + "\t(%s)" % b for a, b in zip(entities[i], entities_types[i])]) + "\n\n")

            self.logger.info("testing results with sentences&entities saved.\n")

        self.sess.close()

    def predict_single(self, sentence):
        X, Sentence, Y = self.dataManager.prepare_single_sentence(sentence)
        _, tokens, entitys, predicts_labels_entitylevel, indexs = self.predictBatch(self.sess, X, Y, Sentence)
        return tokens[0], entitys[0], predicts_labels_entitylevel[0], indexs[0]

    def predictBatch(self, sess, X, y_psydo_label, X_test_str_batch):
        entity_list = []
        tokens = []
        predicts_labels_entitylevel = []
        indexs = []
        predicts_labels_tokenlevel = []

        predicts_label_id, lengths = \
            sess.run([
                self.batch_pred_sequence,
                self.length
            ],
                feed_dict={
                    self.inputs: X,
                    self.targets: y_psydo_label,
                })

        for i in range(len(lengths)):
            x_ = [val for val in X_test_str_batch[i, 0:lengths[i]]]
            tokens.append(x_)

            y_pred = [str(self.dataManager.id2label[val]) for val in predicts_label_id[i, 0:lengths[i]]]
            predicts_labels_tokenlevel.append(y_pred)

            entitys, entity_labels, labled_indexs = extractEntity(x_, y_pred, self.dataManager)
            entity_list.append(entitys)
            predicts_labels_entitylevel.append(entity_labels)
            indexs.append(labled_indexs)

        return predicts_labels_tokenlevel, tokens, entity_list, predicts_labels_entitylevel, indexs

    def soft_load(self):
        self.logger.info("loading model parameter")
        tf.initialize_all_variables().run(session=self.sess)
        saver = tf.train.Saver()
        saver.restore(self.sess, tf.train.latest_checkpoint(self.checkpoints_dir))
        self.logger.info("loading model successfully")
