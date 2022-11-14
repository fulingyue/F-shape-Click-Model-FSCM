import numpy as np
from sklearn import metrics
import tensorflow as tf
from tqdm import tqdm
from FSCM import FSCM

from args import *
import os


class Model(object):
    def __init__(self, args, dataset):
        self.args = args
        self.eval_freq = args.eval_freq
        self.learning_rate = args.learning_rate
        self.global_step = 0
        self.patience = args.patience
        # self.logger = logger
        self.model = FSCM(args, dataset.feat_size)
        self.dataset = dataset

    def save_pb(self, sess, to_path, name_list):
        """
        :param sess: run session
        :param to_path: .pb file
        :param name_list: out name, should be 'pred' in thi session
        :return: none
        """
        constant_graph = tf.graph_util.convert_variables_to_constants(
            sess,
            sess.graph_def,
            name_list)

        with tf.gfile.GFile(to_path, "wb") as f:
            f.write(constant_graph.SerializeToString())
        print("save {} ops within graph.".format(len(constant_graph.node)))

    def compute_perplexity(self, pred_logits, TRUE_CLICKS):
        pos_logits = np.log2(pred_logits + MINF)
        neg_logits = np.log2(1. - pred_logits + MINF)
        perplexity_at_rank = np.where(TRUE_CLICKS == 1, pos_logits, neg_logits)
        return perplexity_at_rank

    def compute_accuracy(self, pred_logits, TRUE_CLICKS, mask):
        round_logits = np.round(pred_logits)
        equal = round_logits == TRUE_CLICKS
        equal = np.multiply(mask, equal)
        if np.sum(equal) > np.sum(mask):
            print('equal > mask!!', round_logits[0], mask[0], TRUE_CLICKS[0])
        return np.sum(equal), np.sum(mask)

    def unfold(self, matrix, mask):
        matrix = np.reshape(matrix, (-1))
        mask = np.reshape(mask, (-1))
        assert mask.shape == matrix.shape
        idx = np.argwhere(mask > 0)
        return matrix[idx]

    def compute_ll(self, pred_logits, TRUE_CLICKS):
        loss = np.multiply(TRUE_CLICKS, np.log2(pred_logits + MINF)) + np.multiply(1. - TRUE_CLICKS,
                                                                             np.log2(1. - pred_logits + MINF))
        return loss

    def _train_epoch(self, train_batches, metric_save, patience, step_pbar, session):
        exit_tag = False
        for bitx, batch in enumerate(train_batches):
            if batch['row_feature'].shape[0] != self.args.batch_size:
                continue
            self.global_step += 1
            step_pbar.update(1)

            feed_dict = {
                self.model.row_feat_input: batch['row_feature'],
                self.model.col_feat_input: batch['col_feature'],
                self.model.row_click_input: batch['row_click'],
                self.model.col_click_input: batch['col_click'],
                self.model.row_loss_mask: batch['row_mask'],
                self.model.col_loss_mask: batch['col_mask'],
                self.model.row_true_clicks: batch['row_label'],
                self.model.col_true_clicks: batch['col_label'],
                self.model.col_length: batch['col_length'],
                self.model.row_length: batch['row_length'],
                self.model.keep_prob: self.args.dropout_rate
            }
            # self.model.row_loss = 0.0
            # self.model.col_loss = 0.0
            # loss1 = session.run([self.model.predict], feed_dict=feed_dict)
            # print(loss1)
            # exit()
            session.run(self.model.optim, feed_dict=feed_dict)
            loss = session.run([self.model.loss], feed_dict=feed_dict)

            print('epoch {}, loss is {}'.format(self.global_step, loss), end='\r')

            if self.global_step % self.eval_freq == 0:
                test_batches = self.dataset.gen_mini_batches('test', self.args.batch_size, shuffle=False)
                test_col_loss, test_row_loss, test_ll, test_col_ppl, test_row_ppl, test_accuracy, test_auc = self.evaluate(
                    test_batches, session, epoch=self.global_step)

                valid_batches = self.dataset.gen_mini_batches('valid', self.args.batch_size, shuffle=False)
                valid_col_loss, valid_row_loss, valid_ll, valid_col_ppl, valid_row_ppl, valid_accuracy, valid_auc = self.evaluate(
                    valid_batches, session)
                print(
                    '====epoch{}:  train loss= {}, valid row loss={}, valid col loss={}, valid ll = {}, valid col ppl={}, valid row ppl={}, valid accuracy={}, valid_auc = {}'
                        .format(self.global_step, loss, valid_col_loss, valid_row_loss, valid_ll, valid_col_ppl,
                                valid_row_ppl, valid_accuracy, valid_auc))

                print(
                    '====epoch{}: test col loss={},test row loss = {}, test_ll = {}, test col ppl={}, test row ppl = {}, test accuracy = {}, test_auc={}======='
                    .format(self.global_step, test_col_loss, test_row_loss, test_ll, test_col_ppl, test_row_ppl,
                            test_accuracy, test_auc))

                test_ppl = (test_row_ppl + test_col_ppl) / 2
                if test_ppl < self.best_PPL:
                    self.best_PPL = test_ppl
                    patience = 0
                else:
                    patience += 1

                if patience >= self.patience:
                    self.learning_rate *= self.args.lr_decay
                    self.model.lr *= self.args.lr_decay
                    print('lr is decayed to {}'.format(self.learning_rate))
                    patience = 0
                    self.patience += 1
            if self.global_step == 850:
                self.save_pb(session, os.path.join(self.args.data_dir, 'model', 'FSCM_final'),
                             ['row_logits', 'col_logits', 'row_loss', 'col_loss'])

            if self.global_step >= self.args.num_steps:
                exit_tag = True
        return exit_tag, metric_save, patience

    def train(self):
        tf.get_default_graph()
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        self.session.run(tf.local_variables_initializer())
        # self.writer.add_graph(session.graph)
        patience, metric_save = 0, 1e10
        step_pbar = tqdm(total=self.args.num_steps)
        exit_tag = False
        # self.writer.add_scalar('train/lr', self.args.learning_rate, self.global_step)
        self.best_PPL = 1e10
        while not exit_tag:
            train_batches = self.dataset.gen_mini_batches('train', self.args.batch_size, shuffle=True)
            exit_tag, metric_save, patience = self._train_epoch(train_batches, metric_save, patience,
                                                                step_pbar, self.session)
        writer = tf.summary.FileWriter('graph')
        writer.add_graph(self.session.graph)

    def evaluate(self, eval_batches, session=None, epoch=0):
        total_col_loss, total_row_loss, total_col_num, total_row_num = 0., 0., 0, 0
        whole_page_ll, whole_page_num = 0., 0.
        col_num_at_rank = np.zeros(MAX_COL_LENGTH, dtype=np.float32)
        row_num_at_rank = np.zeros(MAX_ROW_LENGTH, dtype=np.float32)

        col_perplexity_at_rank = np.zeros(MAX_COL_LENGTH, dtype=np.float32)
        row_perplexity_at_rank = np.zeros(MAX_ROW_LENGTH, dtype=np.float32)

        accurate_col_num, accurate_row_num = 0.0, 0.0
        accuracy_col_div, accuracy_row_div = 0.0, 0.0

        auc_col_logits, auc_col_labels = [], []
        auc_row_logits, auc_row_labels = [], []

        if self.session == 'None':
            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer)

        for b_itx, batch in enumerate(eval_batches):
            if batch['row_feature'].shape[0] != self.args.batch_size:
                continue
            batch_size = self.args.batch_size
            feed_dict = {
                self.model.row_feat_input: batch['row_feature'],
                self.model.col_feat_input: batch['col_feature'],
                self.model.row_click_input: batch['row_click'],
                self.model.col_click_input: batch['col_click'],
                self.model.row_loss_mask: batch['row_mask'],
                self.model.col_loss_mask: batch['col_mask'],
                self.model.row_true_clicks: batch['row_label'],
                self.model.col_true_clicks: batch['col_label'],
                self.model.col_length: batch['col_length'],
                self.model.row_length: batch['row_length'],
                self.model.keep_prob: self.args.dropout_rate
            }

            row_num = np.where(batch['row_length'].reshape(-1) > 0)[0].shape[0]

            _COL_TRUE_CLICKS = batch['col_label']
            ROW_TRUE_CLICKS = batch['row_label']

            col_loss, row_loss, _col_logits, row_logits = self.session.run(
                [self.model.col_loss, self.model.row_loss, self.model.col_logits, self.model.row_logits],
                feed_dict=feed_dict)

            col_logits = []
            col_length = batch['col_length']
            col_mask = []
            COL_TRUE_CLICKS = []
            for idx1, item in enumerate(_col_logits):
                col_logits_i = []
                col_true_clicks_i = []
                for idx2, block in enumerate(item):
                    col_logits_i += list(block[:col_length[idx1, idx2]])
                    col_true_clicks_i += list(_COL_TRUE_CLICKS[idx1, idx2, :col_length[idx1, idx2]])

                length = len(col_logits_i)
                col_logits_i += [0.0] * (MAX_COL_LENGTH - length)
                col_logits.append(col_logits_i)
                col_true_clicks_i += [0.0] * (MAX_COL_LENGTH - length)
                COL_TRUE_CLICKS.append(col_true_clicks_i)
                col_mask.append([1.0] * length + [0.0] * (MAX_COL_LENGTH - length))
                if length != np.sum(batch['col_mask'][idx1]):
                    print(batch['col_mask'][idx1])
                    print('length:', col_length[idx1])
                    exit(0)

            col_logits = np.array(col_logits)
            col_mask = np.array(col_mask)
            COL_TRUE_CLICKS = np.array(COL_TRUE_CLICKS)
            col_loss = col_loss * np.sum(batch['col_mask'])
            row_loss = row_loss * np.sum(batch['row_mask'])

            if np.sum(batch['col_mask']) != np.sum(col_mask):
                print(np.sum(batch['col_mask']), np.sum(col_mask))

            auc_col_logits.append(self.unfold(_col_logits, batch['col_mask']))
            auc_col_labels.append(self.unfold(_COL_TRUE_CLICKS, batch['col_mask']))
            auc_row_logits.append(self.unfold(row_logits, batch['row_mask']))
            auc_row_labels.append(self.unfold(ROW_TRUE_CLICKS, batch['row_mask']))

            total_col_loss += col_loss
            total_row_loss += row_loss
            total_col_num += np.sum(batch['col_mask'])
            total_row_num += np.sum(batch['row_mask'])

            whole_page_ll += col_loss + row_loss
            whole_page_num += np.sum(batch['col_mask']) + np.sum(batch['row_mask'])

            batch_col_perplexity_at_rank = self.compute_perplexity(col_logits, COL_TRUE_CLICKS)
            batch_row_perplexity_at_rank = self.compute_perplexity(row_logits, ROW_TRUE_CLICKS)
            col_perplexity_at_rank += np.sum((batch_col_perplexity_at_rank * col_mask), axis=0)
            row_perplexity_at_rank += np.sum(np.sum((batch_row_perplexity_at_rank * batch['row_mask']), axis=0), axis=0)

            col_num_at_rank += batch_size
            row_num_at_rank += row_num

            batch_accuracy_col_num, batch_accuracy_col_div = self.compute_accuracy(col_logits, COL_TRUE_CLICKS,
                                                                                   col_mask)
            accurate_col_num += batch_accuracy_col_num
            accuracy_col_div += batch_accuracy_col_div
            batch_accuracy_row_num, batch_accuracy_row_div = self.compute_accuracy(row_logits, ROW_TRUE_CLICKS,
                                                                                   batch['row_mask'])
            accurate_row_num += batch_accuracy_row_num
            accuracy_row_div += batch_accuracy_row_div

        total_col_loss = total_col_loss * 1.0 / total_col_num
        total_row_loss = total_row_loss * 1.0 / total_row_num
        loss = whole_page_ll / whole_page_num

        col_perplexity = 2 ** (-col_perplexity_at_rank / col_num_at_rank)

        # print(row_num_at_rank)
        row_perplexity = 2 ** (-row_perplexity_at_rank / row_num_at_rank)

        col_perplexity = np.sum(col_perplexity) / MAX_COL_LENGTH
        row_perplexity = np.sum(row_perplexity) / MAX_ROW_LENGTH

        col_accuracy = accurate_col_num / accuracy_col_div
        row_accuracy = accurate_row_num / accuracy_row_div
        accuracy_div = accuracy_col_div + accuracy_row_div
        accurate_num = accurate_col_num + accurate_row_num
        accuracy = [col_accuracy, row_accuracy, accurate_num / accuracy_div]

        auc_col_logits = np.concatenate(auc_col_logits, axis=0)
        auc_col_labels = np.concatenate(auc_col_labels, axis=0)
        auc_row_logits = np.concatenate(auc_row_logits, axis=0)
        auc_row_labels = np.concatenate(auc_row_labels, axis=0)
        auc_logits = np.concatenate([auc_col_logits, auc_row_logits], axis=0)
        auc_labels = np.concatenate([auc_col_labels, auc_row_labels], axis=0)
        col_auc = metrics.roc_auc_score(auc_col_labels, auc_col_logits)
        row_auc = metrics.roc_auc_score(auc_row_labels, auc_row_logits)
        auc = metrics.roc_auc_score(auc_labels, auc_logits)
        auc = [col_auc, row_auc, auc]

        if epoch == 850:
            col_ll = self.compute_ll(auc_col_logits, auc_col_labels)
            row_ll = self.compute_ll(auc_row_logits, auc_row_labels)
            all_ll = self.compute_ll(auc_logits, auc_labels)
            np.save(os.path.join(self.args.data_dir, 'FSCM_col_ll.npy'), col_ll)
            np.save(os.path.join(self.args.data_dir, 'FSCM_row_ll.npy'), row_ll)
            np.save(os.path.join(self.args.data_dir, 'FSCM_all_ll.npy'), all_ll)

            col_acc = np.array(np.round(auc_col_logits) == auc_col_labels)
            row_acc = np.array(np.round(auc_row_labels) == auc_row_logits)
            all_acc = np.array(np.round(auc_labels) == auc_logits)
            np.save(os.path.join(self.args.data_dir, 'FSCM_col_acc.npy'), col_acc)
            np.save(os.path.join(self.args.data_dir, 'FSCM_row_acc.npy'), row_acc)
            np.save(os.path.join(self.args.data_dir, 'FSCM_all_acc.npy'), all_acc)

            col_auc = np.array(metrics.roc_auc_score(auc_col_labels, auc_col_logits, average=None))
            row_auc = np.array(metrics.roc_auc_score(auc_row_labels, auc_row_logits, average=None))
            all_auc = np.array(metrics.roc_auc_score(auc_labels, auc_logits, average=None))

            np.save(os.path.join(self.args.data_dir, 'FSCM_col_auc.npy'), col_auc)
            np.save(os.path.join(self.args.data_dir, 'FSCM_row_auc.npy'), row_auc)
            np.save(os.path.join(self.args.data_dir, 'FSCM_all_auc.npy'), all_auc)


        return total_col_loss, total_row_loss, loss, col_perplexity, row_perplexity, accuracy, auc
