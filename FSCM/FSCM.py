import numpy as np
import tensorflow as tf
from args import *

class FSCM(object):
    def __init__(self, args, feat_size):
        self.args = args
        self.feat_num = FEAT_NUM
        self.feat_size = feat_size
        self.hidden_size = args.hidden_size
        self.embed_size = args.embed_size
        self.attention_size = args.attention_size
        self.lr = args.learning_rate
        self.batch_size = args.batch_size
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        self.col_feat_input = tf.placeholder(tf.int32, [self.batch_size, COL_NUM, PER_COL_LENGTH, FEAT_NUM], name='col_feat_input')
        self.row_feat_input = tf.placeholder(tf.int32, [self.batch_size, ROW_NUM, MAX_ROW_LENGTH, FEAT_NUM], name='row_feat_input')
        self.col_click_input = tf.placeholder(tf.int32, [self.batch_size, COL_NUM, PER_COL_LENGTH], name='col_click_input')
        self.row_click_input = tf.placeholder(tf.int32, [self.batch_size, ROW_NUM, MAX_ROW_LENGTH], name='row_click_input')

        self.col_loss_mask = tf.placeholder(tf.float32, [self.batch_size, COL_NUM, PER_COL_LENGTH], name='col_loss_mask')
        self.row_loss_mask = tf.placeholder(tf.float32, [self.batch_size, ROW_NUM, MAX_ROW_LENGTH], name='row_loss_mask')

        self.col_true_clicks = tf.placeholder(tf.float32, [self.batch_size, COL_NUM, PER_COL_LENGTH], name='col_true_clicks')
        self.row_true_clicks = tf.placeholder(tf.float32, [self.batch_size, ROW_NUM, MAX_ROW_LENGTH], name='row_true_clicks')

        self.col_length = tf.placeholder(tf.int32, [self.batch_size, COL_NUM])
        self.row_length = tf.placeholder(tf.int32, [self.batch_size, ROW_NUM])

        self.embedding()
        self.revisit()
        self.forward()
        self.optimize()

    def embedding(self):
        with tf.name_scope("encoding"):
            self.click_embedding = tf.get_variable('click_embedding', [2, self.embed_size // 2])
            col_click_embeded = tf.nn.embedding_lookup(self.click_embedding, self.col_click_input)
            row_click_embeded = tf.nn.embedding_lookup(self.click_embedding, self.row_click_input)

            self.feat_embedding = tf.get_variable('feat_embedding', [self.feat_size, self.embed_size])
            col_feat_embeded = tf.nn.embedding_lookup(self.feat_embedding, self.col_feat_input)
            col_feat_embeded = tf.reshape(col_feat_embeded, [-1, COL_NUM, PER_COL_LENGTH, FEAT_NUM * self.embed_size])
            row_feat_embeded = tf.nn.embedding_lookup(self.feat_embedding, self.row_feat_input)
            row_feat_embeded = tf.reshape(row_feat_embeded, [-1, ROW_NUM, MAX_ROW_LENGTH, FEAT_NUM* self.embed_size])

            self.col_embeded = tf.concat([col_click_embeded, col_feat_embeded], 3)
            self.row_embeded = tf.concat([row_click_embeded, row_feat_embeded], 3) # N, ROW_NUM, MAX_ROW_LENGTH, FEAT_NUM *self.embed_size +self.embedsize/2
            self.embeded_size = FEAT_NUM * self.embed_size + self.embed_size // 2

            col_feat_embeded = tf.reshape(col_feat_embeded, [-1, COL_NUM, PER_COL_LENGTH, FEAT_NUM, self.embed_size])
            self.col_user_embeded = tf.gather(col_feat_embeded, tf.constant(user_feat_idx), axis=3)
            self.col_user_embeded = tf.reshape(self.col_user_embeded, [-1,COL_NUM, PER_COL_LENGTH, user_feat_num * self.embed_size])
            self.col_item_embeded = tf.gather(col_feat_embeded, tf.constant(item_feat_idx), axis=3)
            self.col_item_embeded = tf.reshape(self.col_item_embeded, [-1, COL_NUM, PER_COL_LENGTH, item_feat_num * self.embed_size])

            row_feat_embeded = tf.reshape(row_feat_embeded, [-1, ROW_NUM, MAX_ROW_LENGTH, FEAT_NUM, self.embed_size])
            self.row_user_embeded = tf.gather(row_feat_embeded, tf.constant(user_feat_idx), axis=3)
            self.row_user_embeded = tf.reshape(self.row_user_embeded,[-1,ROW_NUM,MAX_ROW_LENGTH , user_feat_num * self.embed_size])
            self.row_item_embeded = tf.gather(row_feat_embeded, tf.constant(item_feat_idx), axis=3)
            self.row_item_embeded = tf.reshape(self.row_item_embeded, [-1, ROW_NUM,MAX_ROW_LENGTH, item_feat_num * self.embed_size])

    def col_model(self):
        with tf.name_scope('col_rnn'):
            self.col_ord_rnn_cell = tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.GRUCell(num_units=self.args.hidden_size, name='col_ord_gru'),output_keep_prob = self.keep_prob)
            self.col_merge_rnn_cell = tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.GRUCell(num_units=self.args.hidden_size, name='col_merge_gru'),output_keep_prob = self.keep_prob)

        with tf.name_scope('col_attention'):
            # input: (N, MAX_ROW_LENGTH, hidden_size)
            self.col_atten_W = tf.Variable(tf.random_normal([self.hidden_size, self.attention_size ], stddev=0.1), name='col_atten_W')
            self.col_atten_b = tf.Variable(tf.random_normal([ self.attention_size], stddev=0.1),
                                           name='col_atten_b')

            self.col_atten_u = tf.Variable(tf.random_normal([ self.attention_size,1], stddev=0.1),
                                           name='col_atten_u')


    def row_model(self):
        with tf.name_scope('row_rnn'):
            self.row_head_rnn_cell = tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.GRUCell(num_units=self.args.hidden_size, name='row_head_gru'),
                output_keep_prob=self.keep_prob)

            self.row_ord_rnn_cell = tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.GRUCell(num_units=self.args.hidden_size, name='row_ord_gru'),
                output_keep_prob=self.keep_prob)

        with tf.name_scope('row_attention'):
            # input: (N,  hidden_size)
            self.row_atten_W = tf.Variable(tf.random_normal([self.hidden_size, self.attention_size], stddev=0.1),
                                           name='row_atten_W')
            self.row_atten_b = tf.Variable(tf.random_normal([self.attention_size], stddev=0.1),
                                           name='row_atten_b')

            self.row_atten_u = tf.Variable(tf.random_normal([self.attention_size,1], stddev=0.1),
                                           name='row_atten_u')


    def mlp_vars(self):
        # mlp layer for four nodes
        with tf.name_scope('mlp'):
            self.revisit_size = self.embed_size * user_feat_num
            self.W1 = tf.Variable(tf.random_normal([self.hidden_size + self.revisit_size,1],stddev=0.1),
                                  name='W1')
            self.W2 = tf.Variable(tf.random_normal([self.hidden_size  + self.revisit_size,1], stddev=0.1),
                                  name='W2')
            self.W3 = tf.Variable(tf.random_normal([self.hidden_size + self.revisit_size,1], stddev=0.1),
                                  name='W3')
            self.W4 = tf.Variable(tf.random_normal([self.hidden_size + self.revisit_size,1], stddev=0.1),
                                  name='W4')

            self.b1 = tf.Variable([0.0], name='b1')
            self.b2 = tf.Variable([0.0], name='b2')
            self.b3 = tf.Variable([0.0], name='b3')
            self.b4 = tf.Variable([0.0], name='b4')

    def loss_with_sigmoid(self, predict, label, mask):
        predict = tf.sigmoid(predict)
        loss = tf.multiply(label, tf.math.log(predict + MINF)) + tf.multiply(1. - label, tf.math.log(1. - predict + MINF))
        loss = tf.multiply(loss, mask)
        return tf.reduce_sum(loss)



    def revisit(self):
        self.row_revisit = []
        self.col_revisit = []
        if comparison_mode == 'kernel':
            self.revisit_attention_W = tf.Variable(tf.random_normal([self.embed_size * user_feat_num, self.embed_size * user_feat_num]))
        elif comparison_mode == 'linear':
            self.revisit_attention_W = tf.Variable(tf.random_normal([self.embed_size * user_feat_num * 2, 1]), name='revisit_W')
            self.revisit_attention_b = tf.Variable([0.0], name='revisit_b')
        self.revisit_linear_for_v = tf.Variable(tf.random_normal([self.embed_size * item_feat_num, self.embed_size * user_feat_num]))

        prev_col_user_feat = None

        for i in range(ROW_NUM + COL_NUM):
            if i % 2 == 0:
                col_idx = i // 2
                length = tf.reshape(tf.squeeze(self.col_length[:, 0]) - 1, (-1,1))
                target_user_feat = []
                candidate_user_feat = [[], [], [], [], [], []]
                candidate_user_mask = [[], [], [], [], [], []]
                target_item_feat = []

                for j in range(PER_COL_LENGTH):
                    target_user_feat.append(tf.expand_dims(self.col_user_embeded[:, col_idx, j, :], axis=1)) # N, 1, M
                    target_item_feat.append(tf.expand_dims(self.col_item_embeded[:, col_idx, j, :], axis=1))
                    if j == 0:
                        if i == 0:
                            pass
                        else:
                            prev_row = self.row_user_embeded[:,col_idx-1, :, :] # (N,10,M)
                            candidate_user_feat[j].append(prev_row)
                            candidate_user_mask[j].append(tf.reshape(self.row_loss_mask[:, col_idx-1, :], [self.batch_size, MAX_ROW_LENGTH]))
                    else:
                        # precursor
                        candidate_user_feat[j].append(tf.expand_dims(self.col_user_embeded[:, col_idx, j-1, :], axis=1))
                        candidate_user_mask[j].append(tf.ones([self.batch_size, 1]))
                    #successor
                    if j != PER_COL_LENGTH - 1:
                        candidate_user_feat[j].append(tf.expand_dims(self.col_user_embeded[:, col_idx, j+1, :], axis=1))
                        candidate_user_mask[j].append(tf.ones([self.batch_size, 1]))
                    else:
                        if i == ROW_NUM + COL_NUM -1:
                            pass
                        else:
                            candidate_user_feat[j].append(self.row_user_embeded[:, col_idx, :, :])
                            candidate_user_mask[j].append(tf.reshape(self.row_loss_mask[:, col_idx, :], [self.batch_size, MAX_ROW_LENGTH]))
                # if i != ROW_NUM + COL_NUM - 1:
                # 
                #     for batch_id in range(self.batch_size):
                #         len = length[batch_id]
                #         candidate_user_mask[len][batch_id, 0, 0] = 0


                revisit_i = []
                for j in range(PER_COL_LENGTH):
                    mask_j = tf.concat(candidate_user_mask[j], axis=1) # N,x
                    candidate_j = tf.concat(candidate_user_feat[j], axis=1) # N,x,M
                    target_j = target_user_feat[j] # N, 1, M
                    item_target_j = tf.matmul(target_item_feat[j], self.revisit_linear_for_v)  # N, 1, M

                    if comparison_mode == 'kernel':
                        candidate_j = tf.expand_dims(candidate_j, axis=2) # N,x,1,M
                        target_j = tf.expand_dims(target_j, axis=3)
                        attention_score = tf.matmul(candidate_j, self.revisit_attention_W) # N,x,1,M
                        attention_score = tf.matmul(attention_score, target_j)
                        candidate_j = tf.squeeze(candidate_j, axis=2)
                        target_j = tf.squeeze(target_j, axis=3)  # N, 1, M

                    elif comparison_mode == 'product':
                        candidate_j = tf.expand_dims(candidate_j, axis=2)  # N,x,1,M
                        target_j = tf.expand_dims(target_j, axis=3)
                        attention_score = tf.matmul(candidate_j, target_j)
                        candidate_j = tf.squeeze(candidate_j, axis=2)
                        target_j = tf.squeeze(target_j, axis=3)  # N, 1, M
                    elif comparison_mode == 'linear':
                        x = candidate_j.shape[1]
                        attention_score = tf.matmul(tf.concat([tf.tile(target_j,[1, x, 1]), candidate_j],axis=2), self.revisit_attention_W) + self.revisit_attention_b
                    else:
                        print('no such comparison mode!')

                    attention_score = tf.reshape(attention_score,[self.batch_size, -1]) # N, x

                    attention_score_exp = tf.exp(attention_score)
                    attention_score_exp = tf.multiply(mask_j, attention_score_exp)
                    alpha = attention_score_exp / (MINF+ tf.expand_dims(tf.reduce_sum(attention_score_exp, axis=1),axis=1))

                    # N, x, M
                    alpha = tf.expand_dims(alpha, axis=2)
                    revisit_j = tf.reduce_sum(alpha * candidate_j, axis=1)  # N, M
                    revisit_j = tf.reshape(revisit_j, [self.batch_size, 1, self.embed_size * user_feat_num])


                    revisit_j = tf.multiply(revisit_j, target_j)  # N, 1, M
                    revisit_j = tf.multiply(revisit_j, item_target_j)  # N, 1, M

                    revisit_i.append(revisit_j)
                self.col_revisit.append(tf.expand_dims(tf.concat(revisit_i, axis=1), axis=1))  # N, 1, 6, M

                range_list = tf.reshape(tf.range(0, self.args.batch_size, 1), (-1, 1))
                indice = tf.concat([range_list, length], axis=1)
                prev_col_user_feat = tf.gather_nd(tf.squeeze(self.col_user_embeded[:, col_idx, :, :]), indices=indice)
                prev_col_user_feat = tf.reshape(prev_col_user_feat,
                                                [self.batch_size, 1, self.embed_size * user_feat_num])


            elif i % 2 == 1:
                # ROW
                row_idx =  i // 2
                target_user_feat = []
                candidate_user_feat = [[], [], [], [], [], [], [], [], [], []]
                target_item_feat = []
                for j in range(MAX_ROW_LENGTH):
                    target_user_feat.append(tf.expand_dims(self.row_user_embeded[:, row_idx, j, :], axis=1))
                    target_item_feat.append(tf.expand_dims(self.row_item_embeded[:, row_idx, j, :], axis=1))
                    # left item
                    if j != 0:
                        candidate_user_feat[j].append(tf.expand_dims(self.row_user_embeded[:, row_idx, j-1, :],axis=1))
                    # right item
                    if j != MAX_ROW_LENGTH - 1:
                        candidate_user_feat[j].append(tf.expand_dims(self.row_user_embeded[:, row_idx, j + 1, :], axis=1))
                    # 上方
                    candidate_user_feat[j].append(prev_col_user_feat)
                    # 下方
                    candidate_user_feat[j].append(tf.expand_dims(self.col_user_embeded[:, row_idx+1, 0, :], axis=1))


                revisit_i = []
                for j in range(MAX_ROW_LENGTH):
                    candidate_j = tf.concat(candidate_user_feat[j], axis=1)  # N,x,M
                    target_j = target_user_feat[j]  # N, 1, M
                    item_target_j = tf.matmul(target_item_feat[j], self.revisit_linear_for_v)  # N, 1, M

                    if comparison_mode == 'kernel':
                        candidate_j = tf.expand_dims(candidate_j, axis=2)  # N,x,1,M
                        target_j = tf.expand_dims(target_j, axis=3)
                        attention_score = tf.matmul(candidate_j, self.revisit_attention_W)  # N,x,1,M
                        attention_score = tf.matmul(attention_score, target_j)
                        target_j = tf.squeeze(target_j, axis=3)  # N, 1, M
                        candidate_j = tf.squeeze(candidate_j, axis=2)  # N, 2, M
                    elif comparison_mode == 'product':
                        candidate_j = tf.expand_dims(candidate_j, axis=2)  # N,x,1,M
                        target_j = tf.expand_dims(target_j, axis=3)
                        attention_score = tf.matmul(candidate_j, target_j)
                        target_j = tf.squeeze(target_j, axis=3)  # N, 1, M
                        candidate_j = tf.squeeze(candidate_j, axis=2)  # N, 2, M
                    elif comparison_mode == 'linear':
                        x = candidate_j.shape[1]
                        attention_score = tf.matmul(tf.concat([tf.tile(target_j, [1, x, 1]), candidate_j], axis=2),
                                                    self.revisit_attention_W) + self.revisit_attention_b
                    else:
                        print('no such comparison mode!')


                    attention_score = tf.reshape(attention_score, [self.batch_size, -1])  # N,2
                    attention_score_exp = tf.exp(attention_score)
                    alpha = attention_score_exp / (tf.expand_dims(tf.reduce_sum(attention_score_exp, axis=1), axis=1) + MINF)


                    alpha = tf.expand_dims(alpha, axis=2)
                    revisit_j = tf.reduce_sum(alpha * candidate_j, axis=1)  # N,M
                    revisit_j = tf.reshape(revisit_j, [self.batch_size, 1, self.embed_size * user_feat_num])


                    revisit_j = tf.multiply(revisit_j, target_j)  # N, 1, M
                    revisit_j = tf.multiply(revisit_j, item_target_j)  # N, 1, M

                    revisit_i.append(revisit_j)
                self.row_revisit.append(tf.expand_dims(tf.concat(revisit_i, axis=1), axis=1))  # N,1, 10,M

        self.row_revisit = tf.concat(self.row_revisit, axis=1) # N, R, 10, M
        self.col_revisit = tf.concat(self.col_revisit, axis=1) # N, C, 6, M



    def forward(self):

        self.col_model()
        self.row_model()
        self.mlp_vars()

        self.row_loss = 0.0
        self.col_loss = 0.0
        self.row_logits = []
        input_next = None
        block_skip_state = None
        block_skip_click = None

        for i in range(ROW_NUM + COL_NUM):
            if i == 0:
                # C1: no input
                input = tf.squeeze(self.col_embeded[:,0,:,:])
                length = tf.reshape(tf.squeeze(self.col_length[:, 0]) - 1, (-1,1))
                click = tf.squeeze(self.col_true_clicks[:,0,:])
                mask = tf.squeeze(self.col_loss_mask[:, 0,:])
                revisit = tf.squeeze(self.col_revisit[:,0, :, :]) # N, 6 ,M

                input = tf.reshape(input, [-1, PER_COL_LENGTH, self.embeded_size])
                col_rnn_output, _ = tf.nn.dynamic_rnn(cell=self.col_ord_rnn_cell,inputs=input, dtype = tf.float32)
                col_rnn_output = tf.reshape(col_rnn_output, [-1, PER_COL_LENGTH, self.hidden_size])
                col_linear_input = tf.concat([col_rnn_output, revisit], axis=2)
                predict = tf.matmul(col_linear_input, self.W1) + self.b1
                predict = tf.squeeze(predict)
                loss1 = self.loss_with_sigmoid(predict, click, mask)
                self.col_loss += loss1
                self.col_logits = [tf.expand_dims(tf.sigmoid(predict),axis=1)]
                range_list = tf.reshape(tf.range(0, self.args.batch_size, 1), (-1,1))
                indice = tf.concat([range_list, length], axis= 1)
                input_next = tf.gather_nd(col_rnn_output, indices=indice)
                input_next = tf.reshape(input_next, (self.args.batch_size, self.hidden_size))

                block_skip_state = tf.expand_dims(input_next, axis=1)  # (N, 1, hidden size)
                # block_skip_click = tf.expand_dims(input_click_next, axis=1)  # N, 1, embed size


            elif i % 2 == 1:
                # ROW
                input = tf.squeeze(self.row_embeded[:, i // 2, :, :])
                input = tf.reshape(input, [-1, MAX_ROW_LENGTH, self.embeded_size])
                # length = self.col_length[:,  i // 2] - 1
                click = tf.squeeze(self.row_true_clicks[:,  i // 2, :])
                mask = tf.squeeze(self.row_loss_mask[:,  i // 2, :])
                revisit = self.row_revisit[:, i // 2, :, :] # N, 10, M

                h0 = None
                predict = []

                attention_u = tf.tanh(tf.matmul( input_next, self.row_atten_W) + self.row_atten_b)
                exp_attention = tf.exp(tf.matmul(attention_u, self.row_atten_u))

                state_list = []
                for j in range(MAX_ROW_LENGTH):
                    if j == 0:
                        # ROW first
                        output, h0 = self.row_head_rnn_cell.__call__(input[:,j,:] , input_next)
                        linear_input = tf.concat([output, revisit[:,j,:]], axis=1)
                        predict.append(tf.matmul(linear_input, self.W3) + self.b3)
                        state_list.append(tf.expand_dims(h0, axis=1))
                    else:
                        attention_h0 = tf.tanh(tf.matmul(h0, self.row_atten_W) + self.row_atten_b)
                        exp_attention_h0 = tf.exp(tf.matmul(attention_h0, self.row_atten_u))
                        state = (exp_attention / (exp_attention_h0 + exp_attention)) * input_next +\
                                (exp_attention_h0 / (exp_attention_h0 + exp_attention)) * h0
                        output, h0 = self.row_ord_rnn_cell.__call__(input[:,j,:], state)
                        linear_input = tf.concat([output, revisit[:,j,:]], axis=1)
                        predict.append(tf.matmul(linear_input, self.W4) + self.b4)
                        state_list.append(tf.expand_dims(h0, axis=1))
                        input_next = h0

                predict = tf.concat(predict,axis=1) #(N,10)
                self.row_logits.append(tf.expand_dims(tf.sigmoid(predict),axis=1))
                self.row_loss += self.loss_with_sigmoid(predict, click, mask)

                state_list.append(block_skip_state)
                state_list = tf.concat(state_list, 1) # N,10,hidden_sz
                state_atten_u = tf.matmul(state_list, self.col_atten_W) + self.col_atten_b
                state_atten_exp = tf.exp(tf.matmul(state_atten_u, self.col_atten_u)) # N, 10, 1
                state_atten_exp = tf.squeeze(state_atten_exp)

                state_mask = mask
                state_mask = tf.concat([state_mask, tf.ones([self.batch_size, 1])], axis=1)
                state_atten_masked_exp = tf.multiply(state_atten_exp, state_mask) # N,10
                divisor = tf.reduce_sum(state_atten_masked_exp, axis=1)
                divisor += MINF
                divisor = tf.expand_dims(divisor, axis=1)
                alpha = tf.div(state_atten_masked_exp ,divisor) # N , 1
                alpha = tf.expand_dims(alpha, 2) # N,10, 1
                input_next = tf.reduce_sum(alpha * state_list, axis=1)  # N,1

            else:
                # COL
                input = tf.squeeze(self.col_embeded[:, i// 2, :, :])
                length = tf.reshape(tf.squeeze(self.col_length[:,  i // 2]) - 1, (-1,1))
                click = tf.squeeze(self.col_true_clicks[:, i // 2, :])
                mask = tf.squeeze(self.col_loss_mask[:, i // 2, :])
                revisit = self.col_revisit[:, i//2, :, :]

                input = tf.reshape(input, [-1, PER_COL_LENGTH, self.embeded_size])
                h0 = None
                predict = []
                state_list = []
                for j in range(PER_COL_LENGTH):
                    if j == 0:
                        output, h0 = self.col_merge_rnn_cell.__call__(input[:,j,:], input_next)
                        linear_input = tf.concat([output, revisit[:,j,:]], axis=1)
                        predict.append(tf.matmul(linear_input, self.W2) + self.b2)
                        state_list.append(tf.expand_dims(h0, axis=1))
                    else:
                        output, h0 = self.col_ord_rnn_cell.__call__(input[:,j,:], h0)
                        linear_input = tf.concat([output, revisit[:,j,:]], axis=1)
                        predict.append(tf.matmul(linear_input, self.W1) + self.b1)
                        state_list.append(tf.expand_dims(h0, axis=1))
                predict = tf.squeeze(tf.concat(predict,axis=1))
                self.col_loss += self.loss_with_sigmoid(predict, click, mask)
                self.col_logits.append(tf.expand_dims(tf.sigmoid(predict), axis=1))
                state_list = tf.concat(state_list, axis=1)# N,6,hidden_sz
                range_list = tf.reshape(tf.range(0, self.args.batch_size, 1), (-1, 1))
                indice = tf.concat([range_list, length], axis=1)
                input_next = tf.gather_nd(state_list,  indices=indice)
                input_next = tf.reshape(input_next, (self.args.batch_size, self.hidden_size))
                block_skip_state = tf.expand_dims(input_next, axis=1)  # (N, 1, hidden size)

    def optimize(self):
        self.row_loss /= tf.reduce_sum(self.row_loss_mask)
        self.row_loss = tf.to_float(self.row_loss, name='row_loss')
        self.col_loss /= tf.reduce_sum(self.col_loss_mask)
        self.col_loss = tf.to_float(self.col_loss, name='col_loss')
        self.loss = (self.col_loss + self.row_loss) / 2
        self.optim = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(-self.loss)
        self.col_logits = tf.concat(self.col_logits, axis=1)
        self.col_logits = tf.to_float(self.col_logits, name='col_logits')
        self.row_logits = tf.concat(self.row_logits, axis=1)
        self.row_logits = tf.to_float(self.row_logits, name='row_logits')

