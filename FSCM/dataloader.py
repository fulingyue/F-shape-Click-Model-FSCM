import numpy as np
import pandas as pd
import os
import random
import pickle
from args import *

class DataLoader(object):
    def __init__(self, args):
        self.args = args
        self.gpu_num = args.gpu_num

        self.train_data = self.load_data('train')

        self.test_data = self.load_data('test')
        featuremap_file = open(os.path.join(args.data_dir, 'featureMap.pkl'), 'rb')
        featuremap = pickle.load(featuremap_file)
        self.feat_size = len(featuremap.values())


        index_list = list(range(self.test_data['col'].shape[0]))
        random.shuffle(index_list)
        valid_idx = index_list[:int(len(index_list) * 0.5)]
        test_idx = index_list[int(len(index_list) * 0.5):]
        self.valid_data = {'col': self.test_data['col'][valid_idx],
                           'row': self.test_data['row'][valid_idx],
                           'col_mask': self.test_data['col_mask'][valid_idx],
                           'row_mask':self.test_data['row_mask'][valid_idx],
                           'col_click': self.test_data['col_click'][valid_idx],
                           'col_feature': self.test_data['col_feature'][valid_idx],
                           'row_click': self.test_data['row_click'][valid_idx],
                           'row_feature': self.test_data['row_feature'][valid_idx],
                           'row_length': self.test_data['row_length'][valid_idx],
                           'col_length': self.test_data['col_length'][valid_idx]
                           }

        print('click num of valid set is', np.sum(self.valid_data['col_click']) + np.sum(self.valid_data['row_click']))

        self.test_data = {'col' : self.test_data['col'][test_idx],
                          'row':self.test_data['col'][test_idx],
                          'col_mask':self.test_data['col_mask'][test_idx],
                          'row_mask':self.test_data['row_mask'][test_idx],
                          'col_click': self.test_data['col_click'][test_idx],
                          'col_feature': self.test_data['col_feature'][test_idx],
                          'row_click': self.test_data['row_click'][test_idx],
                          'row_feature': self.test_data['row_feature'][test_idx],
                          'row_length': self.test_data['row_length'][test_idx],
                          'col_length': self.test_data['col_length'][test_idx]
                          }
        # print('click num of test set is', np.sum(self.test_data['col_click']) + np.sum(self.test_data['row_click']))



    def load_data(self, mode):
        col_data = []
        col_mask = []
        row_mask = []
        col_length = []
        row_length =[]
        col_prefix = np.zeros(0)
        click_num = 0

        if mode == 'train':
            paths = os.listdir(os.path.join(self.args.data_dir, 'train'))
            path_prefix = os.path.join(self.args.data_dir, 'train')
        else:
            paths = os.listdir(os.path.join(self.args.data_dir, 'test'))
            path_prefix = os.path.join(self.args.data_dir, 'test')

        row_file = np.array(pd.read_pickle(os.path.join(path_prefix, mode + '_row_data.pkl')))
        col_file = np.array(pd.read_pickle(os.path.join(path_prefix, mode + '_col_data.pkl')))

        for idx,row in enumerate(row_file):
            assert idx % 4 + 1 == row[1]
            row_mask.append([1] * row[2] + [0] * (MAX_ROW_LENGTH - row[2]))
            row_length.append(row[2])

        col_prefix = np.append(col_prefix, col_file[:, 3]) # click num

        for idx, col in enumerate(col_file):
            row_pos = col[4:8]
            click_num += col[3]
            list = col[8:].reshape(MAX_COL_LENGTH, FEAT_NUM +2)
            mask = [1] * col[0] + [0] * (MAX_COL_LENGTH - col[0])
            col_batch_data = np.zeros((COL_NUM, PER_COL_LENGTH, FEAT_NUM+2)) # feat + label
            col_batch_mask = np.zeros((COL_NUM, PER_COL_LENGTH))

            col_batch_data[0, :row_pos[0]-1, :] = list[:row_pos[0] -1, :]
            col_batch_mask[0, :row_pos[0]-1] = mask[:row_pos[0] - 1]

            row_id = col[4:8]
            col_length_i = []
            col_length_i.append(row_id[0] - 1)
            for i in range(4):
                if i == 3:
                    col_length_i.append(col[0] - np.sum(col_length_i))
                else:
                    col_length_i.append(np.maximum(0, (row_id[i+1]-row_id[i] - 1)))
            col_length.append(col_length_i)


            for i, beg in enumerate(row_pos):

                if i == ROW_NUM - 1:
                    end = 0
                else:
                    end = row_pos[i+1]


                if beg == 0 and end == 0:
                    # do nothing
                    pass
                elif beg != 0 and end == 0:
                    col_batch_data[i+1, :, :] = list[beg-1-i : beg- i -1 +PER_COL_LENGTH, :]
                    col_batch_mask[i+1, :] = mask[beg-1-i: beg-1-i+PER_COL_LENGTH]
                else:
                    assert beg != 0 and end != 0
                    col_batch_data[i+1, :end-beg-1, :] = list[beg-1-i: end-2-i, :]
                    col_batch_mask[i+1, :end-beg-1] = mask[beg-1-i:end-2-i]


            col_data.append(col_batch_data)
            col_mask.append(col_batch_mask)

        col_data = np.array(col_data) # (N, 5, 6, 35)
        col_label = col_data[:,:,:,0].squeeze() # (N, 5, 6)

        col_data = col_data[:,:,:, 2:] # (N, 5, 6, 33)


        session_num = col_prefix.shape[0]
        row_data = np.array(row_file).reshape(session_num, ROW_NUM, -1)
        append_tmp = np.zeros(session_num * COL_NUM).reshape(session_num, COL_NUM, 1)
        col_label = np.append(col_label, append_tmp, axis=2)

        row_prefix = row_data[:, :, :3]
        row_data = row_data[:, :, 3:].reshape(session_num, ROW_NUM, MAX_ROW_LENGTH, FEAT_NUM + 2)
        row_label = row_data[:, :, :, 0].squeeze()
        append_tmp = np.zeros(session_num * 4).reshape(session_num, ROW_NUM, 1)
        row_label = np.append(row_label, append_tmp, axis=2)  # (N,4, MAX_COL_LENGTH+1)
        row_feature = row_data[:, :, :, 2:]


        col_mask = np.array(col_mask, dtype=np.float32)
        row_mask = np.array(row_mask, dtype=np.float32).reshape(session_num, 4, -1)
        row_length = np.array(row_length).reshape(session_num, 4)
        col_length = np.array(col_length)

        data = {'col': col_data, 'row': row_data,
                'col_mask': col_mask, 'row_mask':row_mask,
                'col_click': col_label, 'col_feature': col_data,
                'row_click': row_label, 'row_feature':row_feature,
                'row_length':row_length, 'col_length': col_length
                }
        print('click num of ' + mode + 'is {}'.format(click_num))
        return data

    def _one_mini_batch(self, data, indices):
        """
        Get one mini batch ranking 0
        """

        batch_data = {
            'col_mask':data['col_mask'][indices],
            'row_mask':data['row_mask'][indices], # (N, 4, row_length)
            'col_feature': data['col_feature'][indices],
            'row_feature':data['row_feature'][indices],
            'col_click': np.array(data['col_click'][indices, :, :-1], dtype=np.int64),
            'row_click':np.array(data['row_click'][indices, :, :-1],dtype=np.int64),
            'col_label': np.array(data['col_click'][indices, :, 1:], dtype=np.float32),
            'row_label':np.array(data['row_click'][indices, :, 1:],dtype=np.float32),
            'col_length': data['col_length'][indices],
            'row_length': data['row_length'][indices, :]
            }

        return batch_data


    def gen_mini_batches(self, set_name, batch_size, shuffle=True):
        if set_name == 'train' :
            data = self.train_data
        elif set_name == 'test':
            data = self.test_data
        elif set_name == 'valid':
            data = self.valid_data
        else:
            print('no such dataset!')
            data = None

        data_size = data['col'].shape[0]
        indices = np.arange(data_size)
        if shuffle:
            np.random.shuffle(indices)
        indices = indices.tolist()
        # for 0 parallel in multi-gpu cases
        indices += indices[:(self.gpu_num - data_size % self.gpu_num) % self.gpu_num]
        for batch_start in np.arange(0, len(list(indices)), batch_size):
            batch_indices = indices[batch_start: batch_start + batch_size]
            yield self._one_mini_batch(data, batch_indices)
