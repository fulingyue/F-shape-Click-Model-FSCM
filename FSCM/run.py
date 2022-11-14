from Model import Model
from dataloader import DataLoader
import numpy as np
import tensorflow as tf
import random
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser('LWNCM')
    parser.add_argument('--train', action='store_true',
                        help='train the model')
    parser.add_argument('--valid', action='store_true',
                        help='evaluate the model on valid set')
    parser.add_argument('--test', action='store_true',
                        help='evaluate the model on test set')
    parser.add_argument('--rank', action='store_true',
                        help='rank on train set')

    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--optim', default='adam',
                                help='optimizer type')
    train_settings.add_argument('--learning_rate', type=float, default=1e-3,
                                help='learning rate')
    train_settings.add_argument('--weight_decay', type=float, default=1e-5,
                                help='weight decay')
    train_settings.add_argument('--momentum', type=float, default=0.99,
                                help='momentum')
    train_settings.add_argument('--dropout_rate', type=float, default=0.5,
                                help='dropout rate')
    train_settings.add_argument('--batch_size', type=int, default=1024,
                                help='train batch size')
    train_settings.add_argument('--num_steps', type=int, default=1000,
                                help='number of training steps')
    train_settings.add_argument('--filter_click',  action='store_true', help='whether filter those sessions which do not have a click')

    model_settings = parser.add_argument_group('model settings')
    model_settings.add_argument('--algo', default='LWNCM',
                                help='the name of the algorithm')
    model_settings.add_argument('--embed_size', type=int, default=4,
                                help='size of the embeddings')
    model_settings.add_argument('--hidden_size', type=int, default=128,
                                help='size of RNN hidden units')
    model_settings.add_argument('--attention_size', type=int, default=16,
                                help='size attention size')
    model_settings.add_argument('--feat_num', type=int, default=33)


    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--data_dir', default='data')
    path_settings.add_argument('--output_dir',default='outputs')

    path_settings.add_argument('--eval_freq', type=int, default=10,
                               help='the frequency of evaluating on the valid set when training')
    path_settings.add_argument('--check_point', type=int, default=10,
                               help='the frequency of saving model')
    path_settings.add_argument('--patience', type=int, default=5,
                               help='lr half when more than the patience times of evaluation\' loss don\'t decrease')
    path_settings.add_argument('--lr_decay', type=float, default=0.5,
                               help='lr decay')
    path_settings.add_argument('--load_model', type=int, default=-1,
                               help='load model global step')
    path_settings.add_argument('--data_parallel', type=bool, default=False,
                               help='data_parallel')
    path_settings.add_argument('--gpu_num', type=int, default=1,
                               help='gpu_num')

    return parser.parse_args()

def run():
    random_seed(904)
    args = parse_args()
    dataset = DataLoader(args)
    model = Model(args, dataset)
    model.train()

def check_path(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)

if __name__ == '__main__':
run()
