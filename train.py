# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2022/07/26
@Author  :   zrainj
@Mail    :   rain1709@foxmail.com
@Description:   Based on MindSpore
'''
'''
@File    :   main.py
@Time    :   2020/03/09
@Author  :   jhhuang96
@Mail    :   hjh096@126.com
@Version :   1.0
@Description:   
'''

from src.encoder import Encoder
from src.decoder import Decoder
from src.model import ED
from src.net_params import convlstm_encoder_params, convlstm_decoder_params
from src.MMDatasets import MovingMNIST
from utils.earlystopping import EarlyStopping

import os
import numpy as np
import argparse
import datetime
from skimage.metrics import structural_similarity as ssim

import mindspore
import mindspore.nn as nn
from mindspore.dataset import GeneratorDataset
from mindspore import context

parser = argparse.ArgumentParser()
parser.add_argument('-clstm',
                    '--convlstm',
                    help='use convlstm as base cell',
                    action='store_true')
parser.add_argument('--batch_size',
                    default=24,
                    type=int,
                    help='mini-batch size')
parser.add_argument('-lr', default=1e-4, type=float, help='G learning rate')
parser.add_argument('-frames_input',
                    default=10,
                    type=int,
                    help='sum of input frames')
parser.add_argument('-frames_output',
                    default=10,
                    type=int,
                    help='sum of predict frames')
parser.add_argument('-checkpoints',
                    default='checkpoint_38_0.001265.ckpt',
                    type=str,
                    help='use which checkpoints')
parser.add_argument('-epochs', default=500, type=int, help='sum of epochs')
args = parser.parse_args()

random_seed = 1989
np.random.seed(random_seed)
mindspore.set_seed(random_seed)

save_dir = './save_model/'

trainGenerator = MovingMNIST(is_train=True,
                          root='data/',
                          n_frames_input=args.frames_input,
                          n_frames_output=args.frames_output,
                          num_objects=[3])
validGenerator = MovingMNIST(is_train=False,
                          root='data/',
                          n_frames_input=args.frames_input,
                          n_frames_output=args.frames_output,
                          num_objects=[3])

trainDataset = GeneratorDataset(trainGenerator, column_names=['data', 'label'], shuffle=True).batch(batch_size=args.batch_size, drop_remainder=True)

validDataset = GeneratorDataset(validGenerator, column_names=['data', 'label']).batch(batch_size=args.batch_size)

encoder_params = convlstm_encoder_params
decoder_params = convlstm_decoder_params

def init_group_params(net):
    decayed_params = []
    no_decayed_params = []
    for param in net.trainable_params():
        if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
            decayed_params.append(param)
        else:
            no_decayed_params.append(param)

    group_params = [{'params': decayed_params, 'weight_decay': 0.0},
                    {'params': no_decayed_params},
                    {'order_params': net.trainable_params()}]
    return group_params

def train():
    '''
    main function to run the training
    '''
    encoder = Encoder(encoder_params[0], encoder_params[1])
    decoder = Decoder(decoder_params[0], decoder_params[1])
    net = ED(encoder, decoder)

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=20, verbose=True)
    context.set_context(device_id=0)
    context.set_context(mode=context.PYNATIVE_MODE)
    
    load_checkpoint = os.path.join(save_dir, args.checkpoints)
    if os.path.exists(load_checkpoint):
        # load existing model
        print('==> loading existing model')
        model_info = mindspore.load_checkpoint(load_checkpoint)
        mindspore.load_param_into_net(net, model_info)
        cur_epoch = int(args.checkpoints[11:13])
    else:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        cur_epoch = 0

    lossfunction = nn.MSELoss()
    min_lr = 0.01
    max_lr = 0.1
    decay_steps = 4
    pla_lr_scheduler = nn.CosineDecayLR(min_lr, max_lr, decay_steps)
    group_params = init_group_params(net)
    optimizer = nn.Adam(group_params, learning_rate=pla_lr_scheduler)

    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    # mini_val_loss = np.inf

    # caculate the MAE MSE and SSIM
    total_mse, total_mae,total_ssim = 0,0,0

    ###################
    # train the model #
    ###################
    net_with_loss = nn.WithLossCell(net, lossfunction)
    train_network = nn.TrainOneStepCell(net_with_loss, optimizer)
    train_network.set_train()
    data_iterator = trainDataset.create_tuple_iterator(num_epochs=args.epochs)
    
    for i in range(cur_epoch, args.epochs):
        total_mse, total_mae,total_ssim = 0,0,0
        epoch_len = len(str(args.epochs))
        time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print_msg = (f'[{i:>{epoch_len}}/{args.epochs:>{epoch_len}}] ' +
                     f'TIME: {time} Start Trainning!' )
        print(print_msg)
        for data, label in data_iterator:
            loss = train_network(data, label)
            loss_aver = float(loss) / args.batch_size
            train_losses.append(loss_aver)

        train_loss = np.average(train_losses)
        time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print_msg = (f'# TIME: {time} Train End with loss: {train_loss:.6f}!' )
        print(print_msg)

        ######################
        # validate the model #
        ######################
        data_iterator = validDataset.create_tuple_iterator()
        for data, label in data_iterator:
            output = net(data)
            mse_batch = np.mean((output.asnumpy()-label.asnumpy())**2 , axis=(0,1,2)).sum()
            mae_batch = np.mean(np.abs(output.asnumpy()-label.asnumpy()) ,  axis=(0,1,2)).sum() 
            total_mse += mse_batch
            total_mae += mae_batch
            loss = lossfunction(output, label)
            loss_aver = float(loss) / args.batch_size
            valid_losses.append(loss_aver)
            for a in range(0,label.asnumpy().shape[0]):
                for b in range(0,label.asnumpy().shape[1]):
                    total_ssim += ssim(label.asnumpy()[a,b,0,], output.asnumpy()[a,b,0,]) / (label.asnumpy().shape[0]*label.asnumpy().shape[1]) 

        # print training/validation statistics
        # calculate average loss over an epoch
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print_msg = (f'# TIME: {time} Validate End with loss: {valid_loss:.6f}!' )
        print(print_msg)

        print_msg = (f'[{i:>{epoch_len}}/{args.epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.6f} ' +
                     f'valid_loss: {valid_loss:.6f} |   '+
                     f'ssim: {(total_ssim / validDataset.get_dataset_size()):.6f} '+
                     f'mae: {(total_mae / validDataset.get_dataset_size()):.6f} '+
                     f'mse: {(total_mse / validDataset.get_dataset_size()):.6f} ')

        print(print_msg)
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        early_stopping(valid_loss.item(), net, i, save_dir)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    with open("avg_train_losses.txt", 'wt') as f:
        for i in avg_train_losses:
            print(i, file=f)

    with open("avg_valid_losses.txt", 'wt') as f:
        for i in avg_valid_losses:
            print(i, file=f)
    return None

if __name__ == "__main__":
    train()
