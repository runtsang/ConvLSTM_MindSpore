from src.encoder import Encoder
from src.decoder import Decoder
from src.model import ED
from src.net_params import convlstm_encoder_params, convlstm_decoder_params
from src.MMDatasets import MovingMNIST

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
                    default='checkpoint_66_0.000961.ckpt',
                    type=str,
                    help='use which checkpoints')
parser.add_argument('-epochs', default=500, type=int, help='sum of epochs')
args = parser.parse_args()

random_seed = 1989
np.random.seed(random_seed)
mindspore.set_seed(random_seed)

save_dir = './save_model/'

testGenerator = MovingMNIST(is_train=False,
                          root='data/',
                          n_frames_input=args.frames_input,
                          n_frames_output=args.frames_output,
                          num_objects=[2])

testDataset = GeneratorDataset(testGenerator, column_names=['data', 'label']).batch(batch_size=args.batch_size)

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

def test():
    '''
    main function to run the training
    '''
    encoder = Encoder(encoder_params[0], encoder_params[1])
    decoder = Decoder(decoder_params[0], decoder_params[1])
    net = ED(encoder, decoder)

    # initialize the early_stopping object
    context.set_context(device_id=0)
    context.set_context(mode=context.PYNATIVE_MODE)
    
    load_checkpoint = os.path.join(save_dir, args.checkpoints)
    if os.path.exists(load_checkpoint):
        # load existing model
        print('==> loading existing model')
        model_info = mindspore.load_checkpoint(load_checkpoint)
        mindspore.load_param_into_net(net, model_info)
    else:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

    # to track the test loss as the model trains
    test_losses = []
    # to track the average test loss per epoch as the model trains
    avg_test_losses = []

    lossfunction = nn.MSELoss()

    # caculate the MAE MSE and SSIM
    total_mse, total_mae,total_ssim = 0,0,0

    ######################
    # test the model #
    ######################
    data_iterator = testDataset.create_tuple_iterator()
    for data, label in data_iterator:
        data = mindspore.Tensor(data, dtype=mindspore.float32)
        label = mindspore.Tensor(label, dtype=mindspore.float32)
        output = net(data)

        mse_batch = np.mean((output.asnumpy()-label.asnumpy())**2 , axis=(0,1,2)).sum()
        mae_batch = np.mean(np.abs(output.asnumpy()-label.asnumpy()) ,  axis=(0,1,2)).sum() 
        total_mse += mse_batch
        total_mae += mae_batch
        loss = lossfunction(output, label)
        loss_aver = float(loss) / args.batch_size
        test_losses.append(loss_aver)
        for a in range(0,label.asnumpy().shape[0]):
            for b in range(0,label.asnumpy().shape[1]):
                total_ssim += ssim(label.asnumpy()[a,b,0,], output.asnumpy()[a,b,0,]) / (label.asnumpy().shape[0]*label.asnumpy().shape[1]) 

    # print test statistics
    # calculate test loss over an epoch
    valid_loss = np.average(test_losses)
    avg_test_losses.append(valid_loss)

    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print_msg = (f'# TIME: {time}' +
                f'test_loss: {valid_loss:.6f} |   '+
                f'ssim: {(total_ssim / testDataset.get_dataset_size()):.6f} '+
                f'mae: {(total_mae / testDataset.get_dataset_size()):.6f} '+
                f'mse: {(total_mse / testDataset.get_dataset_size()):.6f} ')

    print(print_msg)

    return None

if __name__ == "__main__":
    test()