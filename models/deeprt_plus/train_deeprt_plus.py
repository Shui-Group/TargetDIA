from .capsule_network_emb import *

import timeit
import os
from tqdm import tqdm

from torch.optim import Adam
from torchnet.engine import Engine
import torchnet as tnt


def train_deeprt(config):

    T1 = timeit.default_timer()

    # read data ========== ========== ========== ========== ========== ==========
    dictionary = Dictionary(config['AA_List'])

    print('>> note: using >>>embedding<<< method.')
    corpus = Corpus(config, dictionary)

    # read data ========== ========== ========== ========== ========== ==========

    flog = open(config['PATH_Log'], 'w')

    conv_kernel = config[f'Conv_Train']
    model = CapsuleNet(conv_kernel, conv_kernel, config, dictionary)
    pretrain_path = config['PATH_PretrainModel']
    if '' == pretrain_path:
        pass
    else:
        print('-------- Param size --------')
        for _, __ in model.state_dict().items():
            print(_, __.shape)
        print('-------- Param size end --------')

        model.load_state_dict(torch.load(pretrain_path))  # epoch.pt
        print('>> note: load pre-trained model from:', pretrain_path)

    model.cuda()

    print("# parameters:", sum(param.numel() for param in model.parameters()))
    flog.write("# parameters:" + str(sum(param.numel() for param in model.parameters())) + '\n')

    optimizer = Adam(model.parameters(), lr=config['LR'])
    # optimizer = SGD(model.parameters(), lr = LR/10., momentum = 0.5)

    engine = Engine()
    meter_loss = tnt.meter.AverageValueMeter()

    capsule_loss = CapsuleLoss()

    data_train = corpus.train
    label_train = corpus.train_label

    def get_rt_iterator(mode):
        if mode:
            data = data_train  # Note: here must be FloatTensor not ByteTensor!
            labels = label_train
        else:
            data = corpus.test
            labels = corpus.test_label
            # print('>>dim: test data:', data.shape, labels.shape)
        tensor_dataset = tnt.dataset.TensorDataset([data, labels])
        return tensor_dataset.parallel(batch_size=config['BATCH_SIZE'], num_workers=1, shuffle=mode)  # 1 for heatmap

    def processor(sample):
        data, labels, training = sample
        # print('>>dim: data, labels, training', data.shape, labels.shape, training)
        # torch.Size([batch, 28, 28]) torch.Size([batch]) True

        # print('>>dim: data augmentation', data.shape) # torch.Size([batch, 1, 28, 28])
        # print('>>dim: labels', labels) # Note: labels is already LongTensor?

        # for regression, we use FloatTensor
        labels = torch.FloatTensor(labels.numpy())
        labels = labels.view(-1, 1)  # from [batch] to [batch, 1]

        data = Variable(data).cuda()
        labels = Variable(labels).cuda()

        if training:
            classes, reconstructions = model(data, labels)
        else:
            classes, reconstructions = model(data)

        loss = capsule_loss(data, labels, classes, reconstructions)

        return loss, classes

    def on_start_epoch(state):
        meter_loss.reset()
        state['iterator'] = tqdm(state['iterator'])

    def on_end_epoch(state):
        print('[Epoch %d] Training Loss: %.4f (MSE: %.4f)' % (
            state['epoch'], meter_loss.value()[0], 7)) # meter_mse.value()
        flog.write('[Epoch %d] Training Loss: %.4f (MSE: %.4f)\n' % (
            state['epoch'], meter_loss.value()[0], 7)) # meter_mse.value()

        meter_loss.reset()

        # iterator
        engine.test(processor, get_rt_iterator(False))

        print('[Epoch %d] Testing Loss: %.4f (MSE: %.4f)' % (
            state['epoch'], meter_loss.value()[0], 7)) # meter_mse.value()
        flog.write('[Epoch %d] Testing Loss: %.4f (MSE: %.4f)\n' % (
            state['epoch'], meter_loss.value()[0], 7)) # meter_mse.value()

        if config['MinEpochToSave'] <= state['epoch']:  # for heatmap
            torch.save(model.state_dict(), config['PATH_SavePrefix']+'/epoch_%d.pt' % state['epoch'])
            print('>> model: saved.')

        # prediction:
        # model.load_state_dict(torch.load(PATH))
        # pred_data = Variable(torch.FloatTensor(RTtest.X)[:,None,:,:])

        PRED_BATCH = 16  # 1000 # 16 for heatmap

        pred = np.array([])
        # TODO: handle int
        pred_batch_number = int(corpus.test.shape[0] / PRED_BATCH)+1
        for bi in range(pred_batch_number):
            test_batch = Variable(corpus.test[bi*PRED_BATCH:(bi+1)*PRED_BATCH, :])
            test_batch = test_batch.cuda()
            pred_batch = model(test_batch)
            pred = np.append(pred, pred_batch[0].data.cpu().numpy().flatten())
        # print('>>dim: pred', pred.shape)
        obse = corpus.test_label.numpy().flatten()
        pearson = Pearson(pred, obse)
        spearman = Spearman(pred, obse)

        print('>> Corr on %d testing samples: %.5f | %.5f' % (len(pred), pearson, spearman))
        flog.write('>> Corr on %d testing samples: %.5f | %.5f\n' % (len(pred), pearson, spearman))

        obse = corpus.test_label.numpy().flatten()

        with open(config['PATH_TrainResult'], 'w') as fo:
            fo.write('observed\tpredicted\n')
            for i in range(len(pred)):
                fo.write('%.5f\t%.5f\n' % (obse[i],pred[i]))
        # writing done

    try:
        os.makedirs(config['PATH_SavePrefix'])
    except:
        pass

    engine.hooks['on_sample'] = lambda state: state['sample'].append(state['train'])
    engine.hooks['on_forward'] = lambda state: meter_loss.add(state['loss'].data[0])
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch

    engine.train(processor, get_rt_iterator(True), maxepoch=config['EPOCHS'], optimizer=optimizer)

    T2 = timeit.default_timer()
    print('>> time: %.5f min\n' % ((T2-T1)/60.))
    flog.write('>> time: %.5f min\n' % ((T2-T1)/60.))
    flog.close()
