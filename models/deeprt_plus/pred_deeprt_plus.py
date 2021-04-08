from .capsule_network_emb import CapsuleNet
from .RTdata_emb import Dictionary, Corpus
from .deeprt_metric import Pearson, Delta_t95

import copy
from os.path import join as join_path

import numpy as np
import pandas as pd

import torch
from torch.autograd import Variable


def pred_from_model(config,
                    conv1_kernel,
                    conv2_kernel,
                    param_path,
                    RTdata,
                    PRED_BATCH,
                    dictionary):
    '''
    write extracted features as np.array to pkl
    '''
    model = CapsuleNet(conv1_kernel, conv2_kernel, config, dictionary)
    model.load_state_dict(torch.load(param_path))
    model.cuda()

    print('>> note: predicting using the model:', param_path)

    pred = np.array([])

    # TODO: handle int
    # TODO: if Batch == 16, peptide number cannot be: 16X+1
    # TODO 最后一个 batch 的 size 如果不符合，digit capsule layer 的输入第 0 个维度 size 为 1（应该是 16）
    pred_batch_number = int(RTdata.test.shape[0] / PRED_BATCH) + 1
    for bi in range(pred_batch_number):
        test_batch = Variable(RTdata.test[bi * PRED_BATCH:(bi + 1) * PRED_BATCH, :])
        test_batch = test_batch.cuda()
        pred_batch = model(test_batch)
        pred = np.append(pred, pred_batch[0].data.cpu().numpy().flatten())
    return RTdata.test_label.numpy().flatten(), pred


def ensemble(obse, pred_list):
    pred_ensemble = copy.deepcopy(pred_list[0])
    for i in range(len(pred_list)-1):
        pred_ensemble += pred_list[i+1]
    pred_ensemble = pred_ensemble/len(pred_list)
    print('[ensemble %d] %.5f %.5f' % (len(pred_list), Pearson(obse, pred_ensemble), Delta_t95(obse, pred_ensemble)))

    return pred_ensemble


def ensemble1round(config, job_seed_round, conv1, conv2, minrt, maxrt, RTtest, dictionary):
    batch = 100
    obse, pred1 = pred_from_model(config, conv1, conv2, join_path(job_seed_round, 'epoch_10.pt'), RTtest, batch, dictionary)
    _, pred2 = pred_from_model(config, conv1, conv2, join_path(job_seed_round, 'epoch_12.pt'), RTtest, batch, dictionary)
    _, pred3 = pred_from_model(config, conv1, conv2, join_path(job_seed_round, 'epoch_14.pt'), RTtest, batch, dictionary)
    _, pred4 = pred_from_model(config, conv1, conv2, join_path(job_seed_round, 'epoch_16.pt'), RTtest, batch, dictionary)
    _, pred5 = pred_from_model(config, conv1, conv2, join_path(job_seed_round, 'epoch_18.pt'), RTtest, batch, dictionary)
    S = maxrt - minrt
    norm = lambda x: x * S + minrt
    obse, pred1, pred2, pred3, pred4, pred5 = norm(obse), norm(pred1), norm(pred2), norm(pred3), norm(pred4), norm(pred5)
    pred_ensemble = ensemble(obse, [pred1, pred2, pred3, pred4, pred5])
    return obse, pred_ensemble


def deeprt_pred(config):

    pred_input_path = config['PATH_TestSet']
    pred_output_path = config['PATH_Pred_Output']

    model_r1_path = config['PATH_R1_Model']
    model_r2_path = config['PATH_R2_Model']
    model_r3_path = config['PATH_R3_Model']

    model_r1_conv = config['Conv_R1_Model']
    model_r2_conv = config['Conv_R2_Model']
    model_r3_conv = config['Conv_R3_Model']

    min_rt = config['Min_RT']
    max_rt = config['Max_RT']

    max_len = config['Max_Len']

    if model_r2_path and model_r3_path:
        ensembl_on = True
    else:
        ensembl_on = False

    dictionary = Dictionary(config['AA_List'])
    corpus = Corpus(config, dictionary)

    if ensembl_on:
        obse, pred_r1 = ensemble1round(config, model_r1_path, model_r1_conv, model_r1_conv, min_rt, max_rt, corpus, dictionary)
        _, pred_r2 = ensemble1round(config, model_r2_path, model_r2_conv, model_r2_conv, min_rt, max_rt, corpus, dictionary)
        _, pred_r3 = ensemble1round(config, model_r3_path, model_r3_conv, model_r3_conv, min_rt, max_rt, corpus, dictionary)
        pred_ensemble = ensemble(obse, [pred_r1, pred_r2, pred_r3])
    else:
        obse, pred1 = pred_from_model(config, model_r1_conv, model_r1_conv, model_r1_path, corpus, 15, dictionary)
        pred_ensemble = pred1 * (max_rt - min_rt) + min_rt
        obse = obse * (max_rt - min_rt) + min_rt

    predict_seq_values = corpus.test_pepseq

    with open(pred_output_path, 'w') as fo:
        fo.write('seq\tobserved\tpredicted\n')
        for i in range(len(obse)):
            fo.write('{}\t{:5f}\t{:5f}\n'.format(predict_seq_values[i], obse[i], pred_ensemble[i]))

    print('seq_list_length: {}'.format(len(predict_seq_values)))
    print('observed_list_length: {}'.format(len(obse)))
    print('predicted_list_length: {}'.format(len(pred_ensemble)))
    print(">> note: prediction done!")

