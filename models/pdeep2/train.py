from .model import fragmentation_config as fconfig
from .model import ion_model
from .model import load_data
from .model.bucket_utils import merge_buckets, print_buckets, count_buckets

import os


def train_pdeep2(param):
    config = fconfig.HCD_CommonMod_Config()

    config.SetFixMod(param['FixMod'])
    config.varmod = param['VarMod']
    ion_types = param['IonType']
    config.SetIonTypes(ion_types)

    config.time_step = 100
    config.min_var_mod_num = param['MinVarMod']
    config.max_var_mod_num = param['MaxVarMod']

    config.tensorboard_dir = os.path.join(param['TensorboardFolder'], 'TensorBoard')

    pdeep = ion_model.IonLSTM(config)

    pdeep.learning_rate = 0.001
    pdeep.layer_size = 256
    pdeep.batch_size = 1024

    nce = param['NCE']
    instrument = param['Instrument']

    pdeep.epochs = param['Epochs']

    train_data_folder = param['TrainsetFolder']
    out_folder = param['ModelOutFolder']
    model_name = param['ModelOutName'] + '.ckpt'  # the model is saved as ckpt file

    try:
        os.makedirs(out_folder)
    except:
        pass

    pretrained_model = param['PretrainedModelPath']

    if pretrained_model:
        pdeep.BuildTransferModel(pretrained_model + ".ckpt")
    else:
        pdeep.build_model(input_size=98, output_size=config.GetTFOutputSize(), nlayers=2)

    buckets = {}
    buckets = merge_buckets(buckets, load_data.load_from_folder(train_data_folder, config, nce=nce, instrument=instrument))
    # you can add more plabel-containing folders here

    print('[I] train data:')
    print_buckets(buckets, print_peplen=False)
    buckets_count = count_buckets(buckets)
    print(buckets_count)
    print(buckets_count["total"])

    pdeep.TrainModel(buckets, save_as=os.path.join(out_folder, model_name))

    pdeep.close_session()
