from .model import fragmentation_config as fconfig
from .model import ion_model
from .model import load_data
from .model.bucket_utils import write_buckets_mgf

import time
import os


def predict_pdeep2(param):

    predict_input = param['PredInputPath']
    predict_output = param['PredOutPath']

    model_folder = param['ModelForPredFolder']
    model = param['ModelForPredName'] + '.ckpt'

    nce = param['NCE']
    instrument = param['Instrument']

    config = fconfig.HCD_CommonMod_Config()

    config.SetFixMod(param['FixMod'])
    config.varmod = param['VarMod']

    ion_types = param['IonType']

    config.SetIonTypes(ion_types)
    config.time_step = 100
    config.min_var_mod_num = param['MinVarMod']
    config.max_var_mod_num = param['MaxVarMod']

    pdeep = ion_model.IonLSTM(config)

    start_time = time.perf_counter()

    buckets = load_data.load_peptide_file_as_buckets(predict_input, config, nce=nce, instrument=instrument)
    read_time = time.perf_counter()

    pdeep.LoadModel(model_file=os.path.join(model_folder, model))
    output_buckets = pdeep.Predict(buckets)
    predict_time = time.perf_counter()

    write_buckets_mgf(predict_output, buckets, output_buckets, config, iontypes=ion_types)

    print('read time = {:.3f}, predict time = {:.3f}'.format(read_time - start_time, predict_time - read_time))

    pdeep.close_session()
