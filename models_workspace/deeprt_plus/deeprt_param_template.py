import json
import os


AAList_20 = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
             'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

DeepRT_Dataset_Folder = os.path.abspath(r'')
DeepRT_Pretrain_Folder = os.path.abspath(r'')

this_task = ''

DeepRT_WorkSpace_Folder = os.path.join(os.path.abspath(r''), this_task)


DeepRTParam = {
    'AA_List': AAList_20 + ['1', '2', '3', '4'],

    'PATH_PretrainModel': os.path.join(DeepRT_Pretrain_Folder, 'dia_all_epo20_dim24_conv8/dia_all_epo20_dim24_conv8_filled.pt'),  # This should be blank if use no pretrain model
    'Conv_Train': 8,

    'PATH_TrainSet': os.path.join(DeepRT_Dataset_Folder, ''),
    'PATH_TrainResult': os.path.join(DeepRT_WorkSpace_Folder, 'TrainResult.txt'),
    'PATH_SavePrefix': os.path.join(DeepRT_WorkSpace_Folder, 'Models'),
    'PATH_TestSet': os.path.join(DeepRT_Dataset_Folder, ''),
    'PATH_Pred_Output': os.path.join(DeepRT_WorkSpace_Folder, 'PredOutput.txt'),

    'PATH_Log': os.path.join(DeepRT_WorkSpace_Folder, 'TrainLog.txt'),

    'PATH_R1_Model': '',  # If ensembl, this three model need only pass the directory of models with different epoches
    'PATH_R2_Model': '',  # This should be blank if no ensembl
    'PATH_R3_Model': '',  # This should be blank if no ensembl

    'Conv_R1_Model': 8,
    'Conv_R2_Model': 10,
    'Conv_R3_Model': 12,

    'Min_RT': -80,
    'Max_RT': 176,

    'Seq_Col_Name': 'IntPep',
    'RT_Col_Name': 'iRT',

    'BATCH_SIZE': 16,
    'EPOCHS': 20,
    'LR': 0.01,
    'MinEpochToSave': 5,

    'Max_Len': 66,
    'Time_Scale': 1,
}


def write_deeprt_template(temp_path):
    with open(temp_path, 'w') as f:
        json.dump(DeepRTParam, f, indent=4)


if __name__ == '__main__':
    write_deeprt_template('DeepRT_Param_Template.json')
