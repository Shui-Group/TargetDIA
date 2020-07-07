import json
import os


pDeep2_Dataset_Folder = os.path.abspath(r'')
pDeep2_Pretrain_Folder = os.path.abspath(r'')

this_task = ''

pDeep2_WorkSpace_Folder = os.path.join(os.path.abspath(r''), this_task)


pDeep2Param = {
    'FixMod': ['Carbamidomethyl[C]', ],
    'VarMod': [
        'Oxidation[M]',
    ],
    'IonType': ['b{}', 'y{}', 'b{}-ModLoss', 'y{}-ModLoss'],

    'MinVarMod': 0,
    'MaxVarMod': 5,

    'NCE': 0.30,
    'Instrument': 'QE',

    'Epochs': 50,

    'TrainsetFolder': os.path.join(pDeep2_Dataset_Folder, ''),
    'ModelOutFolder': os.path.join(pDeep2_WorkSpace_Folder, 'Models'),
    'ModelOutName': this_task,  # Only model name without suffix
    'PretrainedModelPath': os.path.join(pDeep2_Pretrain_Folder, 'pretrain-180921-modloss'),  # Blank if on pretrain model  # With no suffix
    'TensorboardFolder': pDeep2_WorkSpace_Folder,

    'ModelForPredFolder': os.path.join(pDeep2_WorkSpace_Folder, this_task),
    'ModelForPredName': this_task,  # With no suffix
    'PredInputPath': os.path.join(pDeep2_Dataset_Folder, ''),
    'PredOutPath': os.path.join(pDeep2_WorkSpace_Folder, 'PredOutput.txt'),

}


def write_pdeep2_template(temp_path):
    with open(temp_path, 'w') as f:
        json.dump(pDeep2Param, f, indent=4)


if __name__ == '__main__':
    write_pdeep2_template('pDeep2_Param_Template.json')
