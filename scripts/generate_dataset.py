import argparse
import os

import utils.spectronaut as post_sn
from utils.ms_prediction import *
from utils.spectronaut import SpectronautLibrary as SNLib

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description='''''')

    # library
    parser.add_argument('-l', '--library', metavar='Library path', type=str, required=True,
                        help='Library file path')
    # output dir
    parser.add_argument('-o', '--output', metavar='Output dir', type=str, required=True,
                        help='Output directory of library used for training and test, and dataset of training and test')

    # Target list file
    parser.add_argument('-t', '--target', metavar='The target list file for library splitting', default=None,
                        help='''To split a library by target protein family members.
The input is a pure text file with targeted protein accessions in one column (No title).
When this argument is defined, -r or --ratio will not be used''')

    # ratio
    parser.add_argument('-r', '--ratio', metavar='Split ratio', default=None,
                        help='''Split the library to train and test set with a ratio, such as 9:1 or 8:2''')
    args = parser.parse_args()

    lib_path = args.library
    out_dir = args.output
    target_file = args.target
    ratio = args.ratio

    print(f'''Set params:
    library path: {lib_path}
    output dir: {out_dir}
    target file: {target_file}
    split ratio: {ratio}
''')

    if not os.path.exists(out_dir):
        print('Creating output directory')
        os.makedirs(out_dir)

    sn = SNLib()
    print('Loading library')
    sn.set_library(lib_path)

    if target_file:
        # Test
        # Library
        print(f'Splitting library with target proteins in {target_file}')
        sn.library_keep_target(target_file)
        test_lib_path = os.path.join(out_dir, 'TestLibrary.txt')
        print(f'Storing target library in {test_lib_path}')
        sn.write_current_library(test_lib_path)

        # pDeep
        print('Extracting test precursors for intensity model')
        test_prec = sn.get_prec_list()
        inten_test_path = os.path.join(out_dir, 'TestInput-FragmentIntensity.txt')
        print(f'Storing test input of fragment intensity in {inten_test_path}')
        pdeep.pdeep_input(inten_test_path, test_prec)
        # DeepRT
        print('Extracting test modified peptides for RT model')
        test_modpep = sn.get_modpep_list()
        rt_test_path = os.path.join(out_dir, 'TestInput-RT.txt')
        print(f'Storing test input of RT in {rt_test_path}')
        deeprt.deeprt_input(rt_test_path, test_modpep)

        # Train
        # Library
        sn.backtrack_library()
        print(f'Splitting library with non-target proteins in {target_file}')
        sn.library_remove_target(target_file)
        train_lib_path = os.path.join(out_dir, 'TrainLibrary.txt')
        print(f'Storing non-target library in {train_lib_path}')
        sn.write_current_library(train_lib_path)

        # pDeep
        print('Extracting ion intensity for intensity model training')
        inten_dict = sn.extract_fragment_data()
        inten_train_path = os.path.join(out_dir, 'Trainset-FragmentIntensity.txt')
        print(f'Storing trainset of fragment intensity in {inten_train_path}')
        pdeep.pdeep_trainset(inten_train_path, inten_dict)
        # DeepRT
        print('Extracting RT data for RT model training')
        modpep_rt_list = sn.extract_rt_data(return_type='list')
        rt_train_path = os.path.join(out_dir, 'Trainset-RT.txt')
        print(f'Storing trainset of RT in {rt_train_path}')
        deeprt.deeprt_trainset(rt_train_path, modpep_rt_list)

    else:
        if not ratio:
            print('Argument must contain one of -t or -r. See details with -h')
        else:
            ratio = [float(_) for _ in ratio.replace(' ', '').strip(':').split(':')]
            sum_ratio = sum(ratio)
            ratio = [_ / sum_ratio for _ in ratio]
            print(f'Splitting library with ratio {ratio}')
            pdeep_train_lib, pdeep_test_lib = sn.split_library(ratio, focus_col='Precursor')
            deeprt_train_lib, deeprt_test_lib = sn.split_library(ratio, focus_col='ModifiedPeptide')
            print(f'Storing library in {out_dir}')
            pdeep_train_lib.to_csv(os.path.join(out_dir, 'TrainLibrary-pDeep.txt'), index=False, sep='\t')
            pdeep_test_lib.to_csv(os.path.join(out_dir, 'TestLibrary-pDeep.txt'), index=False, sep='\t')
            deeprt_train_lib.to_csv(os.path.join(out_dir, 'TrainLibrary-DeepRT.txt'), index=False, sep='\t')
            deeprt_test_lib.to_csv(os.path.join(out_dir, 'TestLibrary-DeepRT.txt'), index=False, sep='\t')
            print('')

            # pDeep train
            print('Extracting ion intensity for intensity model training')
            inten_dict = post_sn.get_lib_fragment_info(pdeep_train_lib)
            inten_train_path = os.path.join(out_dir, 'Trainset-pDeep.txt')
            print(f'Storing trainset of fragment intensity in {inten_train_path}')
            pdeep.pdeep_trainset(inten_train_path, inten_dict)

            # pDeep test
            print('Extracting test precursors for intensity model')
            test_prec = post_sn.get_lib_prec(pdeep_test_lib)
            inten_test_path = os.path.join(out_dir, 'TestInput-pDeep.txt')
            print(f'Storing test input of fragment intensity in {inten_test_path}')
            pdeep.pdeep_input(inten_test_path, set(test_prec))

            # DeepRT train
            print('Extracting RT data for RT model training')
            modpep_rt_list = post_sn.get_lib_rt_info(deeprt_train_lib, return_type='list')
            rt_train_path = os.path.join(out_dir, 'Trainset-DeepRT.txt')
            print(f'Storing trainset of RT in {rt_train_path}')
            deeprt.deeprt_trainset(rt_train_path, modpep_rt_list)

            # DeepRT test
            print('Extracting test modified peptides for RT model')
            test_modpep = deeprt_test_lib['ModifiedPeptide'].drop_duplicates().tolist()
            rt_test_path = os.path.join(out_dir, 'TestInput-DeepRT.txt')
            print(f'Storing test input of RT in {rt_test_path}')
            deeprt.deeprt_input(rt_test_path, test_modpep)

    print('Done')
