import argparse

from mskit import ms_prediction
from mskit.post_analysis import spectronaut as post_sn
from mskit.seq_process import ModOperation


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description='''
-ion and -rt is a pair of argument for combining predicted ion intensity and rt
-dia and -vir is a pair of argument for combining initial DIA library and virtual library
The two pairs are conflicting''')

    # Virtual library
    parser.add_argument('-ion', '--iondata', metavar='Predicted ion intensity data file', type=str, default=None,
                        help='Prediction from pDeep')
    parser.add_argument('-rt', '--rt', metavar='Predicted RT data file', type=str, default=None,
                        help='Prediction from DeepRT')

    # Hybrid library
    parser.add_argument('-dia', '--dialibrary', metavar='DIA library path', type=str, default=None,
                        help='Path of initial DIA library')
    parser.add_argument('-vir', '--virtuallibrary', metavar='Virtual library path', type=str, default=None,
                        help='Path of virtual library')

    # output path
    parser.add_argument('-o', '--output', metavar='Output path', type=str, required=True,
                        help='Output path of virtual or hybrid library')

    args = parser.parse_args()

    ion_file = args.iondata
    rt_file = args.rt
    dia_lib_path = args.dialibrary
    vir_lib_path = args.virtuallibrary
    out_path = args.output

    print(f'''Set params:
    ion file: {ion_file}
    rt file: {rt_file}
    dia file: {dia_lib_path}
    virtual file: {vir_lib_path}
    output path: {out_path}''')
    print()

    if ion_file and rt_file:
        mod_processor = ModOperation()
        print(f'Loading ion data from {ion_file}')
        ion_data = ms_prediction.pdeep.read_pdeep_pred(ion_file, mod_processor)
        print(f'Loading RT data from {rt_file}')
        rt_data = ms_prediction.deeprt.read_deeprt_pred(rt_file)
        print(f'Writing library to {out_path}')
        post_sn.write_lib(out_path, ion_data, rt_data)

    elif dia_lib_path and vir_lib_path:
        print(f'Loading DIA library from {dia_lib_path}')
        dia_lib = post_sn.SpectronautLibrary()
        dia_lib.set_library(dia_lib_path)
        dia_lib.library_keep_main_cols_pgout()
        dia_lib_df = dia_lib.return_library()
        print('Merging library')
        hybrid_lib_df = post_sn.merge_lib(dia_lib_df, vir_lib_path, drop_col=None)
        print(f'Writing library to {out_path}')
        hybrid_lib = post_sn.SpectronautLibrary()
        hybrid_lib.set_library(hybrid_lib_df)
        hybrid_lib.write_current_library(out_path)

    else:
        raise argparse.ArgumentError

    print('Done')
