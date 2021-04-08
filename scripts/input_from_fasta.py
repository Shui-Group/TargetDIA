import argparse
import os

from utils import ms_prediction
from utils import seq_process

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description='''''')

    # library
    parser.add_argument('-f', '--fasta', metavar='Fasta path', type=str, required=True, default=None,
                        help='Fasta file path')
    # output dir
    parser.add_argument('-o', '--output', metavar='Output dir', type=str, required=True, default=None,
                        help='Output directory of pDeep and DeepRT input files')

    # digestion params
    parser.add_argument('-min', '--minlen', metavar='Output dir', type=int, required=True,
                        help='Output directory of pDeep and DeepRT input files')
    parser.add_argument('-max', '--maxlen', metavar='Output dir', type=int, required=True,
                        help='Output directory of pDeep and DeepRT input files')
    parser.add_argument('-mc', '--misscleavage', metavar='Output dir', type=int, required=True,
                        help='Output directory of pDeep and DeepRT input files')
    parser.add_argument('-c', '--charge', metavar='Output dir', type=str, required=True,
                        help='Output directory of pDeep and DeepRT input files')
    parser.add_argument('-oxi', '--oxidation', metavar='Output dir', type=int, choices=[0, 1], default=0,
                        help='Output directory of pDeep and DeepRT input files')

    args = parser.parse_args()

    fasta_path = args.fasta
    out_dir = args.output

    min_len = args.minlen
    max_len = args.maxlen
    mc = args.misscleavage
    mc = tuple(range(int(mc) + 1))
    charge = args.charge
    charge = [int(c) for c in charge.split(',')]
    oxi = args.oxidation

    print(f'''Set params:
    fasta path: {fasta_path}
    output dir: {out_dir}
    min length: {min_len}
    max length: {max_len}
    miss cleavage: {mc}
    charge: {charge}
    oxidation: {oxi}
'''
          )

    if not os.path.exists(out_dir):
        print(f'Creating output directory {out_dir}')
        os.makedirs(out_dir)

    print(f'Loading fasta file from {fasta_path}')
    fasta = seq_process.FastaParser(fasta_path, title_type='uniprot')
    print(f'''Digesting protein sequence with params:
    length: {min_len}-{max_len}
    miss cleavage: {mc}
    ''')
    peps = fasta.get_total_seqlist(miss_cleavage=mc, min_len=min_len, max_len=max_len)
    print(f'Adding modification: {oxi}')
    mod_processor = seq_process.ModOperation()
    modpeps = seq_process.batch_add_target_mod(peps, mod_type={'Carb': -1, 'ox': int(oxi)}, mod_processor=mod_processor)

    fasta_name = os.path.splitext(os.path.basename(fasta_path))[0]
    deeprt_input_path = os.path.join(out_dir, f'Input-DeepRT-{fasta_name}.txt')
    print(f'Generating DeepRT input: {deeprt_input_path}')
    ms_prediction.deeprt.deeprt_input(deeprt_input_path, modpeps)
    print(f'Adding charge {charge}')
    precs = seq_process.batch_add_target_charge(modpeps, charge)
    pdeep_input_path = os.path.join(out_dir, f'Input-pDeep-{fasta_name}.txt')
    print(f'Generating pDeep input: {pdeep_input_path}')
    ms_prediction.pdeep.pdeep_input(pdeep_input_path, precs)

    print('Done')
