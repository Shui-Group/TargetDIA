import argparse

from utils.spectronaut import SpectronautLibrary as SNLib

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description='''''')

    # library
    parser.add_argument('-l', '--library', metavar='Library path', type=str, required=True,
                        help='Library file path')
    # output path
    parser.add_argument('-o', '--output', metavar='Output path', type=str, required=True,
                        help='Output path of filtered library')
    # pep length
    parser.add_argument('-min', '--minlen', metavar='Min peptide length', default=None,
                        help='The minimum length of peptide. Input should be a integer or no -min argument')
    parser.add_argument('-max', '--maxlen', metavar='Max peptide length', default=None,
                        help='''The maximum length of peptide. Input should be a integer, 
or a float between 0 and 1 which indicates the percentage of the remained peptides,
or no -max argument''')

    # no loss peaks
    parser.add_argument('-p', '--peaks', metavar='Min number of noloss fragment', default=None,
                        help='''The minimum number of fragment with loss type as noloss. Input should be a integer, 
or no -max argument''')

    # charge
    parser.add_argument('-c', '--charge', metavar='Charge range', default=None,
                        help='''The charge state of precursor. Input should be integers with delimiter as ',', 
or no -max argument
Example: 1,2,3,4,5''')
    args = parser.parse_args()

    lib_path = args.library
    out_path = args.output
    min_len = args.minlen
    max_len = args.maxlen
    min_peaks = args.peaks
    charge = args.charge
    if min_len:
        min_len = int(min_len)
    if max_len:
        if max_len.isdigit():
            max_len = int(max_len)
        else:
            max_len = float(max_len)
    if min_peaks:
        min_peaks = int(min_peaks)
    if charge:
        charge = [int(_) for _ in charge.strip(',').split(',')]

    print(f'''Set params:
    library path: {lib_path}
    min peptide length: {min_len}
    max peptide length: {max_len}
    min noloss peaks: {min_peaks}
    charge state: {charge}
''')

    sn = SNLib()
    print('Loading library')
    sn.set_library(lib_path)
    print('Filtering library')
    sn.filter_library(min_len, max_len, min_peaks, charge)
    print('Saving filtered library')
    sn.write_current_library(out_path)
    print('Done')
