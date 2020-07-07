import pandas as pd

from mskit import rapid_kit


# TODO: Re-write this


def get_one_prefix_result(result_df, prefix, suffixes):
    return [set(result_df[f'{prefix}-{each_suffix}'].dropna()) for each_suffix in suffixes]


def get_search_result(result_file, sheet_names, prefixes=('Protein', 'Precursor'), suffixes=('R1', 'R2', 'R3')):
    if isinstance(sheet_names, str):
        sheet_names = [sheet_names]
    with pd.ExcelFile(result_file) as f:
        result_list = []
        for sheet in sheet_names:
            df = f.parse(sheet_name=sheet)
            store_dict = dict()
            for each_prefix in prefixes:
                store_dict[each_prefix] = get_one_prefix_result(df, each_prefix, suffixes)
                store_dict[f'{each_prefix}-Total'] = rapid_kit.data_struc_kit.sum_set_in_list(
                    store_dict[each_prefix])
            if 'Precursor' in prefixes:
                store_dict['Peptide'] = [set([_.split(
                    '.')[0] for _ in each_suff_data]) for each_suff_data in store_dict[each_prefix]]
                store_dict['Peptide-Total'] = rapid_kit.data_struc_kit.sum_set_in_list(
                    store_dict['Peptide'])
            result_list.append(store_dict)
    return result_list


def select_target_df(original_df, region_identifier):
    region_df = rapid_kit.extract_df_with_col_ident(original_df, region_identifier, focus_col='R.Instrument (parsed from filename)')
    return region_df


def read_search_result_intensity(result_file):
    result_df = pd.read_csv(vlib_search_result, sep='\t', low_memory=False)
    region_intensity_list = []
    for _ in ['region3', 'region5', 'region6']:
        region_intensity_dict = dict()
        each_region_df = select_target_df(result_df, _)
        for each_prec in each_region_df['EG.PrecursorId'].drop_duplicates():
            each_prec_df = each_region_df[each_region_df['EG.PrecursorId'] == each_prec]
            noloss_prec_df = each_prec_df[each_prec_df['F.FrgLossType'] == 'noloss']
            fragment_list = (noloss_prec_df['F.FrgType'] +
                             noloss_prec_df['F.FrgNum'].astype(str) + '+' +
                             noloss_prec_df['F.Charge'].astype(str)).tolist()
            fragment_intensity = noloss_prec_df['F.MeasuredRelativeIntensity'].tolist(
            )
            region_intensity_dict[each_prec] = dict(
                zip(fragment_list, fragment_intensity))
        region_intensity_list.append(region_intensity_dict)
    return region_intensity_list
