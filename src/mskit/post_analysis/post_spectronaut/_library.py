from .sn_constant import *
from .basic_operations import *

import re
import os
import pandas as pd
try:
    from sklearn.model_selection import train_test_split
except ModuleNotFoundError:
    pass

from mskit import rapid_kit


class SpectronautLibrary(object):
    """
    self._library_storage is used for the original library dataframe storage so that self._lib_df can be restored when it is filtered.
    """
    def __init__(self, spectronaut_version=12):
        self._spectronaut_version = spectronaut_version
        self._Mod = ModType(self._spectronaut_version)

        self._lib_path = None
        self._lib_df = None
        self._library_storage = None
        self._initial_lib_title = None

        self._prec = None
        self._modpep = None
        self._stripped_pep = None

    def set_library(self, lib):
        if isinstance(lib, pd.DataFrame):
            self._lib_df = lib
        else:
            if os.path.isfile(lib):
                self._lib_path = lib
                self._lib_df = pd.read_csv(self._lib_path, sep='\t')
            else:
                raise

        self._initial_lib_title = self._lib_df.columns
        self._library_storage = self._lib_df.copy()
        self.add_prec()

    def __len__(self):
        return len(self._lib_df)

    def backtrack_library(self):
        """
        replace lib_df with storage
        """
        self._lib_df = self._library_storage

    def store_current_library(self):
        self._library_storage = self._lib_df.copy()

    def switch_curr_store_lib(self):
        _ = self._lib_df.copy()
        self._lib_df = self._library_storage.copy()
        self._library_storage = _

    def add_prec(self):
        if 'Precursor' not in self._lib_df:
            self._lib_df['Precursor'] = get_lib_prec(self._lib_df)

    def remove_library_add_col(self):
        new_cols = [_ for _ in self._lib_df.columns if _ in self._initial_lib_title]
        self._lib_df = self._lib_df[new_cols]

    def return_library(self, remove_add_col=False):
        if remove_add_col:
            self.remove_library_add_col()
        return self._lib_df

    def write_current_library(self, output_path, remove_add_col=True):
        """
        write current lib to file
        """
        if remove_add_col:
            self.remove_library_add_col()
        self._lib_df.to_csv(output_path, index=False, sep='\t')

    def library_basic_info(self):
        """
        Count basic library infomation includes
            'protein_group', 'protein', 'modpep', 'prec', 'stripped_pep'
        :return:
        """
        protein_group = set(self._lib_df['ProteinGroups'])
        protein = set(sum([_.split(';') for _ in protein_group], []))
        self._modpep = set(self._lib_df['ModifiedPeptide'])
        self.add_prec()
        self._prec = set(self._lib_df['Precursor'])
        self._stripped_pep = set(self._lib_df['StrippedPeptide'])
        library_info_dict = dict((('ProteinGroup', len(protein_group)),
                                  ('SingleProtein', len(protein)),
                                  ('ModifiedPeptide', len(self._modpep)),
                                  ('Precursor', len(self._prec)),
                                  ('StrippedPeptide', len(self._stripped_pep)),
                                  ))
        charge = self._lib_df['PrecursorCharge'].tolist()
        return library_info_dict

    def strippep_length_distrib(self) -> pd.Series:
        """
        Count peptides length in the library
        :return: pandas.Series, which has all peptide length as index and corresponding peptide number as value. The index is sorted increasely.
        """
        stripped_pep = self._lib_df['StrippedPeptide'].drop_duplicates()
        stripped_pep_len = stripped_pep.str.len()
        len_ser = stripped_pep_len.value_counts()
        sorted_len_ser = len_ser.sort_index()
        return sorted_len_ser

    def strippep_miss_cleavage_distrib(self) -> pd.Series:
        """
        Count miss cleavage in the library
        :return: pandas.Series, which has miss cleavage number as index and corresponding peptide number as value. The index is sorted increasely.
        """
        stripped_pep = self._lib_df['StrippedPeptide'].drop_duplicates()
        mc_ser = stripped_pep.str.count('[KR].*?') - stripped_pep.str.count('.+[KR]$')
        mc_count = mc_ser.value_counts()
        sorted_mc_count = mc_count.sort_index()
        return sorted_mc_count

    def prec_charge_distrib(self) -> pd.Series:
        """
        Count charge in the library
        :return: pandas.Series, which has charge state as index and corresponding peptide number as value. The index is sorted increasely.
        """
        prec_nonredundant_df = self.get_nonredundant_prec_df(self._lib_df)
        prec_charge = prec_nonredundant_df['PrecursorCharge']
        charge_count = prec_charge.value_counts()
        sorted_charge_count = charge_count.sort_index()
        return sorted_charge_count

    def library_ms1_intensity_distrib(self) -> pd.Series:
        """
        MS1 response of each precursor
        :return:
        """
        prec_nonredundant_df = self.get_nonredundant_prec_df(self._lib_df)
        prec_ms1_response = prec_nonredundant_df['ReferenceRunMS1Response']
        return prec_ms1_response

    def library_oxidation_distrib(self):
        """
        :return: Two pd.Series, which are oxidation count and methionine count, respectively
        """
        stripped_pep = self._lib_df['ModifiedPeptide'].drop_duplicates()
        oxi_ser = stripped_pep.str.count(r'M\[Oxi.*?')
        oxi_count = oxi_ser.value_counts()
        sorted_oxi_count = oxi_count.sort_index()
        return sorted_oxi_count

    def library_length_charge_distrib(self) -> pd.Series:
        pass

    def filter_library(self, min_len=None, max_len=None, min_noloss_peaks: int = 6, charge=(1, 2, 3, 4, 5)):
        """
        For psm filter, to get get_one_prefix_result library with better quantity for training.
        This will filter the library with three conditions:
            1. stripped peptide length: from min length to 95% max length of whole library length distribution,
            which means the max length range is set as the length when cumulation of sorted pep length num achieves 95% of total pep num.
            The min length is usually set to 7 since the default set in spectronaut is 7.
            The max length can be calculated automatically with ##########################
            2. fragment loss type: 'noloss' type should equals or more than 6 as default
            3. precursor charge: this will be 1-5 in DIA
            Additional condition: 4. miss cleavage: 0-1, this is not so useful

        Either min_len or max_len is not None, the length filter will be done.
        :param min_len: The min length of stripped peptide
        :param max_len: The max length of stripped peptide. An int number will lock the max length to it,
        whill a float number will make the max length an expected number to cover certein percentage of total peptides.
        Example: max_len=26 means the max length is 26, and max_len=0.95 means the length covers 95% of the total peptides will be max length
        :param min_noloss_peaks:
        :param charge:
        """
        if isinstance(charge, int):
            charge = (charge, )

        if isinstance(max_len, float):
            len_distrib = self.strippep_length_distrib()
            if min_len:
                len_distrib = len_distrib[len_distrib.index >= min_len]
            accu_num = 0
            targeted_pep_num = sum(len_distrib) * max_len
            for _l, _n in len_distrib.items():
                accu_num += _n
                if accu_num >= targeted_pep_num:
                    max_len = _l
                    break

        def length_filter(x):
            if min_len:
                if len(x) < min_len:
                    return False
            if max_len:
                if len(x) > max_len:
                    return False
            return True

        length_filtered_df = self._lib_df[self._lib_df['StrippedPeptide'].apply(
            length_filter)] if any((min_len, max_len)) else self._lib_df.copy()  # filter_condition 1

        charge_filtered_df = length_filtered_df[
            length_filtered_df['PrecursorCharge'].astype(int).isin(charge)] if charge else length_filtered_df  # filter_condition 3

        if min_noloss_peaks:
            peak_count = charge_filtered_df.groupby('Precursor')['FragmentLossType'].transform(lambda x: len(x[x == 'noloss']))
            noloss_only_df = charge_filtered_df.loc[peak_count[peak_count >= min_noloss_peaks].index]
            self._lib_df = noloss_only_df
        else:
            self._lib_df = charge_filtered_df

    def library_keep_main_cols(self):
        """
        Remove some columns that are not necessary in the library
        """
        self._lib_df = self._lib_df[SNLibraryTitle.LibraryMainCol]

    def library_keep_main_cols_pgout(self):
        """
        Keep the same columns as library_keep_main_cols except 'ProteinGroups'
        """
        self._lib_df = self._lib_df[SNLibraryTitle.LibraryMainColPGOut]

    def get_prec_list(self):
        if self._prec:
            return list(self._prec)
        else:
            if 'Precursor' not in self._lib_df:
                self.add_prec()
            prec_list = self._lib_df['Precursor'].drop_duplicates().tolist()
            self._prec = prec_list
            return prec_list

    def get_modpep_list(self):
        if self._modpep:
            return list(self._modpep)
        else:
            modpep_list = self._lib_df['ModifiedPeptide'].drop_duplicates().tolist()
            self._modpep = modpep_list
            return modpep_list

    def extract_rt_data(self, pep_col='ModifiedPeptide', rt_col='iRT', return_type='dict'):
        """
        This will extract nonredundancy modpep and corresponding rt
        :return: dict like {'modpep': rt, ...}, or list like [(modpep, rt), ...]
        """
        return get_lib_rt_info(self._lib_df, pep_col, rt_col, return_type)

    def extract_fragment_data(self, norm=False):
        """
        :param norm False means no normalization and a number means the max number of intensity
        """
        frag_dict = get_lib_fragment_info(self._lib_df, norm=norm)
        return frag_dict

    def library_keep_target(self, target_protein_list):
        """
        Keep psms that contain target protein
        target protein may be consist of protein groups, which are separated by semicolons
        :return:
        """
        target_list = rapid_kit.process_list_or_file(target_protein_list)
        target_set = set(target_list)
        self._lib_df = self._lib_df[self._lib_df.apply(rapid_kit.protein_groups_match, col='ProteinGroups', args=(target_set, ), axis=1)]

    def library_remove_target(self, target_protein_list):
        """
        Remove psms that contain target protein
        target protein may be consist of protein groups, which are separated by semicolons
        196 ms ± 97.3 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
        :return:
        """
        target_list = rapid_kit.process_list_or_file(target_protein_list)
        target_set = set(target_list)
        self._lib_df = self._lib_df[~self._lib_df.apply(rapid_kit.protein_groups_match, col='ProteinGroups', args=(target_set, ), axis=1)]

    def library_remove_target_re(self, target_protein_list):
        """
        Remove psms that contain target protein
        target protein may be consist of protein groups, which are separated by semicolons
        1min 56s ± 199 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        Use library_remove_target instead
        """
        if os.path.isfile(target_protein_list):
            with open(target_protein_list, 'r') as target_handle:
                target_list = [_.strip('\n') for _ in target_handle.readlines()]
                while '' in target_list:
                    target_list.remove('')
        else:
            target_list = target_protein_list
        match_target = re.compile('|'.join(['(' + _ + ')' for _ in target_list]))
        self._lib_df = self._lib_df[~(self._lib_df['ProteinGroups'].str.contains(match_target))]

    def split_library(self, train_test_val=(0.8, 0.1, 0.1), focus_col='ModifiedPeptide'):
        split_content = self._lib_df[focus_col].drop_duplicates().tolist()
        train_size = train_test_val[0]
        test_size = train_test_val[1]
        if len(train_test_val) == 2:
            _train, _test = train_test_split(split_content, train_size=train_size, test_size=test_size)
            return self._lib_df[self._lib_df[focus_col].isin(_train)], self._lib_df[self._lib_df[focus_col].isin(_test)]
        if len(train_test_val) == 3:
            val_size = train_test_val[2]
            _train, _test_val = train_test_split(split_content, train_size=train_size, test_size=test_size + val_size)
            test_val_size = test_size + val_size
            test_size, val_size = test_size / test_val_size, val_size / test_val_size
            _test, _val = train_test_split(_test_val, train_size=test_size, test_size=val_size)
            return self._lib_df[self._lib_df[focus_col].isin(_train)], self._lib_df[self._lib_df[focus_col].isin(_test)], self._lib_df[self._lib_df[focus_col].isin(_val)]

    @staticmethod
    def get_nonredundant_prec_df(lib_df: pd.DataFrame) -> pd.DataFrame:
        return lib_df.drop_duplicates(('PrecursorCharge', 'ModifiedPeptide'))

    @staticmethod
    def get_nonredundant_precursor_index(lib_df: pd.DataFrame, position=1) -> pd.Index:
        """
        Consider get_one_prefix_result psm as get_one_prefix_result block.
        :param lib_df:
        :param position: 1 means block start index, -1 means block end index
        :return: pd.Index, which is the first line index or the end line index of each psm block
        """
        precursor_block_index = lib_df.loc[(lib_df['PrecursorCharge'].astype(str) +
                                            lib_df['ModifiedPeptide']) !=
                                           (lib_df['PrecursorCharge'].astype(str).shift(position) +
                                            lib_df['ModifiedPeptide'].shift(position))].index
        return precursor_block_index

    @staticmethod
    def get_psm_block_index(lib_df: pd.DataFrame) -> list:
        """
        Consider get_one_prefix_result psm as get_one_prefix_result block. This can return get_one_prefix_result list contains index tuples for library dataframe slice
        If the first precursor occupy the first 9 lines (which means line 0-8), the tuple will be (0, 9), then library dataframe can be sliced directly.
        :return: list of tuples, like [(0, 9), (9, 13), (13, 19), ...]
        """
        precursor_block_start_index = SpectronautLibrary.get_nonredundant_precursor_index(lib_df, position=1)
        precursor_block_end_index = SpectronautLibrary.get_nonredundant_precursor_index(lib_df, position=-1)
        block_index_list = list(zip(precursor_block_start_index, precursor_block_end_index))
        return block_index_list
