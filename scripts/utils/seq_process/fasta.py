import os

from .ted import ted
from .. import rapid_kit


class FastaParser(object):
    def __init__(self, fasta_path, title_type='uniprot'):
        """
        :param fasta_path:
        :param title_type: 'uniprot' will get the second string for title split by '|', and others will be the first string split by get_one_prefix_result blank,
        while maybe other formats of fasta title is needed later
        """
        try:
            self._fasta_path = os.path.abspath(fasta_path)
        except NameError:
            print('Incorrect fasta file path')
            raise
        self._title_type = title_type

        self._fasta_content = None  # The whole text of the fasta file
        self._protein_info = dict()  # The description information of each protein in the fasta file
        self._protein_to_seq = dict()  # The whole sequence of each protein (No digestion)
        self._seq_to_protein = dict()  # Digested peptide to protein. The protein may be str if one else list.
        self._seq_list = []  # Digested peptides of all protein sequence in the fasta file

    def __call__(self, *args, **kwargs):
        pass

    def __iter__(self):
        return iter(self.get_total_seqlist())

    def get_fasta_content(self):
        """
        Get the whole content of the input fasta file
        """
        if not self._fasta_content:
            with open(self._fasta_path, 'r') as fasta_handle:
                self._fasta_content = fasta_handle.read()
        return self._fasta_content

    def one_protein_generator(self):
        """
        Generate title and sequence of each protein in fasta file
        """
        seq_title = ''
        seq_list = []
        with open(self._fasta_path, 'r') as fasta_handle:
            for _line in fasta_handle:
                if not _line:
                    print('Blank line existed in fasta file')
                    continue
                if _line.startswith('>'):
                    if seq_title and seq_list:
                        yield seq_title, ''.join(seq_list)
                    seq_title = _line.strip('\n')
                    seq_list = []
                else:
                    seq_list.append(_line.strip('\n'))
            if seq_title and seq_list:
                yield seq_title, ''.join(seq_list)

    def protein2seq(self, protein_info=False):
        if not self._protein_to_seq:
            for _title, _seq in self.one_protein_generator():
                protein_ident = rapid_kit.fasta_title(_title, self._title_type)
                self._protein_to_seq[protein_ident] = _seq
                if protein_info:
                    self._protein_info[protein_ident] = _title
        return self._protein_to_seq

    def seq2protein(
            self,
            miss_cleavage=(0, 1, 2),
            min_len=7,
            max_len=33) -> dict:

        if not self._seq_to_protein:
            if not self._protein_to_seq:
                self.protein2seq()

            for protein_acc, seq in self._protein_to_seq.items():
                compliant_seq = ted(
                    seq,
                    miss_cleavage=miss_cleavage,
                    min_len=min_len,
                    max_len=max_len)
                for _each_seq in compliant_seq:
                    self._seq_list.append(_each_seq)
                    if _each_seq not in self._seq_to_protein:
                        self._seq_to_protein[_each_seq] = protein_acc
                    else:
                        if isinstance(self._seq_to_protein[_each_seq], str):
                            self._seq_to_protein[_each_seq] = [
                                self._seq_to_protein[_each_seq], protein_acc]
                        elif isinstance(self._seq_to_protein[_each_seq], list):
                            self._seq_to_protein[_each_seq].append(
                                protein_acc)
        return self._seq_to_protein

    def get_total_seqlist(
            self,
            miss_cleavage=(0, 1, 2),
            min_len=7,
            max_len=33):

        if not self._seq_list:
            self.seq2protein(miss_cleavage=miss_cleavage,
                             min_len=min_len,
                             max_len=max_len)
        self._seq_list = rapid_kit.drop_list_duplicates(self._seq_list)
        return self._seq_list
