import re


def ted(seq, miss_cleavage=(0, 1, 2), min_len=7, max_len=33, enzyme='trypsin', return_type='seq'):
    """
    Theoretical Enzyme Disgestion -> ted
    :param seq:
    :param miss_cleavage: this can be int or tuple, while it will be converted into tuple when use
    :param min_len:
    :param max_len:
    :param enzyme: only trypsin is supported now
    :param return_type: 'seq' or 'site_seq'
    :return:
    """
    if isinstance(miss_cleavage, tuple):
        pass
    elif isinstance(miss_cleavage, int):
        miss_cleavage = (miss_cleavage, )
    else:
        try:
            miss_cleavage = (int(miss_cleavage), )
        except TypeError:
            raise TypeError('miss cleavage shoule be int or tuple of int')

    seq = seq.replace('\n', '').replace(' ', '')

    split_seq_list = [(_.start(), _.group())
                      for _ in re.finditer('.*?[KR]|.+', seq)]

    compliant_seq = []
    for i in range(len(split_seq_list)):
        for mc in miss_cleavage:
            one_seq = ''.join([_[1] for _ in split_seq_list[i: i + mc + 1]])
            if min_len <= len(one_seq) <= max_len:
                if return_type == 'seq':
                    compliant_seq.append(one_seq)
                elif return_type == 'site_seq':
                    compliant_seq.append((split_seq_list[i][0], one_seq))
    return compliant_seq
