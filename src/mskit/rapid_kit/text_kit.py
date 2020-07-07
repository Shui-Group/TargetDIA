from .data_struc_kit import str_mod_to_list

import re
import os


def get_title_dict(title_content: str) -> dict:
    title_dict = dict([(__, _) for _, __ in enumerate(
        title_content.strip('\n').split('\t'))])
    return title_dict


def semicolon_combination(s1, s2, keep_order=False):
    """
    This will combine two strings with semicolons and drop duplication
    Example: s1='Q1;Q2', s2='Q2;Q3' -> 'Q1;Q2;Q3'
    Note that the order may change if keep_order=False
    """
    s_list = map(lambda _: _.strip(';').split(';'), (s1, s2))
    flatten_s = sum(s_list, [])
    unique_s = list(set(flatten_s))
    if keep_order:
        unique_s = sorted(unique_s, key=flatten_s.index)
    return ';'.join(unique_s)


def extract_bracket(str_with_bracket):
    bracket_start = [left_bracket.start()
                     for left_bracket in re.finditer(r'\[', str_with_bracket)]
    bracket_end = [right_bracket.start()
                   for right_bracket in re.finditer(']', str_with_bracket)]
    return bracket_start, bracket_end


def split_fragment_name(fragment_name):
    frag_type, frag_num, frag_charge, frag_loss = re.findall(
        '([abcxyz])(\\d+)\\+(\\d+)-?(.*)', fragment_name)[0]
    return frag_type, int(frag_num), int(frag_charge), frag_loss


def split_prec(prec: str, keep_underline=False):
    modpep, charge = prec.split('.')
    if not keep_underline:
        modpep = modpep.replace('_', '')
    return modpep, int(charge)


def assemble_prec(modpep, charge):
    if not modpep.startswith('_'):
        modpep = f'_{modpep}_'
    return f'{modpep}.{charge}'


def split_mod(modpep, mod_ident='bracket'):
    if mod_ident == 'bracket':
        mod_ident = ('[', ']')
    elif mod_ident == 'parenthesis':
        mod_ident = ('(', ')')
    else:
        pass
    re_find_pattern = '(\\{}.+?\\{})'.format(*mod_ident)
    re_sub_pattern = '\\{}.*?\\{}'.format(*mod_ident)
    modpep = modpep.replace('_', '')
    mod_len = 0
    mod = ''
    for _ in re.finditer(re_find_pattern, modpep):
        _start, _end = _.span()
        mod += '{},{};'.format(_start - mod_len, _.group().strip(''.join(mod_ident)))
        mod_len += _end - _start
    stripped_pep = re.sub(re_sub_pattern, '', modpep)
    return stripped_pep, mod


def add_mod(pep, mod, mod_processor):
    """
    mod_process is the ModOperation class
    """
    if mod:
        if isinstance(mod, str):
            mod = str_mod_to_list(mod)
        mod = sorted(mod, key=lambda x: x[0])
        mod_pep_list = []
        prev_site_num = 0
        for mod_site, mod_name in mod:
            mod_pep_list.append(pep[prev_site_num: mod_site])
            if mod_site != 0:
                mod_aa = mod_pep_list[-1][-1]
            else:
                mod_aa = pep[0]
            mod_pep_list.append(mod_processor(mod=mod_name, aa=mod_aa))
            prev_site_num = mod_site
        mod_pep_list.append(pep[prev_site_num:])
        mod_pep = ''.join(mod_pep_list)
    else:
        mod_pep = pep
    mod_pep = f'_{mod_pep}_'
    return mod_pep


def read_one_col_file(file, header=None):
    with open(file, 'r') as f:
        one_col_list = [_.strip('\n') for _ in f.readlines()]
        one_col_list = one_col_list[1:] if header else one_col_list
        while '' in one_col_list:
            one_col_list.remove('')
    return one_col_list


def process_list_or_file(x):
    if isinstance(x, list) or isinstance(x, set):
        target_list = x
    else:
        if os.path.isfile(x):
            target_list = read_one_col_file(x)
        else:
            raise
    return target_list


def fasta_title(title: str, title_type='uniprot'):
    title = title.lstrip('>')

    if '|' in title:
        ident = title.split('|')[1]
    else:
        ident = title.split(' ')[0]
    return ident
