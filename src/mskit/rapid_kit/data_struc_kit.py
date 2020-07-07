
def sum_set_in_list(list_with_sets, return_type='set'):
    summed_list = sum([list(each_set) for each_set in list_with_sets], [])
    if return_type == 'set':
        return set(summed_list)
    elif return_type == 'list':
        return summed_list
    else:
        print('Not supported return type when sum set list')


def drop_list_duplicates(initial_list):
    unique_list = list(set(initial_list))
    unique_list = sorted(unique_list, key=initial_list.index)
    return unique_list


def get_coincide_data(dict_1, dict_2):
    shared_keys = list(set(dict_1.keys()) & set(dict_2.keys()))
    value_list_1 = [dict_1[_] for _ in shared_keys]
    value_list_2 = [dict_2[_] for _ in shared_keys]
    return shared_keys, value_list_1, value_list_2


def str_mod_to_list(mod):
    mod_list = [each_mod.split(',') for each_mod in mod.strip(';').split(';')]
    mod_list = [(int(_[0]), _[1]) for _ in mod_list]
    return mod_list
