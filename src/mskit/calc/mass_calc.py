from mskit.constants.mass import Mass
from mskit.rapid_kit import split_prec, split_fragment_name


def calc_prec_mz(pep: str, charge: int = 2, mod=None) -> float:
    """
    Example:
    pep = 'LGRPSLSSEVGVIICDISNPASLDEMAK'
    charge = 3
    mod = '15,Carbamidomethyl;26,Oxidation;'
    -> 992.1668502694666

    :param pep: peptide that is modified with num, e.g. ACDM1M, where 1 equals to M[Oxidation]
    :param charge: precursor charge
    :param mod: str like 1,Carbamidomethyl;3,Oxidation; or list like [(1, 'Carbamidomethyl'), (3, 'Oxidation')]
    :return: Precursor m/z
    """
    if '.' in pep:
        pep, charge = split_prec(pep)

    pep_mass = 0.
    for _aa in pep.replace('_', ''):
        pep_mass += Mass.AAMass[_aa]
    pep_mass += Mass.CompoundMass['H2O']
    pep_mass += Mass.ProtonMass * charge

    if mod:
        if isinstance(mod, str):
            mod = [_.split(',') for _ in mod.strip(';').split(';')]
        for _each_mod in list(zip(*mod))[1]:
            pep_mass += Mass.ModificationMass[_each_mod] if _each_mod != 'Carbamidomethyl' else 0.
    return pep_mass / charge


def calc_fragment_mz(pep, frag_type, frag_num, frag_charge, mod=None) -> float:
    """
    Example:
    pep = 'LGRPSLSSEVGVIICDISNPASLDEMAK'
    frag_type = 'y'
    frag_num = 10
    frag_charge = 1
    mod = '15,Carbamidomethyl;26,Oxidation;'
    -> 1091.5037505084001

    :param pep: peptide sequence
    :param frag_type: support b and y ion
    :param frag_num:
    :param frag_charge:
    :param mod:
    :return: Fragment m/z
    """
    frag_num = int(frag_num)
    frag_charge = int(frag_charge)
    frag_mass = Mass.ProtonMass * frag_charge

    if mod:
        if isinstance(mod, str):
            mod = [_.split(',') for _ in mod.strip(';').split(';')]
            mod = [(int(_[0]), _[1]) for _ in mod]
        mod_dict = dict(mod)
    else:
        mod_dict = dict()

    if frag_type == 'b':
        for i in range(frag_num):
            frag_mass += Mass.AAMass[pep[i]]
            if i + 1 in mod_dict:
                frag_mass += Mass.ModificationMass[mod_dict[i + 1]]
        if 0 in mod_dict:
            frag_mass += Mass.ModificationMass[mod_dict[0]]

    elif frag_type == 'y':
        frag_mass += Mass.CompoundMass['H2O']
        pep_len = len(pep)
        for i in range(pep_len - 1, pep_len - 1 - frag_num, -1):
            frag_mass += Mass.AAMass[pep[i]]
            if i + 1 in mod_dict:
                frag_mass += Mass.ModificationMass[mod_dict[i + 1]]
        if frag_num == pep_len:
            frag_mass += Mass.ModificationMass[mod_dict[0]]
    else:
        raise NameError('Only b and y ion are supported')
    return frag_mass / frag_charge


def get_fragment_mz_dict(pep, fragments, mod=None):
    """
    :param pep:
    :param fragments:
    :param mod:
    :return:
    """
    mz_dict = dict()
    for each_fragment in fragments:
        frag_type, frag_num, frag_charge = split_fragment_name(each_fragment)
        mz_dict[each_fragment] = calc_fragment_mz(
            pep, frag_type, frag_num, frag_charge, mod)
    return mz_dict
