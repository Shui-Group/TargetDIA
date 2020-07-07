
class BasicModification:
    # Pre-defined mods, where key means query mod and value is used for assembling queried mod
    StandardMods = {
        'Carbamidomethyl': 'Carbamidomethyl',
        'Oxidation': 'Oxidation',
        'Phospho': 'Phospho',
        'Acetyl': 'Acetyl'
    }

    # To extend the query space. Each mod has its alias and itself for quering
    __ModAliasList = {
        'Carbamidomethyl': ['Carbamidomethyl', 'Carbamid', 'Carb', 'Carbamidomethyl[C]'],
        'Oxidation': ['Oxidation', 'Oxi', 'Ox', 'Oxidation[M]'],
        'Phospho': ['Phospholation', 'Phospho', 'Phos', ],
        'Acetyl': ['_[Acetyl (Protein N-term)]'],
    }
    ModAliasDict = {}
    for standard, aliases in __ModAliasList.items():
        for alias in aliases:
            ModAliasDict[alias] = standard
    for alias in list(ModAliasDict.keys()):
        ModAliasDict[alias.upper()] = ModAliasDict[alias]
        ModAliasDict[alias.lower()] = ModAliasDict[alias]

    # Mod rule. This defines the method for mod assembly
    StandardModRule = r'[{mod} ({aa})]'
    ModRuleDict = {'standard': StandardModRule,
                   }

    ModAA = {
        'Carbamidomethyl': ['C'],
        'Oxidation': ['M'],
        'Phospho': ['P', 'T', 'S'],
    }
