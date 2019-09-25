import numpy as np
from itertools import groupby

def format_xyz(filename, precision=4):
    with open(filename) as fin:
        natoms = int(fin.readline())
        title = fin.readline()[:-1]
        all_coords = np.zeros([natoms, 3], dtype="float64")
        atomtypes = []
        for x in all_coords:
            line = fin.readline().split()
            atomtypes.append(line[0])
            x[:] = list(map(float, line[1:4]))
    atoms = []
    for atomtype, atom_coords in zip(atomtypes, all_coords):
        atoms.append(atomtype + ' ' +  ' '.join("{:.{}f}".format(
            atom_coord, precision) for atom_coord in atom_coords))

    return '; '.join(atoms)


def get_human_readable_reaction(indexes, mols):
    reaction = str()

    for i, (ind, mol) in enumerate(zip(indexes, mols)):
        if i != 0:
            if indexes[i] * indexes[i - 1] < 0:
                reaction += ' => '
            else:
                reaction += ' + '

        if abs(ind) != 1:
            reaction += str(ind)

        reaction += mol
        
    return reaction


def get_charges_multiplicities(filename):  
    with open(filename) as f:
        charges = {}
        multiplicities = {}
        for line in f:
                li = line.strip()
                data = li.split(' ')
                system, charge, multiplicity = data[0], int(data[1]), int(data[2]) - 1
                charges[system] = charge
                multiplicities[system] = multiplicity

    return charges, multiplicities


def tmer2_gmtkn_parser(dataset_directory):
    systems, stoichiometry, reference_value = [], [], []
    charges_dict, multiplicities_dict = get_charges_multiplicities(
        dataset_directory + "/CHARGE_MULTIPLICITY.txt")
    
    with open(dataset_directory + "/.res") as f:
        lines = (line for line in f if line)
        for line in lines:
            li=line.strip()
            if li and not li.startswith("#") and not li.startswith("w="):
                data = [list(g) for k, g in groupby(
                    line.rstrip().split()[1:-2], lambda x: x == "x") if not k]
                systems.append(data[0])
                stoichiometry.append(list(map(int, data[1])))
                reference_value.append(float(line.rstrip().split(' ')[-1]))
                
    reactions = []

    for indexes, mols in zip(stoichiometry, systems):
        reactions.append(get_human_readable_reaction(indexes, mols))
                
    charges = [[charges_dict[x] for x in sy] for sy in systems]
    multiplicities = [[multiplicities_dict[x] for x in sy] for sy in systems]
    systems_adapted = [[dataset_directory + "/" + x + '/struc.xyz' for x in sy] for sy in systems]
    systems_adapted = [list(map(format_xyz, systems)) for systems in systems_adapted]
    
    all_data = []
    
    for a, s, v, c, m, r in zip(
        systems_adapted, stoichiometry, reference_value, charges, multiplicities, reactions):
        system_dict = {}
        system_dict["atoms"] = a
        system_dict["stoichiometry"] = s
        system_dict["reference_value"] = v
        system_dict["charges"] = c
        system_dict["multiplicities"] = m
        system_dict["reaction"] = r
        all_data.append(system_dict)

    return all_data