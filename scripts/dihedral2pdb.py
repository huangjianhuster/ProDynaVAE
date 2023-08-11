from Bio.PDB import PDBParser, PICIO, PDBIO
from Bio.PDB.internal_coords import *
from Bio.PDB.PICIO import write_PIC, read_PIC, read_PIC_seq
from Bio.PDB.ic_rebuild import write_PDB, IC_duplicate, structure_rebuild_test


# functions
def get_dihedral(pdb):
    parser = PDBParser(PERMISSIVE=1, QUIET=1)
    structure = parser.get_structure("test", pdb)[0]

    structure.atom_to_internal_coordinates()
    chain = list(structure.get_chains())[0]
    ic_chain = chain.internal_coord
    return ic_chain.dihedra

def reconstructPDB(pic_file, pdb_output):
    myprotein = read_PIC(pic_file, verbose=1)
    myprotein.internal_to_atom_coordinates()
    write_PDB(myprotein, pdb_output)
    return None

def update_dihedral(pdb, dihedral_array, pdb_output):
    parser = PDBParser(PERMISSIVE=1, QUIET=1)
    structure = parser.get_structure("test", pdb)[0]

    structure.atom_to_internal_coordinates()
    # check structure makes sense
    resultDict = structure_rebuild_test(structure)
    assert resultDict['pass'] == True
    
    # update diherdals
    chain = list(structure.get_chains())[0]
    ic_chain = chain.internal_coord
    # make changes to dihedral angles accodring to dihedral_array
    for i in dihedral_array:
        value = i   # may need to modify
        # do something here; need to consider which dihedral to change
        # for eample, which resid, which dihedral angle and value
        # let's say, [(resid1, dihedral, value), ]
        res = next(structure.get_residues())
        res_chi2 = res.internal_coord.get_angle("chi2")
        res.internal_coord.set_angle("chi2", value)
    structure.internal_to_atom_coordinates()
    write_PDB(structure, "pdb_output")
    return None

# parse structure from PDB
parser = PDBParser(PERMISSIVE=1, QUIET=1)
structure = parser.get_structure("test", "./test.pdb")[0]

structure.atom_to_internal_coordinates()
chain = list(structure.get_chains())[0]
ic_chain = chain.internal_coord
d = ic_chain.dihedra

# make changes to the dihedra angles
cnt = 1
for key in d:    
    if key[0].akl[3] == 'N':
        if key[1].akl[3] == 'CA':
            if key[2].akl[3] == 'C':
                if key[3].akl[3] == 'N':
                    print ('\n',cnt,' :   ',  [x.akl[3] for x in key], d[key].angle)
                    d[key].angle += 45
                    cnt += 1

cnt = 1
for key in d:
    if key[0].akl[3] == 'N':
        if key[1].akl[3] == 'CA':
            if key[2].akl[3] == 'C':
                if key[3].akl[3] == 'N':
                    print ('\n',cnt,' :   ',  [x.akl[3] for x in key], d[key].angle)
                    cnt += 1
                    
structure.internal_to_atom_coordinates(verbose = True)

io = PDBIO()
io.set_structure(structure)
io.save('modified.pdb',  preserve_atom_numbering=True) 
