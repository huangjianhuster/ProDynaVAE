# Test
# 1. take a protein PDB
# 2. write out torsion data
# 3. only use torsion data, rebuild back the PDB
# 4. compare original PDB and rebuilt PDB

import MDAnalysis as mda
from MDAnalysis.analysis.bat import BAT
import numpy as np

test_psf = "./files/test.psf"
test_pdb = "./files/test.pdb"

# write out torsion angles
u = mda.Universe(test_psf, test_pdb)
selection = u.select_atoms("protein")
R = BAT(selection)
R.run()
IC_matrix = R.results.bat[0, :] # this actually will contain EC (the first 9 elements are EC)
print("Internal Coordinate (the first nine are external Coordinates)\n", IC_matrix)
np.save("./files/test_bat.npy", IC_matrix)

# build back PDB
# use PSF
IC_matrix_readin = np.load("./files/test_bat.npy")
XYZ = R.Cartesian(IC_matrix_readin)

print("Rebuilt coordinates: ")
print(XYZ)

def writePDB(old_pdb, XYZ, out_pdb):
    with open(old_pdb, "r") as f:
        lines = f.readlines()
    atoms = []
    for line in lines:    
        if line.startswith("ATOM"):
            atoms.append(line)
    assert len(atoms) == len(XYZ)
    
    new_atoms = []
    for i,j in zip(atoms, XYZ):
        head = i[:32]
        new_coor = "%.3f  %.3f  %.3f" % (j[0], j[1], j[2])
        end = i[54:]
        new_atoms.append( head + new_coor + end )
    with open(out_pdb, "w") as g:
        g.writelines(new_atoms)

    return None

out_pdb = "./files/test_rebuilt.pdb"
writePDB(test_pdb, XYZ, out_pdb)
# output_pdb = "./files/test_rebuild.pdb"
