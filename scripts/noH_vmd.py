from vmd import atomsel
from vmd import molecule
import sys

pdb = sys.argv[1]
mol = molecule.load("pdb", pdb)
noH = atomsel("protein and not hydrogen")
noH.write("pdb", "noH-vmd.pdb")
