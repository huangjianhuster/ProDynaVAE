from psfgen import PsfGen
import sys

gen = PsfGen()
top = "/home2/jianhuang/tutorial/forcefield/toppar/top_all22_prot.rtf"
pdb = sys.argv[1]

gen.read_topology(top)
segid='A'
gen.add_segment(segid=segid, pdbfile=pdb)
gen.patch('ACE', [(segid, gen.get_resids(segid)[0]),])
gen.patch('CT2', [(segid, gen.get_resids(segid)[-1]),])

# Delete hydrogens
for resid in gen.get_resids(segid):
    hydrogens = [i for i in gen.get_atom_names(segid, resid) if i.startswith("H")]
    for hydro in hydrogens:
        gen.delete_atoms(segid, resid, hydro)

gen.write_psf("./test.psf")

