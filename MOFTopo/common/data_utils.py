from os import listdir
from os.path import join, isfile

from pathlib import Path

from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis import local_env

from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops


########## Helper Functions ##########
def list_file_paths_in_dir(dir):
    """
    Creates a list of all full paths to the files in a directory
    """
    fpaths = []
    for fname in listdir(dir):
        pth = join(dir, fname)

        if isfile(pth):
            fpaths.append(pth)
    
    return fpaths


########## Tools for Parsing .cif Topology Files ##########
# For usage, call build_crystal first, then call build_crystal_graph on the crystal. See parse_zeolites for an example. 
def build_crystal(crystal_fpath,):
    crystal = Structure.from_file(crystal_fpath).get_reduced_structure()

    canonical_crystal = Structure(
        lattice=Lattice.from_parameters(*crystal.lattice.parameters),
        species=crystal.species,
        coords=crystal.frac_coords,
        coords_are_cartesian=False,
    )

    return canonical_crystal


def build_crystal_graph(crystal, name, graph_method):
    crystal_graph = StructureGraph.from_local_env_strategy(crystal, graph_method)

    frac_coords = crystal.frac_coords
    atom_types = crystal.atomic_numbers
    lattice_parameters = crystal.lattice.parameters
    lengths = lattice_parameters[:3]
    angles = lattice_parameters[3:]

    edge_indices, to_jimages = [], []
    for i, j, to_jimage in crystal_graph.graph.edges(data='to_jimage'):
        edge_indices.append([j, i])
        to_jimages.append(to_jimage)
        edge_indices.append([i, j])
        to_jimages.append(tuple(-tj for tj in to_jimage))

    atom_types = torch.tensor(atom_types)
    lengths, angles = torch.tensor(lengths), torch.tensor(angles)
    edge_indices, _ = add_self_loops(torch.tensor(edge_indices).T)
    to_jimages = torch.tensor(to_jimages)
    num_atoms = atom_types.shape[0]

    graph_data = Data(
        name=name,
        num_nodes=num_atoms,
        frac_coords=frac_coords,
        atom_types=atom_types,
        length=lengths,
        angles=angles,
        edge_index=edge_indices,
        to_jimages=to_jimages,
    )

    return graph_data


########## Parsing Zeolite Topologies ##########
def parse_single_zeolite(zeo_raw_path, zeo_out_dir):
    zeo_name = Path(raw_path).stem

    zeo_cryst = build_crystal(zeo_raw_path)
    crystal_graph_data = build_crystal_graph(zeo_cryst, zeo_name, crystal_nn)

    return crystal_graph_data


def parse_zeolites(zeo_raw_dir, zeo_out_dir):
    crystal_nn = local_env.CrystalNN(distance_cutoffs=None, x_diff_weight=-1, porous_adjustment=False)

    zeo_raw_paths = list_file_paths_in_dir(zeo_raw_dir)

    print("Starting to Process Zeolites")

    for zeo_raw_path in zeo_raw_paths:
        crystal_graph_data = parse_single_zeolite(zeo_raw_path, zeo_out_dir,)

        torch.save(crystal_graph_data, parsed_path)

    print("Finished Processing Zeolites")





