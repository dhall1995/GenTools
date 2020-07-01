import numpy as np
import os
import h5py
from itertools import combinations_with_replacement, combinations, product
import logging
from tqdm import tqdm
import networkx as nx
import math
from collections import defaultdict, deque, Counter

from ..utils.distance_utils.all_pairs_euc_dist import nuc_dist_pairs
from ..utils.depth_utils.alpha_shape import AlphaShape, circular_subgroup
from ..utils.depth_utils.point_surface_dist import points_tris_dists
from ..utils.file_io import process_ncc
from .Contig import Chromosome, Contig
from .Trans import Trans, TransContig


def surf_norm(tri):
    a = tri[1] - tri[0]
    b = tri[2] - tri[0]
    return (np.cross(a, b))


class NucFrame(object):
    """
    Class to represent a Nuc file, in a consistent fashion.

    -- Used by a NucFrame
    -- Calculates the maximum information possible (e.g. all distances)
    -- Somehow allow initialisation with allowed basepair limits.
    -- All getters / setters should then use these limits to return the appropriate stuff.

    """

    @classmethod
    def from_nuc(cls, nuc_file, nuc_slice_file):
        """
        Factory to produce a NucFrame (and the associated file) from a
        .nuc file.
        """

        try:
            os.remove(nuc_slice_file)
        except OSError:
            # File didn't actually exist.
            pass

        store = h5py.File(nuc_slice_file, 'a', libver="latest")

        store["chrms"] = cls._extract_chrms(nuc_file)
        chrms = [x.decode("utf8") for x in store["chrms"]]
        store["bin_size"] = cls._get_bin_size(nuc_file, chrms)
        store.attrs["name"] = cls._get_name(nuc_file)
        cls._store_bp_positions(nuc_file, store, chrms)
        cls._store_expr_contacts(nuc_file, store, chrms)
        cls._store_dists(nuc_file, store, chrms)
        cls._store_positions(nuc_file, store, chrms)
        return (cls(nuc_slice_file))

    @staticmethod
    def _extract_chrms(nuc_file):
        # TODO: Extract from nuc_file, rather than just setting.
        chrms = ["X"] + [str(x) for x in range(1, 20)]
        return (list(map(lambda x: x.encode("utf-8"), chrms)))

    @staticmethod
    def _get_bin_size(nuc_file, chrms):
        sizes = set()
        particle = h5py.File(nuc_file, 'r')["structures"]["0"]["particles"]
        for chrm in chrms:
            positions = particle[chrm]["positions"][:]
            chrm_sizes = np.diff(positions)
            sizes = sizes.union(set(chrm_sizes))

        # TODO: I need some messed up files to work.
        sizes = {math.floor(x / 1000) * 1000 for x in sizes}
        if len(sizes) != 1:
            raise ValueError("Inconsistent bin sizes: {}".format(len(sizes)))
        else:
            return (list(sizes)[0])

    @staticmethod
    def _get_name(nuc_file):
        raw_name, _ = os.path.splitext(os.path.basename(nuc_file))
        raw_name = raw_name.replace("-", "_")
        name = "_".join(raw_name.split("_")[:3])
        return (name)

    @staticmethod
    def _store_bp_positions(nuc_file, store, chrms):
        """
        Create the datastore for each chromosome that stores the basepair
        positions of each structural particle.
        """
        nuc = h5py.File(nuc_file, 'r')
        chrm_parts = nuc["structures"]["0"]["particles"]
        for chrm in chrms:
            positions = chrm_parts[chrm]["positions"][:]

            positions = [math.floor(x / 1000) * 1000 for x in positions]

            if np.all(np.sort(positions) != positions):
                raise ValueError("Positions not in sorted order.")

            store.create_dataset(os.path.join("bp_pos", chrm), data=positions)
            logging.info("Stored basepair positions for chrm {} in {}".format(chrm, store.attrs["name"]))

    @staticmethod
    def _store_expr_contacts(nuc_file, store, chrms):
        """
        Store experimental contacts from .nuc file.
        """
        f = h5py.File(nuc_file, 'r')
        k = list(f["contacts"]["working"].keys())[0]
        contact_chrm_as = f["contacts"]["working"][k]
        for chrm_a in contact_chrm_as.keys():
            if chrm_a not in chrms:
                continue
            contact_chrm_bs = contact_chrm_as[chrm_a]
            for chrm_b in contact_chrm_bs.keys():
                if chrm_b not in chrms:
                    continue

                contact_vals = contact_chrm_bs[chrm_b][:].T
                chrm_path = os.path.join("expr_contacts", chrm_a, chrm_b)
                store.create_dataset(chrm_path, data=contact_vals)
                logging.info("Created {}".format(chrm_path))

    @staticmethod
    def _store_dists(nuc_file, store, chrms):
        """
        Store the average distance between particles in a chromosome.
        """

        # A:B == B:A.T, so just add one, and add an accessor method
        chrm_pairs = list(combinations_with_replacement(chrms, 2))
        for (chrm_a, chrm_b) in tqdm(chrm_pairs):
            a_to_b, b_to_a = nuc_dist_pairs(nuc_file, chrm_a, chrm_b)
            dists_path = os.path.join("dists", chrm_a, chrm_b)

            dists = np.median(a_to_b, axis=2)

            try:
                store.create_dataset(dists_path, data=dists)
            except RuntimeError as e:
                logging.info("{}".format(e))
            else:
                logging.info("Created {}".format(dists_path))

    @staticmethod
    def _store_positions(nuc_file, store, chrms):
        """
        Store 3d positions of each particle in each model.
        """
        f = h5py.File(nuc_file, 'r')["structures"]["0"]["coords"]
        for chrm in chrms:
            position_path = os.path.join("position", chrm)
            store.create_dataset(position_path, data=f[chrm])
            logging.info("Created positions for chrm {} in {}".format(chrm, store.attrs["name"]))

    def _store_alpha_shape(self, rmsd_lim=5):
        """Calculates and stores an AlphaShape.
        If called from a NucFrames group, will be incorrect, as not all positions would
        be considered."""
        store = self.store

        all_positions = []
        all_void = []
        for chrm in self.chrms:
            chrm_pos = chrm.positions[0, :, :]
            chrm_rmsd = chrm.rmsd

            void = chrm_rmsd < rmsd_lim

            all_void.append(void)
            all_positions.append(chrm_pos)

        all_void = np.concatenate(all_void)
        all_positions = np.vstack(all_positions)
        all_idx = np.arange(all_positions.shape[0])

        filtered_pos = all_positions[all_void]
        filtered_idx = all_idx[all_void]

        # Store alpha_shape.interval_dict
        alpha_shape = AlphaShape.from_points(filtered_pos)
        try:
            del (store["alpha_shape"])
        except KeyError as e:
            pass

        for k in {len(x) for x in alpha_shape.interval_dict.keys()}:
            simplices = []
            ab_values = []
            for simplex, (a, b) in alpha_shape.interval_dict.items():
                if len(simplex) == k:
                    # Convert back to unfiltered coordinates.
                    simplex = tuple(filtered_idx[np.array(simplex)])
                    simplices.append(simplex)
                    ab_values.append([a, b])

            path = os.path.join("alpha_shape", str(k))
            store.create_dataset(os.path.join(path, "simplices"), data=simplices)
            store.create_dataset(os.path.join(path, "ab"), data=ab_values)
    logging.info("Created AlphaShape dataset")

    def _load_alpha_shape(self):
        interval_dict = {}
        for k in self.store["alpha_shape"].keys():
            simplices = self.store["alpha_shape"][k]["simplices"][:]
            ab_values = self.store["alpha_shape"][k]["ab"][:]
            for simplex, ab in zip(simplices, ab_values):
                interval_dict[tuple(simplex)] = ab

        self.alpha_shape = AlphaShape(interval_dict, self.all_pos)

    def alpha_surface(self, alpha=1.6):
        """
        For a given value of alpha, return all surfaces, ordered by size.
        """
        if not self.alpha_shape:
            self._load_alpha_shape()

        all_pos = self.all_pos
        all_facets = list(self.alpha_shape.get_facets(alpha))
        # Construct the graph
        edges = {x for y in all_facets for x in circular_subgroup(y, 2)}
        nodes = {x for y in edges for x in y}
        g = nx.Graph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)

        surfaces = []

        # Iterate over subgraphs, ordered by size.
        for sg in (sorted(nx.connected_component_subgraphs(g), key=lambda x: len(x), reverse=True)):

            valid_nodes = set(sg.nodes())

            # Filter facets
            facet_vert_idxs = np.array([x for x in all_facets if all_in(x, valid_nodes)])
            facet_vert_coords = np.array([all_pos[x] for x in facet_vert_idxs], dtype=np.float32)

            flip_order = [1, 0, 2]
            flip_facet_vert_coords = facet_vert_coords[:, flip_order, :]
            # Precompute norms
            facet_norms = np.cross(facet_vert_coords[:, 0, :] - facet_vert_coords[:, 1, :],
                             facet_vert_coords[:, 1, :] - facet_vert_coords[:, 2, :])
            flip_facet_norms = np.cross(flip_facet_vert_coords[:, 0, :] - flip_facet_vert_coords[:, 1, :],
                                  flip_facet_vert_coords[:, 1, :] - flip_facet_vert_coords[:, 2, :])

      # Ensure consistent vertex ordering
      # Check that the normal of each facet is in the same direction as its neighbour.

            vert_idx_facet_idx_lu = defaultdict(set)
            for facet_idx, facet in enumerate(facet_vert_idxs):
                for vert_idx in facet:
                    vert_idx_facet_idx_lu[vert_idx].add(facet_idx)

            facet_neighbor_lu = defaultdict(set)
            for facet_idx, facet in enumerate(facet_vert_idxs):
                c = Counter()
                for vert_idx in facet:
                    c.update(vert_idx_facet_idx_lu[vert_idx] - set([facet_idx]))
                facet_neighbor_lu[facet_idx] = {x for x, n in c.items() if n >= 2}

            processed_facets = set([0])
            d = deque()
            d.append(0)
            while True:
                try:
                    facet_idx = d.popleft()
                except IndexError:
                    break

                facet_n = facet_norms[facet_idx]

                # Neighboring facets
                neighbor_idxs = facet_neighbor_lu[facet_idx] - processed_facets

                for neighbor_idx in neighbor_idxs:
                    neighbor_n = facet_norms[neighbor_idx]
                    proj = np.dot(facet_n, neighbor_n)

                    if proj < 0:
                        t = facet_vert_coords[neighbor_idx]
                        t_ = facet_norms[neighbor_idx]

                        facet_vert_coords[neighbor_idx] = flip_facet_vert_coords[neighbor_idx]
                        facet_norms[neighbor_idx] = flip_facet_norms[neighbor_idx]

                        flip_facet_vert_coords[neighbor_idx] = t
                        flip_facet_norms[neighbor_idx] = t_

                    if proj != 0:
                        d.append(neighbor_idx)
                        processed_facets.add(neighbor_idx)

            surfaces.append(facet_vert_coords)
        return (surfaces)

    def store_surface_dists_tag(self, alpha, tag):
        """
        Since there are so many surfaces, sometimes we want
        to add a tag. E.g. EXTERNAL for distances to exterior.
        This is because different cells might have different alpha values
        for the external surface.
        """
        path = "surface_dist"

        # Check tag isn't already present
        for test_alpha in self.store[path].keys():
            try:
                self.store[path][test_alpha].attrs["tag"]
            except KeyError:
                pass
            else:
                del self.store[path][test_alpha].attrs["tag"]

        path = os.path.join("surface_dist", str(alpha))
        self.store[path].attrs["tag"] = tag

    def store_surface_dists(self, alpha=1.6):
        """
        Store the absolute distance of each particle from each surface.
        It is likely that there will be multiple disconnected surfaces found. The
        outer surface will be valid, as will an inner surface if present.

        Use a value of alpha to define the surface, and a percentage to decide how
        big stored subgraphs should be.
        """
        surfaces = self.alpha_surface(alpha)

        try:
            del self.store["surface_dist"][str(alpha)]
        except KeyError:
            pass

        for i, facets in enumerate(surfaces):
            facets = facets.astype(np.float32)
            surface_size = facets.shape[0]

            # Store information about surface.
            path = os.path.join("surface_dist", str(alpha), str(i))
            self.store.create_dataset(os.path.join(path, "surface_size"), data=surface_size)

            for chrm in self.chrms:
                chrm_pos = chrm.positions[0, :, :].astype(np.float32)
                surface_dists = np.min(points_tris_dists(facets, chrm_pos), axis=1)
                self.store.create_dataset(os.path.join(path, chrm.chrm), data=surface_dists)

    def __init__(self, nuc_slice_file, chrm_limit_dict=None, mode="r", rmsd_lim=8):
        """
        HDF5 hierarchy:
        name :: String -- the name of the NucFrame
        bin_size :: Int -- the common bin_size of the nuc files.
        chrms :: ["X", "1", ..] -- all of the chromosomes that are present.
        bp_pos/chrm :: [Int] -- The start bp index of each particle in each chrm.
        position/chrm :: [[[Float]]] -- (model, bead_idx, xyz)
        expr_contacts/chrm/chrm :: [[Int]] -- (bp, bp), raw contact count.
        dists/chrm/chrm :: [[Float]] -- (bead_idx, bead_idx), distanes between beads.
        depths/i/alpha :: Float -- alpha value used to calculate depths.
        depths/i/chrm/ :: [Float] -- (bead_idx, ), depth of point from surface i.
        alpha_shape/k/simplices :: [[Int]] -- (n_simplicies, k), indices of k-simplices.
        alpha_shape/k/ab :: [(a, b)] -- (n_simplicies, 2), a and b values for k-simplices.
                        ^ -- NOTE: length of the two alpha_shape entries align.
        surface_dist/alpha_val/tag :: optional tag for this value of alpha.
        surface_dist/alpha_val/i/surface_size :: size of surface i for alpha
        """
        self.alpha_shape = None
        self.store = h5py.File(nuc_slice_file, mode=mode, libver="latest")
        self.bin_size = self.store["bin_size"].value
        self.nuc_slice_file = nuc_slice_file
        chromosomes = [x.decode("utf-8") for x in self.store["chrms"]]
        if not chrm_limit_dict:
            chrm_limit_dict = {chrm: (None, None) for chrm in chromosomes}

        self.chrms = ChromosomeGroup(self.store, chromosomes, chrm_limit_dict)
        self.trans = TransGroup(self.store, chromosomes, chrm_limit_dict)

        try:
            self.store["alpha_shape"]
        except KeyError:
            self._store_alpha_shape(rmsd_lim=rmsd_lim)

    @property
    def all_pos(self):
        all_positions = []
        for chrm in self.chrms:
            all_positions.append(chrm.positions[0, :, :])

        all_positions = np.vstack(all_positions)
        return (all_positions)

    @property
    def all_pos_all_models(self):
        all_positions = []
        for chrm in self.chrms:
            pos = chrm.positions[:, :, :]
            all_positions.append(pos)

        all_positions = np.concatenate(all_positions, 1)
        return (all_positions)

    @property
    def cell_name(self):
        return (self.store.attrs["name"])

    @property
    def all_rmsd(self):
        pos = self.all_pos_all_models
        mean_pos = np.mean(pos, axis=0)
        sq_vec_diff = np.square(pos - mean_pos)
        sq_diff = sq_vec_diff[:, :, 0] + sq_vec_diff[:, :, 1] + sq_vec_diff[:, :, 2]
        rmsd = np.mean(sq_diff, axis=0)
        return (rmsd)


class ChromosomeGroup(object):
    def __init__(self, store, chromosomes, chrm_limit_dict):
        self.store = store
        self.chrms = chromosomes
        self.chrm_limit_dict = chrm_limit_dict

    def __getitem__(self, chrm):
        lower, upper = self.chrm_limit_dict[chrm]
        return (Chromosome(self.store, chrm, lower, upper))

    def __iter__(self):
        for chrm in self.chrms:
            lower, upper = self.chrm_limit_dict[chrm]
            yield (Chromosome(self.store, chrm, lower, upper))


class TransGroup(object):
    def __init__(self, store, chromosomes, chrm_limit_dict):
        self.store = store
        self.chrms = chromosomes
        self.chrm_limit_dict = chrm_limit_dict

    def __getitem__(self, chrm_tuple):
        chrm_a, chrm_b = chrm_tuple
        return (Trans(self.store, chrm_a, chrm_b, self.chrm_limit_dict))

    def __iter__(self):
        for (chrm_a, chrm_b) in product(self.chrms, repeat=2):
            yield (Trans(self.store, chrm_a, chrm_b, self.chrm_limit_dict))

def all_in(tup, s):
    for t in tup:
        if t not in s:
            return False
    return True

#############################################################################
#############################################################################
#############################################################################
#############################################################################

class ConFrame(object):
    """
    Class to represent a sructure configuration, in a consistent fashion.
    """

    @classmethod
    def from_hdf5(cls,
                  archive,
                  config_frame_file,
                  contig_ids = None,
                  config_name = None,
                  position_path = "structures/0/particles",
                  coord_path = "structures/0/coords",
                  contact_path = None
                  dists = None):
        """
        Factory to produce a Config (and the associated file) from an
        .hdf5 file archive. The archive should contain at minimum
        information about both coordinates and bins or 
        """

        try:
            os.remove(config_frame_file)
        except OSError:
            # File didn't actually exist.
            pass

        store = h5py.File(config_frame_file, 'a', libver="latest")
        
        if contig_ids is None:
            store["contigs"] = cls._extract_contigs(archive,
                                            path_to_contigs = coord_path)
        else:
            store["contigs"] = contig_ids
        contig_ids = [x.decode("utf8") for x in store["ids"]]
        store["bin_size"] = cls._get_bin_size(archive,
                                              contig_ids,
                                              position_path=position_path)
        
        if archive_name is None:
            store.attrs["name"] = cls._get_name(archive)
        else:
            store.attrs["name"] = archive_name
            
        cls._store_coords(archive,
                          store,
                          contig_ids,
                          path = coord_path)
        
        cls._store_positions(archive,
                             store,
                             contig_ids,
                             path = position_path)
        
        if contact_path is not None:
            cls._hdf5_store_expr_contacts(archive,
                                 store,
                                 contig_ids,
                                 path = contact_path
                                )
            
        return (cls(nuc_slice_file))

    @staticmethod
    def _extract_contigs(archive, path_to_contigs):
        particle = h5py.File(archive, 'r')[path_to_contigs]
        contigs = list(particle)
        return (list(map(lambda x: x.encode("utf-8"), contigs)))

    @staticmethod
    def _get_bin_size(archive,
                      contig_ids,
                      position_path = "structures/0/particles",
                     ):
        sizes = set()
        particle = h5py.File(archive, 'r')[position_path]
        for contig_id in contig_ids:
            positions = particle[contig_id]['positions'][:]
            contig_sizes = np.diff(positions)
            sizes = sizes.union(set(contig_sizes))

        # TODO: I need some messed up files to work.
        sizes = {math.floor(x / 1000) * 1000 for x in sizes}
        if len(sizes) != 1:
            print("Inconsistent bin sizes: {}".format(len(sizes)))
            return np.nan
        else:
            return (list(sizes)[0])

    @staticmethod
    def _get_name(archive):
        raw_name, _ = os.path.splitext(os.path.basename(archive))
        raw_name = raw_name.replace("-", "_")
        name = "_".join(raw_name.split("_")[:3])
        return (name)

    @staticmethod
    def _store_positions(archive,
                         store,
                         contig_ids,
                         path = "structures/0/particles"
                        ):
        """
        Create the datastore for each chromosome that stores the basepair
        positions of each structural particle.
        """
        nuc = h5py.File(nuc_file, 'r')
        contig_parts = archive[path]
        for contig_id in contig_ids:
            positions = particle[contig_id]['positions'][:]

            positions = [math.floor(x / 1000) * 1000 for x in positions]

            if np.all(np.sort(positions) != positions):
                raise ValueError("Positions not in sorted order.")

            store.create_dataset(os.path.join("positions", contig_id), data=positions)
            logging.info("Stored basepair positions for chrm {} in {}".format(contig_id, store.attrs["name"]))

    @staticmethod
    def _hdf5_store_expr_contacts(archive,
                                 store,
                                 contigs,
                                 path = "contacts/working"
                                ):
        """
        Store experimental contacts from .hdf5 file.
        """
        f = h5py.File(archive, 'r')
        k = list(f[path].keys())[0]
        contact_contig_as = f[path][k]
        for contig_a in contact_contig_as.keys():
            if contig_a not in contigs:
                continue
            contact_contig_bs = contact_contig_as[contig_a]
            for contig_b in contact_contig_bs.keys():
                if contig_b not in contigs:
                    continue

                contact_vals = contact_contig_bs[contig_b][:].T
                contig_path = os.path.join("expr_contacts", contig_a, contig_b)
                store.create_dataset(contig_path, data=contact_vals)
                logging.info("Created {}".format(contig_path))
                
    @staticmethod
    def _ncc_store_expr_contacts(ncc,
                                 store,
                                 contigs = None):
        """
        Store experimental contacts from .ncc file.
        """
        if contigs is None:
            contigs = store["contigs"]
            
        contacts = process_ncc(ncc,
                               chroms = contigs):
        
        for contig_a in contacts.keys():
            if contig_a not in contigs:
                continue
            for contig_b in contacts[contig_a].keys():
                if contig_b not in contigs:
                    continue

                contact_vals = contacts[contig_a][contig_b]
                contig_path = os.path.join("expr_contacts", contig_a, contig_b)
                store.create_dataset(contig_path, data=contact_vals)
                logging.info("Created {}".format(contig_path))
                
    @staticmethod
    def _store_coords(archive,
                      store,
                      contig_ids,
                      path = "structures/0/coords"
                     ):
        """
        Store 3d positions of each particle in each model.
        """
        f = h5py.File(archive, 'r')[path]
        for contig_id in contig_ids:
            position_path = os.path.join("coords", contig_id)
            data = f[contig_id]
            if len(data.shape) < 3:
                #only one model
                data = data[None,:,:]
            store.create_dataset(position_path, data=data)
            logging.info("Created coords for chrm {} in {}".format(contig_id, store.attrs["name"]))

    def _store_alpha_shape(self, rmsd_lim=5):
        """Calculates and stores an AlphaShape.
        If called from a NucFrames group, will be incorrect, as not all positions would
        be considered."""
        store = self.store

        all_positions = []
        all_void = []
        for contig in self.contigs:
            contig_coords = contig.positions[0, :, :]
            contig_rmsd = contig.rmsd

            void = contig_rmsd < rmsd_lim

            all_void.append(void)
            all_positions.append(contig_coords)

        all_void = np.concatenate(all_void)
        all_coords = np.vstack(all_coords)
        all_idx = np.arange(all_coords.shape[0])

        filtered_coords = all_coords[all_void]
        filtered_idx = all_idx[all_void]

        # Store alpha_shape.interval_dict
        alpha_shape = AlphaShape.from_points(filtered_pos)
        try:
            del (store["alpha_shape"])
        except KeyError as e:
            pass

        for k in {len(x) for x in alpha_shape.interval_dict.keys()}:
            simplices = []
            ab_values = []
            for simplex, (a, b) in alpha_shape.interval_dict.items():
                if len(simplex) == k:
                    # Convert back to unfiltered coordinates.
                    simplex = tuple(filtered_idx[np.array(simplex)])
                    simplices.append(simplex)
                    ab_values.append([a, b])

            path = os.path.join("alpha_shape", str(k))
            store.create_dataset(os.path.join(path, "simplices"), data=simplices)
            store.create_dataset(os.path.join(path, "ab"), data=ab_values)
    logging.info("Created AlphaShape dataset")

    def _load_alpha_shape(self):
        interval_dict = {}
        for k in self.store["alpha_shape"].keys():
            simplices = self.store["alpha_shape"][k]["simplices"][:]
            ab_values = self.store["alpha_shape"][k]["ab"][:]
            for simplex, ab in zip(simplices, ab_values):
                interval_dict[tuple(simplex)] = ab

        self.alpha_shape = AlphaShape(interval_dict, self.all_coords)

    def alpha_surface(self, alpha=1.6):
        """
        For a given value of alpha, return all surfaces, ordered by size.
        """
        if not self.alpha_shape:
            self._load_alpha_shape()

        all_coords = self.all_coords
        all_facets = list(self.alpha_shape.get_facets(alpha))
        # Construct the graph
        edges = {x for y in all_facets for x in circular_subgroup(y, 2)}
        nodes = {x for y in edges for x in y}
        g = nx.Graph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)

        surfaces = []

        # Iterate over subgraphs, ordered by size.
        for sg in (sorted(nx.connected_component_subgraphs(g), key=lambda x: len(x), reverse=True)):

            valid_nodes = set(sg.nodes())

            # Filter facets
            facet_vert_idxs = np.array([x for x in all_facets if all_in(x, valid_nodes)])
            facet_vert_coords = np.array([all_pos[x] for x in facet_vert_idxs], dtype=np.float32)

            flip_order = [1, 0, 2]
            flip_facet_vert_coords = facet_vert_coords[:, flip_order, :]
            # Precompute norms
            facet_norms = np.cross(facet_vert_coords[:, 0, :] - facet_vert_coords[:, 1, :],
                             facet_vert_coords[:, 1, :] - facet_vert_coords[:, 2, :])
            flip_facet_norms = np.cross(flip_facet_vert_coords[:, 0, :] - flip_facet_vert_coords[:, 1, :],
                                  flip_facet_vert_coords[:, 1, :] - flip_facet_vert_coords[:, 2, :])

      # Ensure consistent vertex ordering
      # Check that the normal of each facet is in the same direction as its neighbour.

            vert_idx_facet_idx_lu = defaultdict(set)
            for facet_idx, facet in enumerate(facet_vert_idxs):
                for vert_idx in facet:
                    vert_idx_facet_idx_lu[vert_idx].add(facet_idx)

            facet_neighbor_lu = defaultdict(set)
            for facet_idx, facet in enumerate(facet_vert_idxs):
                c = Counter()
                for vert_idx in facet:
                    c.update(vert_idx_facet_idx_lu[vert_idx] - set([facet_idx]))
                facet_neighbor_lu[facet_idx] = {x for x, n in c.items() if n >= 2}

            processed_facets = set([0])
            d = deque()
            d.append(0)
            while True:
                try:
                    facet_idx = d.popleft()
                except IndexError:
                    break

                facet_n = facet_norms[facet_idx]

                # Neighboring facets
                neighbor_idxs = facet_neighbor_lu[facet_idx] - processed_facets

                for neighbor_idx in neighbor_idxs:
                    neighbor_n = facet_norms[neighbor_idx]
                    proj = np.dot(facet_n, neighbor_n)

                    if proj < 0:
                        t = facet_vert_coords[neighbor_idx]
                        t_ = facet_norms[neighbor_idx]

                        facet_vert_coords[neighbor_idx] = flip_facet_vert_coords[neighbor_idx]
                        facet_norms[neighbor_idx] = flip_facet_norms[neighbor_idx]

                        flip_facet_vert_coords[neighbor_idx] = t
                        flip_facet_norms[neighbor_idx] = t_

                    if proj != 0:
                        d.append(neighbor_idx)
                        processed_facets.add(neighbor_idx)

            surfaces.append(facet_vert_coords)
        return (surfaces)

    def store_surface_dists_tag(self,
                                alpha,
                                tag,
                                path = "surface_dist"
                               ):
        """
        Since there are so many surfaces, sometimes we want
        to add a tag. E.g. EXTERNAL for distances to exterior.
        This is because different cells might have different alpha values
        for the external surface.
        """

        # Check tag isn't already present
        for test_alpha in self.store[path].keys():
            try:
                self.store[path][test_alpha].attrs["tag"]
            except KeyError:
                pass
            else:
                del self.store[path][test_alpha].attrs["tag"]

        path = os.path.join(path, str(alpha))
        self.store[path].attrs["tag"] = tag

    def store_surface_dists(self,alpha=1.6):
        """
        Store the absolute distance of each particle from each surface.
        It is likely that there will be multiple disconnected surfaces found. The
        outer surface will be valid, as will an inner surface if present.

        Use a value of alpha to define the surface, and a percentage to decide how
        big stored subgraphs should be.
        """
        surfaces = self.alpha_surface(alpha)

        try:
            del self.store["surface_dist"][str(alpha)]
        except KeyError:
            pass

        for i, facets in enumerate(surfaces):
            facets = facets.astype(np.float32)
            surface_size = facets.shape[0]

            # Store information about surface.
            path = os.path.join("surface_dist", str(alpha), str(i))
            self.store.create_dataset(os.path.join(path, "surface_size"), data=surface_size)

            for contig in self.contigs:
                contig_pos = contig.positions[0, :, :].astype(np.float32)
                surface_dists = np.min(points_tris_dists(facets, contig_pos), axis=1)
                self.store.create_dataset(os.path.join(path, contig.id), data=surface_dists)

    def __init__(self,
                 config_frame_file,
                 limit_dict=None,
                 mode="r",
                 rmsd_lim=8):
        """
        HDF5 hierarchy:
        name :: String -- the name of the NucFrame
        bin_size :: Int -- the common bin_size of the nuc files.
        contigs :: ["X", "1", ..] -- all of the chromosomes that are present.
        positions/contig :: [Int] -- The start bp index of each particle in each chrm.
        coords/contig :: [[[Float]]] -- (model, bead_idx, xyz)
        expr_contacts/contig/contig :: [[Int]] -- (bp, bp), raw contact count.
        dists/chrm/contig :: [[Float]] -- (bead_idx, bead_idx), distanes between beads.
        depths/i/alpha :: Float -- alpha value used to calculate depths.
        depths/i/contig/ :: [Float] -- (bead_idx,), depth of point from surface i.
        alpha_shape/k/simplices :: [[Int]] -- (n_simplicies, k), indices of k-simplices.
        alpha_shape/k/ab :: [(a, b)] -- (n_simplicies, 2), a and b values for k-simplices.
                        ^ -- NOTE: length of the two alpha_shape entries align.
        surface_dist/alpha_val/tag :: optional tag for this value of alpha.
        surface_dist/alpha_val/i/surface_size :: size of surface i for alpha.
        alt_positions/tag/contig :: for some new binning, store the positions of each
                                    bin start.
        alt_coords/tag/contig :: for some new binning, store the coordinates of each
                                 bin middle.
        """
        self.alpha_shape = None
        self.store = h5py.File(config_frame_file, mode=mode, libver="latest")
        self.bin_size = self.store["bin_size"].value
        self.config_frame_file = config_frame_file
        contigs = [x.decode("utf-8") for x in self.store["contigs"]]
        if not limit_dict:
            limit_dict = {contig: (None, None) for contig in contigs}

        self.contigs = ContigGroup(self.store, contigs, limit_dict)
        self.trans = MyTransGroup(self.store, contigs, limit_dict)

        try:
            self.store["alpha_shape"]
        except KeyError:
            self._store_alpha_shape(rmsd_lim=rmsd_lim)

    @property
    def all_coords(self):
        all_coords = []
        for contig in self.contigs:
            all_coords.append(contig.coords[0, :, :])

        all_coords = np.vstack(all_coords)
        return (all_coords)

    @property
    def all_coords_all_models(self):
        all_coords = []
        for contig in self.contigs:
            pos = contig.coords[:, :, :]
            all_coords.append(pos)

        all_coords = np.concatenate(all_coords, 1)
        return (all_coords)
    
    @property
    def all_cubic_splines(self, **kwargs):
        all_cs = {}
        for contig in self.contigs:
            all_cs[contig_id] = contig.cubic_spline()
            
        return (all_cs)
    
    @property
    def cell_name(self):
        return (self.store.attrs["name"])

    @property
    def all_rmsd(self):
        coords = self.all_coords_all_models
        mean_pos = np.mean(pos, axis=0)
        sq_vec_diff = np.square(pos - mean_pos)
        sq_diff = sq_vec_diff[:, :, 0] + sq_vec_diff[:, :, 1] + sq_vec_diff[:, :, 2]
        rmsd = np.mean(sq_diff, axis=0)
        return (rmsd)

    def rebin_coords(self, bins, tag = None):
        if type(bins) == 'int':
            bins = {contig_id: bins for contig_id in [contig.id for contig in self.contigs]}
        
        for contig in self.contigs:
            newcoords = contig.interpolate_coords(bins[contig.id])
       
        #TO DO: MAKE THE NEW BINNING STORE STUFF CORRECTLY INSIDE THE ConFrame
        return newcoords
class ContigGroup(object):
    def __init__(self,
                 store,
                 contigs,
                 limit_dict):
        self.store = store
        self.contigs = contigs
        self.limit_dict = limit_dict

    def __getitem__(self, contig):
        limits = self.limit_dict[contig]
        return (Contig(self.store, contig, limits))

    def __iter__(self):
        for contig in self.contigs:
            limits = self.limit_dict[UID]
            yield (Contig(self.store, contig, limits))


class TransContigGroup(object):
    def __init__(self,
                 store,
                 contigs,
                 limit_dict):
        self.store = store
        self.contigs = contigs
        self.limit_dict = limit_dict

    def __getitem__(self, contig_tuple):
        contig_a, contig_b = contig_tuple
        return (TransContig(self.store, contig_a, contig_b, self.limit_dict))

    def __iter__(self):
        for (contig_a, contig_b) in product(self.contigs, repeat=2):
            yield (TransContig(self.store, contig_a, contig_b, self.limit_dict))


