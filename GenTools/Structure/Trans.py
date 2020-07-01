import numpy as np
import os
import h5py
from .Contig import Chromosome, Contig
from ..utils.misc import bp_to_idx
from scipy.sparse import coo_matrix

class TransContig(object):
    """
    Class to represent the relationship between two tertiary configurations.
    Should work with both true trans, and cis.
    """
    def __init__(self,
                 store,
                 id_a,
                 id_b,
                 limit_dict):
    """
    Follows the HDF5 layout of NucFrame
    Contains cis-distances, coordinates, depths.
    Each property obeys the chromosome limits it has been passed, meaning
    indexes are directly comparable.
    """
        self.store = store
        self.limit_dict = limit_dict

        limits_a = self.limit_dict[id_a]
        self.id_a = Contig(self.store, id_a, limits_a)
        limits_b = self.limit_dict[id_b]
        self.id_b = Contig(self.store, id_b, limits_b)

    @property
    def cell(self):
        return(self.store["name"][()])

    @property
    def positions(self):
        return(self.id_a.positions, self.id_b.positions)

    @property
    def coords(self):
        return(self.id_a.coords, self.id_b.coords)

    @property
    def expr_contacts(self):
        
        try:
            return self.store["expr_contacts"][self.id_a.id][self.id_b.id]
        except KeyError:
            try:
                return self.store["expr_contacts"][self.id_b.id][self.id_a.id]
            except:
                return np.empty((0,3)).astyp(np.int32)
 
    @property
    def binned_contacts(self,
                        binning_a = None,
                        binning_b = None):
        contact_data = self.expr_contacts
        
        if binning_a is None:
            binning_a = np.append(self.id_a.positions, self.id_a.limits[1])
            
        if binning_b is None:
            binninb_b = np.append(self.id_b.positions, self.id_b.limits[1])
        
        size_a = binning_a.shape[0] - 1
        size_b = binning_b.shape[0] - 1
        
        idxs_a = bp_to_idx(self.expr_contacts[:,0].astype(np.int32), binning_a)
        idxs_b = bp_to_idx(self.expr_contacts[:,1].astype(np.int32), binning_b)
        counts = self.expr_contacts[:,2]

        if self.id_a.id == self.id_b.id:
            mat = coo_matrix(np.append(counts,counts),
                              (np.append(idxs_a,idxs_b),np.append(idxs_b,idxs_a),
                              shape=(size_a, size_b))
        else:
            mat = coo_matrix(counts,(idxs_a,idxs_b),
                              shape=(size_a, size_b))               

        return(mat)

        
class Trans(object):
    """
    Class to represent the relationship between two chromosomes of a single cell.
    Should work with both true trans, and cis.
    """
    def __init__(self, store, chrm_a, chrm_b, chrm_limit_dict):
    """
    Follows the HDF5 layout of NucFrame
    Contains cis-distances, coordinates, depths.
    Each property obeys the chromosome limits it has been passed, meaning
    indexes are directly comparable.
    """
        self.store = store
        self.chrm_limit_dict = chrm_limit_dict

        lower_a, upper_a = self.chrm_limit_dict[chrm_a]
        self.chrm_a = Chromosome(self.store, chrm_a, lower_a, upper_a)
        lower_b, upper_b = self.chrm_limit_dict[chrm_b]
        self.chrm_b = Chromosome(self.store, chrm_b, lower_b, upper_b)

    @property
    def cell(self):
        return(self.store["name"][()])

    @property
    def bp_pos(self):
        return(self.chrm_a.bp_pos, self.chrm_b.bp_pos)

    @property
    def positions(self):
        return(self.chrm_a.positions, self.chrm_b.positions)

    @property
    def expr_contacts(self):

        try:
            contact_data = self.store["expr_contacts"][self.chrm_a.chrm][self.chrm_b.chrm]
            bps_a = contact_data[:, 0].astype(np.int32)
            bps_b = contact_data[:, 1].astype(np.int32)
            counts = contact_data[:, 2]
        except KeyError:
        # Trans contacts may be empty.
            try:
                contact_data = self.store["expr_contacts"][self.chrm_b.chrm][self.chrm_a.chrm]
                bps_a = contact_data[:, 1].astype(np.int32)
                bps_b = contact_data[:, 0].astype(np.int32)
                counts = contact_data[:, 2]
            except KeyError:
                bps_a = np.array([], dtype=np.int32)
                bps_b = np.array([], dtype=np.int32)
                counts = np.array([])


        size_a = self.chrm_a.bp_pos.shape[0]
        size_b = self.chrm_b.bp_pos.shape[0]

        positions_a = self.chrm_a.bp_pos
        positions_b = self.chrm_b.bp_pos

        idxs_a, valid_a = bp_to_idx(bps_a, positions_a, self.chrm_a.bin_size)
        idxs_b, valid_b = bp_to_idx(bps_b, positions_b, self.chrm_b.bin_size)


        valid = np.logical_and(valid_a, valid_b)

        idxs_a = idxs_a[valid]
        idxs_b = idxs_b[valid]
        counts = counts[valid]

        mat = np.zeros((size_a, size_b), dtype=np.int64)

        mat[idxs_a, idxs_b] = counts
        if self.chrm_a.chrm == self.chrm_b.chrm:
            mat[idxs_b, idxs_a] = counts

        return(mat)

    @property
    def dists(self):
        try:
            return(self.store["dists"][self.chrm_a.chrm][self.chrm_b.chrm][:][self.chrm_a.valid, :][:, self.chrm_b.valid])
        except KeyError:
            return(self.store["dists"][self.chrm_b.chrm][self.chrm_a.chrm][:][self.chrm_b.valid, :][:, self.chrm_a.valid].T)
