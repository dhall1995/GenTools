import numpy as np
import os
import h5py
from ..utils.misc import bp_to_idx
from scipy.interpolate import CubicSpline

from ..utils.file_io import process_ncc 



class Contig(object):
    def __init__(self,
                 store,
                 UID,
                 limits
                ):
        self.store = store
        self.id = UID
        if limits is None:
            self.limits = self._calc_limits_from_positions()
        else:
            self.limits = limits
    
        
    @classmethod
    def from_binned_coords(cls,
                           UID,
                           bins,
                           coords,
                           tertiary_store,
                           tertiary_store_id):
        '''
        Generate a slice file from some binned coordinates.
        Arguments:
        - UID: The name of the tertiary structure
        - bins: (N+1,) shape array detailing the bin starts. Given that
                this is a Tertiary structure, bins are assumed to be 
                contiguous so the start of bin i is the end of bin i-1.
                coordinates are assumed to be the middle of the bins.
        - coords: (K,N,3) shape array detailing the coordinates of the bin
                  mids. This includes K different models. 
        - slice_file: The 
        '''
        store = h5py.File(tertiary_store, 'a', libver="latest")
        store.create_dataset(os.path.join("positions", UID), data=bins)
        store.create_dataset(os.path.join("coords", UID), data=coords)
        
        if "ids/{}".format(UID) not in store:
            store.create_group("ids/{}".format(UID))
        
        store.attrs["name"] = tertiary_store_id
        limits = [np.min(bins), np.max(bins)]
        
        return cls(store, UID, limits)
    
    @classmethod
    def from_contacts(cls,
                      UID,
                      contacts,
                      tertiary_store,
                      tertiary_store_id):
        
        store = h5py.File(tertiary_store, 'a', libver="latest")
        
        store.create_dataset(os.path.join("expr_contacts", UID), data=contacts)
        
        if "ids/{}".format(UID) not in store:
            store.create_group("ids/{}".format(UID))
        
        store.attrs["name"] = tertiary_store_id
        
        limits = [np.min(contacts[:,2:]),np.max(contacts[:,:2])]
        
        return cls(store, UID, limits)
    
    @static_method
    def _store_coords(store,
                      bins,
                      coords):
        try:
            del store["positions"][UID]
        except:
            # No coords in the store anyway
            pass
        
        try:
            del store["coords"][UID]
        except:
            pass
        
        store.create_dataset(os.path.join("positions", self.id), data=bins)
        store.create_dataset(os.path.join("coords", self.id), data=coords)
        
        logging.info("Stored basepair positions for contig {} in {}".format(self.id, store.attrs["name"]))
        logging.info("Stored binned coordinates for contig {} in {}".format(self.id, store.attrs["name"]))
     
    @static_method
    def _ncc_store_contacts(store,
                            ncc,
                            allowed_regions = None):
        try:
            del store["expr_contacts"][self.id][self.id]
        except:
            # No coords in the store anyway
            pass
        
        contacts = process_ncc(ncc,
                               chroms = [self.id],
                               contact_region = contact_regions)[self.id][self.id]
        
        store.create_dataset(os.path.join("expr_contacts", self.id, self.id), data=contacts)
        
        logging.info("Stored experimental contacts for contig {} in {}".format(self.id, store.attrs["name"]))
        
    @staticmethod
    def _hdf5_store_expr_contacts(archive,
                                  store,
                                  allowed_regions = None,
                                  path = "contacts/working"
                                ):
        """
        Store experimental contacts from .nuc file.
        """
        f = h5py.File(archive, 'r')
        k = list(f[path].keys())[0]
        contacts =  f[path][k][self.id][self.id][:]
        contig_path = os.path.join("expr_contacts", self.id, self.id)
        store.create_dataset(contig_path, data=contacts)
        logging.info("Created {}".format(contig_path))
        
    def depths(self, alpha=None, i=0, tag=None):
        i = str(i)

        if alpha is None and tag is None:
            raise TypeError("Neither alpha nor tag passed to function")

        dset = self.store["surface_dist"]

        if tag:
            for k in dset.keys():
                try:
                    t = dset[k].attrs["tag"]
                except KeyError:
                    pass
                else:
                    if t == tag:
                        return(dset[k][i][self.id][:])

            raise ValueError("No entry found for tag: {}".format(tag))

        elif alpha:
            return(dset[str(alpha)][self.id][str(i)][:])
        
    @property
    def cell(self):
        return(self.store.attrs["name"])

    @property
    def positions(self):
        return(self.store["positions"][self.id].astype(np.int32))

    @property
    def expr_contacts(self):
        return(self.store["expr_contacts"][self.id][self.id].astype(np.int32))
    
    def binned_contacts(self,
                        binning = None):
        if binning is None:
            binning = np.append(self.positions, self.limits[1])
            
        size = binning.shape[0] - 1

        idxs_a = bp_to_idx(self.expr_contacts[:,0].astype(np.int32), binning)
        idxs_b = bp_to_idx(self.expr_contacts[:,1].astype(np.int32), binning)
        counts = self.expr_contacts[:,2]

        return(coo_matrix(np.append(counts,counts),
                          (np.append(idxs_a,idxs_b),np.append(idxs_b,idxs_a),
                          shape=(size, size)))
    @property
    def dists(self):
        return(self.store["dists"][self.id][self.id])

    @property
    def coords(self):
        return(self.store["coords"][self.id][0,:,:])

    @property
    def rmsd(self):
        coords = self.coords
        mean_coord = np.mean(coord, axis=0)
        sq_vec_diff = np.square(coord - mean_coord)
        sq_diff = sq_vec_diff[:, :, 0] + sq_vec_diff[:, :, 1] + sq_vec_diff[:, :, 2]
        rmsd = np.mean(sq_diff, axis=0)
        return(rmsd)
    
    @property
    def cubicspline(self):
        binmids = np.floor(0.5*(self.positions[1:]+positions[:-1])).astype('int32')
        last_mid = 0.5*(self.positions[-1] + self.limits[1])
        binminds = np.append(binminds, last_mid)
            
        csx = CubicSpline(binmids, self.coords[:,0])
        csy = CubicSpline(binmids, self.coords[:,1])
        csz = CubicSpline(binmids, self.coords[:,2])
    
        return(csx, csy, csz)
               
    def interpolate_coords(self, bins):
               
        cs = self.CubicSpline
               
        if type(bins) == 'int':
            bins = np.append(np.arange(self.limits[0],
                                       self.limits[1],
                                       bins),
                             self.limits[1])
               
        nBins = len(bins) - 1
        newcoords = np.empty((nBins, 3))
    
        newcoords[:,0] = [cs[0].integrate(bins[i], bins[i+1]) / (bins[i+1] - bins[i])
                            for i in np.arange(nBins)]
        newcoords[:,1] = [cs[1].integrate(newbins[i], newbins[i+1]) / (newbins[i+1] - bins[i])
                            for i in np.arange(nBins)]
        newcoords[:,2] = [cs[2].integrate(bins[i], bins[i+1]) / (bins[i+1] - bins[i])
                            for i in np.arange(nBins)]
    
    
        return newcoords

#####################################################################################
#####################################################################################
               
class Chromosome(object):
    """
    Class to represent a single chromosome of a single cell.
    """
    def __init__(self, store, chrm, lower_bp_lim=None, upper_bp_lim=None):
        """
        Follows the HDF5 layout of NucFrame
        Contains cis-distances, coordinates, depths.
        Each property obeys the chromosome limits it has been passed, meaning
        indexes are directly comparable.
        """
        self.store = store
        self.chrm = chrm
        self.bin_size = store["bin_size"][()].astype(np.int32)
        self.valid = self.valid_idxs(lower_bp_lim, upper_bp_lim)

    def valid_idxs(self, lower, upper):
        # Assumes position is in sorted order .
        all_positions = self.store["bp_pos"][self.chrm][:]
        if lower and upper:
            return(np.logical_and(all_positions >= lower, all_positions < upper))
        elif lower and not upper:
            return(all_positions >= lower)
        elif not lower and upper:
            return(all_positions < upper)
        else:
            return(~np.isnan(all_positions))

    def depths(self, alpha=None, i=0, tag=None):
        i = str(i)

        if alpha is None and tag is None:
            raise TypeError("Neither alpha nor tag passed to function")

        dset = self.store["surface_dist"]

        if tag:
            for k in dset.keys():
                try:
                    t = dset[k].attrs["tag"]
                except KeyError:
                    pass
                else:
                    if t == tag:
                        return(dset[k][i][self.chrm][self.valid])

            raise ValueError("No entry found for tag: {}".format(tag))

        elif alpha:
            return(dset[str(alpha)][self.chrm][str(i)][:][self.valid])


    @property
    def cell(self):
        return(self.store.attrs["name"])

    @property
    def bp_pos(self):
        return(self.store["bp_pos"][self.chrm][:][self.valid].astype(np.int32))

    @property
    def expr_contacts(self):

        size = self.bp_pos.shape[0]
        
        bps_a = self.store["expr_contacts"][self.chrm][self.chrm][:, 0].astype(np.int32)
        bps_b = self.store["expr_contacts"][self.chrm][self.chrm][:, 1].astype(np.int32)
        counts = self.store["expr_contacts"][self.chrm][self.chrm][:, 2].astype(np.float32)

        idxs_a, valid_a = bp_to_idx(bps_a, self.bp_pos, self.bin_size)
        idxs_b, valid_b = bp_to_idx(bps_b, self.bp_pos, self.bin_size)

        valid = np.logical_and(valid_a, valid_b)

        idxs_a = idxs_a[valid]
        idxs_b = idxs_b[valid]
        counts = counts[valid]

        return(coo_matrix(np.append(counts,counts),
                          (np.append(idxs_a,idxs_b),np.append(idxs_b,idxs_a),
                          shape=(size, size)))

    #return(self.store["expr_contacts"][self.chrm][self.chrm][:][self.valid,:][:, self.valid])

    @property
    def dists(self):
        return(self.store["dists"][self.chrm][self.chrm][:][self.valid, :][:, self.valid])

    @property
    def positions(self):
        return(self.store["position"][self.chrm][:][:, self.valid, :])

    @property
    def rmsd(self):
        pos = self.positions
        mean_pos = np.mean(pos, axis=0)
        sq_vec_diff = np.square(pos - mean_pos)
        sq_diff = sq_vec_diff[:, :, 0] + sq_vec_diff[:, :, 1] + sq_vec_diff[:, :, 2]
        rmsd = np.mean(sq_diff, axis=0)
        return(rmsd)