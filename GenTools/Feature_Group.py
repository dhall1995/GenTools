import numpy as np
from ..utils.dtrack_utils import link_parent_and_child_regions, link_parent_and_child_multi_regions, non_overlapping
from ..utils.dtrack_io import save_obj, load_obj
from ..utils.
from .Feature import Feature, link_features

import hdf5 as h
import pandas as pd
import os
import logging
from sklearn.metrics import pairwise_distances

def sim_affinity(X):
    return pairwise_distances(X, metric=hausdorff_dist_single_regions)

def get_node_members(children, distances, labels, dist_thresh):
        used = np.zeros(children.shape[0]+1)
    
        idxs = np.where(distances < dist_thresh)[0]
        if len(idxs) == 0:
            return [i for i in np.arange(children.shape[0])]
        else:
            children = children[:np.max(idxs),:]
    
        n_samples = len(labels)
        samps = [[i] for i in np.arange(n_samples)]
        for merge in children:
            x = samps[merge[0]]
            y = samps[merge[1]]
            samps.append(x + y)
        
            samps[merge[0]] = 0
            samps[merge[1]] = 0
        
        members = [item for item in samps if item != 0]
        return members 
    
    
'''
GROUPS OF FEATURES
'''
def Feature(object):
    '''
    Class to represent a group of features
    '''
    def __init__(self,
                 fgroup_file):
        """
        HDF5 hierarchy:
        name :: String -- the name of the Feature group
        subgroups :: ["X", "1", ..] -- all of the groups that are present. In most cases this
                                    will just be chromosomes
        regions/subgroup :: [[Int]] -- The regions associated with each feature. since there
                                       could be multiple regions associated with each feature
                                       this will be an (N,3) shape array where the first two
                                       collumns detail the region and the third collumn details
                                       the index of the feature id.
        ids/subgroup :: [[[int]]] -- (model, bead_idx, xyz)
        {ATTRIBUTE_NAME}/subgroup :: ?? shape array with some attribute per ID. If we have a >1
                                    dimensional feature then we also need to include a collumn
                                    to detail the index of the feature ID associated with that
                                    attribute. 
        """
        self.store = h5py.File(fgroup_file, mode=mode, libver="latest")
        
        self.groups = FeatureGroup(self.store, contigs, contig_limit_dict)
        self.links = FeatureLink(self.store, contigs, contig_limit_dict)
        
    @staticmethod
    def store_attribute(store,
                        attr_data,
                        attr_path,
                        attr_name = 'attribute'
                       ):
        '''
        General function to take a dictionary of attributes
        and store in in our feature group file. For example, we could
        have an (N,) shape array detailing the naive expression
        of each promoter on chromosome '1'. This would be in
        attr_docts['1']
        '''
        store.create_dataset(attr_path, data = attr_data)
        logging.info("Stored {} info at {} in {}".format(attr_name, attr_path, store.attrs["name"]))
    
    @staticmethod
    def _store_feature_clusters(self,
                                links,
                                clustering_name = "clusters"):
        for group in links:
            path = "{}/{}".format(clustering_name, group)
            store.create_dataset(path, data = links[group])
            logging.info("Stored {} info at {} in {}".format(clustering_name, path, store.attrs["name"]))
            
    @classmethod
    def from_feature_dict(cls,
                          fgroup_file,
                          feature_dict,
                          ftype,
                          regionskey = 'regions',
                          idskey = 'ids'
                          attrnames = None):
        try:
            os.remove(fgroup_file)
        except OSError:
            # File didn't actually exist.
            pass

        store = h.File(fgroup_file, 'a', libver="latest")
        store.attrs["name"] = ftype
        store["subgroups"] = featuredict['regions'].keys()
        if attrnames is None:
            attrnames = []
            
        for fgroup in feature_dict:
            if regionskey not in feature_dict[fgroup]:
                raise ValueError("no regions key for feature subgroup {}".format(fgroup)
            
            if idskey not in feature_dict[fgroup]:
                raise ValueError("no ids key for feature subgroup {}".format(fgroup)
            
            for attr in fgroup.keys():
                if attr in attrnames:
                    cls.store_attribute(store,
                                        feature_dict[fgroup][attr],
                                        "{}/{}".format(attr,fgroup),
                                        attr_name = "subgroup:{}\tattr:{}".format(fgroup, attr)
                
         return cls(fgroup_file)
                                    
    @classmethod
    def from_pickle(cls,
                    pickle_archive,
                    fgroup_file,
                    ftype,
                    **kwargs
                   ):
        fdict = load_obj(pickle_archive)
        
        return cls.from_feature_dict(fgroup_file,
                                     fdict,
                                     ftype,
                                     **kwargs)
    
    @classmethod
    def from_hdf5_archive(cls,
                         fgroup_file,
                         hdf5_archive,
                         ftype,
                         **kwargs):
        f = h.File(hdf5_archive, 'r')
        fdict = {}
        for attr in f:
            fdict[attr] = {}
            for fgroup in f[attr]:
                fdict[attr][fgroup] = f[attr][fgroup]
        
        return cls.from_feature_dict(fgroup_file,
                                     feature_dict,
                                     ftype)
    
    @classmethod
    def from_csv(cls,
                 fgroup_file,
                 csv,
                 ftype,
                 region_cols = ['start','end'],
                 group_by = ['Chromosome']
                 id_cols = ['id'],
                 attr_cols = None,
                 **kwargs):
        
        df = pd.read_csv(csv, **kwargs)
        fdict = {}
        fdict['regions'] = {}                                
        fdict['ids'] = {}                                
        grouped = df.groupby(group_by)
        for group_tuple in grouped.groups:
            group_path = "/".join([str(subg) for subg in group_tuple])
            fdict['regions'][group_path] = df.loc(grouped.groups[group_tuple],region_cols).values
            fdict['ids'][group_path] = np.apply_along_axis(lambda x: "-".join([str(item) for item in x]),
                                                           1,
                                                           df.loc(grouped.groups[group_tuple],
                                                                  id_col).values)
        if attr_cols is not None:
            for key in attr_cols:
                fdict[key] = {}
                for group_tuple in grouped.groups:
                    group_path = "/".join([str(subg) for subg in group_tuple])
                    fdict[key][group_path] = df.loc(grouped.groups[group_tuple],attr_cols[key]).values
        
        return cls.from_feature_dict(fgroup_file,
                                     fdict,
                                     ftype)
    
                                        
    def get_attrs(self,
                  attr,
                  allowed_subgroups = lambda x: True):
        
        def visitor_func(name, node):
            if isinstance(node, h.Dataset) and allowed_subgroups(name):
                names.append(name)
            else:
                pass
                                        
        group_names = {}
        for group in self.store[attr]:
            names = []
            self.store[attr][chrom].visititems(visitor_func)
        
            group_names[group] = names
        
        attrs = {group: np.concatenate([self.store[attrpath] for attrpath in group_names[group]], axis = 0) for group in self.store[attr]}
        return attrs
    
    def _cluster_features_by_attr(self,
                                  attr,
                                  attrs_transform = lambda x: x,
                                  metric = hausdorff_dist_single_regions,
                                  **kwargs
                                 ):
        clusters = {}
        attrs = self.get_attrs(attr, **kwargs)
        attrs = {group: attrs_transform(attrs[group]) for group in attrs}
        def fn(group):
            cluster = AgglomerativeClustering(distance_threshold=0,
                                  n_clusters=None,
                                  affinity=sim_affinity,
                                  linkage='complete',
                                  compute_full_tree = True
                                 )
                                        
            
            
            cluster = cluster.fit(np.array(attrs[group]).astype('float32'))

            return cluster, group

        p = Pool()
        temp_outputs = p.imap(fn, (group for group in self.store[attr]))
        for temp_output in temp_outputs:
            print("Done Group {}".format(temp_output[1]))
            clusters[temp_output[1]] = temp_output[0]

        p.close()
        p.terminate()
        
        return clusters
    
    @staticmethod
    def store_clustering(self,
                         clusters,
                         path):
        for group in clusters:
            store.create_dataset(path+group+"/distances", data = clusters[group].distances_)
            store.create_dataset(path+group+"/children", data = clusters[group].children_)  
            store.create_dataset(path+group+"/labels", data = clusters[group].labels_)
            logging.info("Stored {} info at {} in {}".format(attr_name, attr_path, store.attrs["name"]))
                                  
    @staticmethod
    def get_links_from_clustering(
        