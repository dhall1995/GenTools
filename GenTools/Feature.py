import numpy as np
from .utils.datatracks.dtrack_utils import link_parent_and_child_regions, link_parent_and_child_multi_regions, non_overlapping
from .utils.file_io import save_obj, load_obj

'''
BASE CLASSES
'''
class Feature(object):
    '''
    Base class for our feature object. Information should be provided detailing:
    - fid: The feature ID. For example, if a gene then this would be the ensemble ID. Basically
           just a unique identifier for the feature.
    - chromosome:
    - region:
    
    Optional arguments:
    - parents: Often we may want to consider a feature as some sub-feature of something larger.
               For example, a gene may belong to a certain TAD or an enhancer may appear within
               a certain hub. This should be a dictionary with keys as ftypes and key-values
               detailing the parent features. Depending on memory considerations this could be 
               a dictionary which literally contains the parent objects or it could just be
               the feature IDs of the parent features (which can then be searched for).
    - children: Similarly, a feature may naturally contain subfeatures which we may want to 
                have access to when we look at a feature. An example might be a TAD containing
                a CG island. This should be a dictionary with keys as ftypes and key-values
                detailing the child features. Depending on memory considerations this could be 
                a dictionary which literally contains the child objects or it could just be
                the feature IDs of the child features (which can then be searched for).
             
    '''
    def __init__(self,
                 fid,
                 chromosome,
                 regions,
                 attrs = None,
                 parents = None,
                 children = None
                ):
        self.id = fid
        if parents is None:
            self.parents = {}
        else:
            self.parents = parents
        
        if children is None:
            self.children = {}
        else:
            self.children = children
        
        self.attrs = {}
        self.attrs['regions'] = regions
        self.attrs['chromosome'] = chromosome
        if attrs is not None:
            for key in attrs:
                self.attrs[key] = attrs[key]

    def plot_flattened_regions(self,
                               y_idx,
                               ax,
                               col = 'lightskyblue',
                               s = 1,
                               region_transform = lambda x: x):
        points = np.ravel(region_transform(self.attrs['regions']))
        
        ax.scatter(points,
                   np.full(points.shape,y_idx),
                   marker = 'X',
                   c = c,
                   s = s)
        
    def add_child(self,
                  child,
                  child_type):
        '''
        Function to add a child feature to a given feature. The child is stored in the self.children 
        dictionary under the key given by child_type. If not child_type is supplied then the child_type
        is inferred from the given child.
        Arguments:
        - child: Another Feature type object (or a Feature subclass type object). If no child_type is passed
                 then child must have a .type attribute or the child can't be added to self.children.
        - child_type: A given child will be stored in the self.children dictionary and child_type specifies
                      what type of child is passed. If no child_type is specified then the method will try and
                      infer the child_type from the child. However, manually specifying the child_type is
                      allowed since users may want to specify different sub-types of children of the same
                      type (e.g. hub-anchor associated promoters vs. hub-contact associated promoters)
        '''    
        if child_type in self.children:
            self.children[child_type].append(child)
        else:
            self.children[child_type] = [child]
    
    def remove_child(self,
                     child_type,
                     child_id):
        '''
        Function to remove a child given some child_id
        '''
        if child_type not in self.children:
            raise ValueError("{} type not in self.children".format(child_type))
        
        self.children[child_type] = [item for item in self.children[child_type] if item.id != child_id]
        
    def remove_child_type(child_type):
        '''
        Function to remove all children of a certain type
        '''
        try:
            del self.children[child_type]
        except:
            print("Couldn't delete {} from self.children. It may not exist already".format(child_type))
            
    def add_parent(self,
                   parent,
                   parent_type):
        '''
        Function to add a parent feature to a given feature. The child is stored in the self.parent 
        dictionary under the key given by child_type. If no parent_type is supplied then the parent_type
        is inferred from the given parent.
        Arguments:
        - parent: Another Feature type object (or a Feature subclass type object). If no parent_type is passed
                 then parent must have a .type attribute or the parent can't be added to self.parents.
        - parent_type: A given parent will be stored in the self.parents dictionary and child_type specifies
                      what type of parent is passed. If no parent_type is specified then the method will try and
                      infer the parent_type from the parent. However, manually specifying the parent_type is
                      allowed since users may want to specify different sub-types of parent of the same
                      type (e.g. hub-anchor associated TADs vs. hub-contact associated TADs)
        '''
        if parent_type in self.parents:
            self.parents[parent_type].append(parent)
        else:
            self.parents[parent_type] = [parent]
            
    def remove_parent(self,
                     parent_type,
                     parent_id):
        '''
        Function to remove a parent given some parent_id
        '''
        if parent_type not in self.parents:
            raise ValueError("{} type not in self.parents".format(parent_type))
        
        self.parents[parent_type] = [item for item in self.parents[parent_type] if item.id != parent_id]
    
    def remove_parent_type(parent_type):
        '''
        Function to remove all children of a certain type
        '''
        try:
            del self.parents[parent_type]
        except:
            print("Couldn't delete {} from self.parents. It may not exist already".format(parent_type))
            
            
    def get_child_ids(self, child_type):
        '''
        Return the .id attributes of the children within self.children[child_type]
        - child_type: Key to search for in the dictionary self.children in order to return a list of .id
                      attributes.
        '''
        if child_type not in self.children:
            raise ValueError("{} not in self.children.".format(child_type))
        elif len(self.children[child_type]) == 0:
            raise ValueError("No children in self.children[{}]".format(child_type))
        
        try:
            return [child.id for child in self.children[child_type]]
        except:
            raise ValueError("Looks like objects in self.children are not a Feature objects! Stopping.")
    
    def get_parent_ids(self, parent_type):
        '''
        Return the .id attributes of the parents within self.parents[child_type]
        - parent_type: Key to search for in the dictionary self.parents in order to return a list of .id
                      attributes.
        '''
        if parent_type not in self.parents:
            raise ValueError("{} not in self.parents.".format(parent_type))
        elif len(self.children[parent_type]) == 0:
            raise ValueError("No parents in self.parents[{}]".format(parent_type))
            
        try:
            return [parent.id for parent in self.parents[parent_type]]
        except:
            raise ValueError("Looks like objects in self.parents are not a Feature objects! Stopping.")
    
    def get_parent_attr(self,
                        parent_type,
                        attr_type):
        '''
        Searches for and returns the .attr attribute of parents within self.parents[parent_type]
        - parent_type: Key to search for in the dictionary self.parents
        - attr_type: Key to search for in the dictionary parent.attrs for each parent in
                     self.parents[parent_type]
        '''
        if parent_type not in self.parents:
            raise ValueError("{} not in self.parents.".format(parent_type))
        elif len(self.children[parent_type]) == 0:
            raise ValueError("No parents in self.parents[{}]".format(parent_type))
            
        out = []
        for parent in self.parents[parent_type]:
            try:
                if attr_type not in parent.attrs:
                    print("Couldn't find {} in parent.attrs for parent: {}".format(attr_type,parent.id))
                    print("Appending np.nan to output")
                    out.append(np.nan)
                else:
                    out.append(parent.attrs[attr_type])
            except:
                raise ValueError("Looks like {} is not a Feature object! Stopping.".format(parent))
        return out
    
    def get_children_attr(self,
                          child_type,
                          attr_type):
        '''
        Searches for and returns the .attr attribute of children within self.children[child_type]
        - child_type: Key to search for in the dictionary self.children
        - attr_type: Key to search for in the dictionary child.attrs for each child in
                     self.children[child_type]
        '''
        if child_type not in self.children:
            raise ValueError("{} not in self.children.".format(child_type))
        elif len(self.children[parent_type]) == 0:
            raise ValueError("No children in self.children[{}]".format(child_type))
            
        out = []
        for child in self.children[child_type]:
            try:
                if attr_type not in child.attrs:
                    print("Couldn't find {} in child.attrs for child: {}".format(attr_type,child.id))
                    print("Appending np.nan to output")
                    out.append(np.nan)
                else:
                    out.append(child.attrs[attr_type])
            except:
                raise ValueError("Looks like {} is not a Feature object! Stopping.".format(child))
                
        return out
    
    def get_parent_types(self):
        '''
        Return all the types of parents which are associated with this object
        '''
        return [key for key in self.parents]
    
    def get_child_types(self):
        '''
        Return all the types of children which are associated with this object
        '''
        return [key for key in self.children]
    
    def get_attr_types(self):
        '''
        Return all the attribute types associated with this object
        '''
        return [key for key in self.attrs]
    
    def to_pickle(self, path):
        save_obj(self, path)
        
class Feature_single_condition(Feature):
    '''
    Essentially the same as a normal feature except that it is one which we observe within
    a single cell. An example of this would be a chromatin 'hub' where one anchor region
    makes mutliple simultaneous long distance contacts in some single-cell dataset.
    '''
    def __init__(self,
                 fid,
                 condition,
                 chromosome,
                 regions,
                 attrs = None,
                 parents = None,
                 children = None):
        super(Feature_single_condition, self).__init__(fid,
                                                       chromosome,
                                                       regions,
                                                       attrs = attrs,
                                                       parents = parents,
                                                       children = children)
        self.attrs['condition'] = condition
                             
                             
class Feature_single_cell(Feature_single_condition):
    '''
    Essentially the same as a normal single-condition feature except that it is one 
    which we observe within a single cell. An example of this would be a chromatin
    'hub' where one anchor region makes mutliple simultaneous long distance contacts
    in some single-cell dataset.
    '''
    def __init__(self,
                 fid,
                 condition,
                 cell,
                 chromosome,
                 regions,
                 attrs = None,
                 parents = None,
                 children = None
                ):
        super(Feature_single_cell, self).__init__(fid,
                                                  condition,
                                                  chromosome,
                                                  regions,
                                                  attrs = attrs,
                                                  parents = parents,
                                                  children = children)
        self.attrs['cell'] = cell
        

'''
GENERAL FEATURE FUNCTIONS
'''
def link_features(parent_regions,
                  child_regions):
    '''
    Given some list of regions associated with each parent and child, this returns an (N,2) shape
    array detailing the links between parents and children. 
    Arguments:
    parent_regions: A list of (M,2) shape arrays (one per parent) which need not be the same length
                    but which detail the regions associated with each parent.
    child_regions: A list of (M,2) shape arrays (one per child) which need not be the same length
                   but which detail the regions associated with each child.
    
    Returns:
    Links: (N,2) shape array (essentially a COO format sparse matrix) where each row details 
           a link between a parent and a child. 
    '''
    
    biggest_p = np.max([region.shape[0] for region in parent_regions])
    biggest_c = np.max([region.shape[0] for region in child_regions])
    
    smallest_p = np.min([np.min(region) for region in parent_regions])
    smallest_c = np.min([np.min(region) for region in child_regions])
    
    minval = np.minimum(smallest_p, smallest_c)
    
    pregions = np.full((len(parent_regions), biggest_p, 2), minval - 1)
    cregions = np.full((len(child_regions), biggest_c, 2), minval - 1)
    
    for idx in np.arange(len(parent_regions)):
        pregion = parent_regions[idx]
        pregions[idx,:pregion.shape[0],:] = pregion

    for idx in np.arange(len(child_regions)):
        cregion = child_regions[idx]
        cregions[idx,:cregion.shape[0],:] = cregion

    if pregions.shape[1] == 1 and cregions.shape[1] == 1:
        links = link_parent_and_child_regions(pregions[:,0,:].astype('int32'),
                                              cregions[:,0,:].astype('int32'),
                                              allow_partial = True
                                                  )
    else:
        links = link_parent_and_child_multi_regions(pregions.astype('int32'),
                                                        cregions.astype('int32'),
                                                        cutoff = minval-1,
                                                        allow_partial = True
                                                         )
    
    return links
    
'''
SPECIFIC CLASSES
'''    

'''
HUB CLASSES
'''
class Experimental_Hub(Feature_single_cell):
    def __init__(self,
                 UID,
                 condition, 
                 cell,
                 chromosome,
                 contacts,
                 attrs = None,
                 parents = None,
                 children = None):
        super(Experimental_Hub, self).__init__(UID,
                                               condition,
                                               cell,
                                               chromosome,
                                               contacts,
                                               attrs = attrs,
                                               parents = parents,
                                               children = children)
        
class Hub(Feature):
    '''
    Class to represent groups of experimental hubs. Note that this class assumes that
    actual Experimental_Hub objects are added as children since methods such as 
    get_conditions require access to the Experimental_hub.condition attributes.  
    '''
    def __init__(self,
                 name,
                 chromosome,
                 attrs = None,
                 parents = None,
                 children = None):
        super(Hub, self).__init__(name,
                                  chromosome,
                                  None,
                                  attrs = attrs,
                                  parents = parents,
                                  children = children)
        self.attrs['conditions'] = []
        if children is None:
            self.children['experimental_hub'] = []
    
    def add_experimental_hub(self, exp_hub):
        if exp_hub.attrs['chromosome'] == self.attrs['chromosome']:
            self.add_child(exp_hub, 'experimental_hub')
        else:
            print("Experimental hub chromosome doesn't match abstract hub chromosome") 
            
    def get_cells(self):
        '''
        Utility to return the cell names of the experimental hubs within
        this hub system.
        '''
        cells = {}
        for cell in self.children['experimental_hub']:
            condition = cell.attrs['condition']
            if condition not in cells:
                cells[condition] = [cell.attrs['cell']]
            else:
                cells[condition].append(cell.attrs['cell'])
                                        
        return cells
                                        
    def get_conditions(self):
        '''
        Utility to return the conditions in which this hub is observed
        '''
        conds = []
        for cell in self.children['experimental_hub']:
            condition = cell.attrs['condition']
            if condition not in conds:
                conds.append(condition)
                                        
        return conds 
                                        
                                        
    def get_contact_data(self,
                         distance_cond = lambda x: True,
                         accepted_conditions = set(['naive','rexpos','rexneg','primed']),
                         hub_cond = lambda x: True,
                         accepted_cells = None
                        ):
        if accepted_cells is None:
            accepted_cells = self.get_cells()
        data = np.empty((0,4))
        
        if len(self.children['experimental_hub']) == 0:
            return data
         
        for hub in self.children['experimental_hub']:
            if not hub_cond(hub):
                continue
                
            if hub.attrs['condition'] not in accepted_conditions:
                continue
                
            if hub.attrs['cell'] not in accepted_cells[hub.attrs['condition']]:
                continue
                
            cont_lengths = abs(np.mean(hub.attrs['regions'][:,2:],
                                       axis = 1) - np.mean(hub.attrs['regions'][:,:2],
                                                           axis = 1)
                              )
            
            idxs = []
            for idx,item in enumerate(cont_lengths):
                if distance_cond(item):
                    idxs.append(idx)
            idxs = np.array(idxs)
            if len(idxs) >0:
                contact_addition = hub.attrs['regions'][idxs,:]
                data = np.append(data, contact_addition, axis = 0)
            
         
        return data

    def get_nonoverlapping_regions(self,
                                   buffer = 1e4,
                                   distance_cond = lambda x: True,
                                   accepted_conditions = set(['naive','rexpos','rexneg','primed']),
                                   hub_cond = lambda x: True,
                                   accepted_cells = None,
                                   region_type = 'regions'):
        data = self.get_contact_data(distance_cond, 
                                     accepted_conditions,
                                     hub_cond,
                                     accepted_cells,
                                    )
        if data.shape[0] == 0:
            return None
        else:
            if region_type == 'anchors':
                regions = np.floor(np.mean(data[:,:2],axis = 1))
            elif region_type == 'contacts':
                regions = np.floor(np.mean(data[:,2:],axis = 1))
            else:
                regions = np.floor(np.mean(np.append(data[:,:2],data[:,2:],axis = 0),axis = 1))

            regions = np.repeat(regions[:,None],2,axis = 1)
            regions[:,0] -= (buffer*np.ones((regions.shape[0],))).astype('int32')
            regions[:,1] += (buffer*np.ones((regions.shape[0],))).astype('int32')
            
        return non_overlapping(regions.astype('int32'))

'''
TAD Class
'''
class TAD(Feature_single_condition):
    '''
    Base class to represent a TAD
    '''
    def __init__(self,
                 fid,
                 condition,
                 chromosome,
                 region,
                 attrs = None,
                 parents = None,
                 children = None):
                         
        super(TAD,self).__init__(fid,
                                 condition,
                                 chromosome,
                                 region,
                                 attrs = attrs,
                                 parents= parents,
                                 children = children)
                         
                         
                         
'''
Gene Class
'''
class Gene(Feature):
    def __init__(self,
                 fid,
                 chromosome,
                 genebody,
                 strand,
                 promoter = None,
                 attrs = None,
                 parents = None,
                 children = None):
        super(Gene,self).__init__(fid,
                                  chromosome,
                                  genebody,
                                  attrs = attrs,
                                  parents= parents,
                                  children = children)
        self.attrs['strand'] = strand
        if promoter is not None:
            self.children['promoter'] = promoter
                         
'''
Promoter Class
'''
class Promoter(Feature):
    def __init__(self,
                 fid,
                 chromosome,
                 region,
                 strand,
                 gene = None,
                 attrs = None,
                 parents = None,
                 children = None):
        super(Promoter,self).__init__(fid,
                                      chromosome,
                                      region,
                                      attrs = attrs,
                                      parents= parents,
                                      children = children)
        self.attrs['strand'] = strand
        if gene is not None:
            self.parents['gene'] = gene  
                     
                                            
                         
'''
Enhancer Class
'''
class Enhancer(Feature):
    def __init__(self,
                 fid,
                 chromosome,
                 region,
                 enhancer_type,
                 attrs = None,
                 parents = None,
                 children = None):
        super(Enhancer,self).__init__(fid,
                                      chromosome,
                                      region,
                                      attrs = attrs,
                                      parents = parents,
                                      children = children)
        self.attrs['enhancer_type'] = enhancer_type
                         
        