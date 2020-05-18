import numpy as np

'''
BASE CLASSES
'''
class Feature(object):
    '''
    Base class for our feature object. Information should be provided detailing:
    - ftype: The type of feature e.g. promoter or hub. Note that the ftype should be timepoint
             independent - active promoter wouldn't be a type since a promoter need not be 
             active at all timepoints. Rather, being active is a property of a promoter at 
             a given timepoint/ in a given condition.
    - fid: The feature ID. For example, if a gene then this would be the ensemble ID. Basically
           just a unique identifier for the feature.
    
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
                 ftype,
                 fid,
                 attrs = None,
                 parents = None,
                 children = None
                ):
        self.type = ftype
        self.id = fid
        if parents is None:
            self.parents = {}
        else:
            self.parents = parents
        
        if children is None:
            self.children = {}
        else:
            self.children = children
            
        if attrs is None:
            self.attrs = {}
        else:
            self.attrs = attrs
            
    def add_child(self,
                  child,
                  child_type = None):
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
        if child_type is None:
            try:
                child_type = child.type
            except:
                raise ValueError("{} not a Feature object. Child sas no attribute 'type' and no child_type passed to add_child.".format(child))
            
        if child_type in self.children:
            self.children[child_type].append(child)
        else:
            self.children[child_type] = [child]
    
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
        if parent_type is None:
            try:
                parent_type = parent.type
            except:
                raise ValueError("{} not a Feature object. Parent has no attribute 'type' and no parent_type passed to add_parent.".format(parent))
            
        if parent_type in self.parents:
            self.parents[parent_type].append(parent)
        else:
            self.parent[parent_type] = [parent]
    
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
            raise ValueError("{} not in self.parents.".format(parent_type)
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
        
class Feature_single_condition(Feature_single_condition):
    '''
    Essentially the same as a normal feature except that it is one which we observe within
    a single cell. An example of this would be a chromatin 'hub' where one anchor region
    makes mutliple simultaneous long distance contacts in some single-cell dataset.
    '''
    def __init__(self,
                 ftype,
                 fid,
                 condition,
                 attrs = None,
                 parents = None,
                 children = None):
        super(Feature_sc, self).__init__(ftype,
                                         fid,
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
                 ftype,
                 fid,
                 cell,
                 condition,
                 attrs,
                 parents = None,
                 children = None
                ):
        super(Feature_sc, self).__init__(ftype,fid, parents, children, condition)
        self.attrs['cell'] = cell

'''
GENERAL FEATURE FUNCTIONS
'''
def link_features(parent_list,
                  child_list,
                  parent_key = 'regions',
                  child_key = 'regions',
                  parent_preprocess = lambda x: x,
                  child_preprocess = lambda x: x,
                  child_type = None,
                  parent_type = None
                 ):
    if child_type is None:
        try:
            child_type = child_list[0].type
        except:
            ValueError("Looks like children in child_list aren't Feature objects! Stopping.")
    if parent_type is None:
        try:
            parent_type = parent_list[0].type
        except:
            ValueError("Looks like parents in parent_list aren't Feature objects! Stopping.")
                             
    pregions = np.empty((0,2))
    cregions = np.empty((0,2))
    for feature in parent_list:
        pregions = np.append(pregions, parent_preprocess(feature.attrs[parent_key], axis = 0)
    for feature in child_list:
        cregions = np.append(cregions, child_preprocess(feature.attrs[child_key], axis = 0)
    
    pchildren = dtu.multi_pairRegionsIntersection(pregions.astype('int32'),cregions.astype('int32'))
    cparents = dtu.multi_pairRegionsIntersection(cregions.astype('int33'), pregions.astype('int32'))
    
    pchildren, cparents
    
'''
SPECIFIC CLASSES
'''    

'''
HUB CLASSES
'''
class Experimental_Hub(Feature_sc):
    def __init__(self,
                 condition, 
                 cell,
                 chromosome,
                 contacts,
                 UID,
                 parents = None,
                 children = None
                ):
        super(Experimental_Hub, self).__init__('experimental_hub',
                                               UID,
                                               cell,
                                               condition,
                                               parents,
                                               children)
        self.attrs['contacts'] = contacts
        self.attrs['chromosome'] = chromosome
        
class Hub(Feature):
    '''
    Class to represent groups of experimental hubs. Note that this class assumes that
    actual Experimental_Hub objects are added as children since methods such as 
    get_conditions require access to the Experimental_hub.condition attributes.  
    '''
    def __init__(self,
                 name,
                 chromosome,
                 parents = None,
                 children = None):
        super(Hub, self).__init__('Hub', name, parents, children)
        self.attrs['chromosome'] = chromosome
        self.attrs['conditions'] = []
        if children is None:
            self.children['experimental_hub'] = []
    
    def add_experimental_hubs(self, exp_hub):
        if exp_hub.chromosome == self.chromosome:
            self.add_child(exp_hub, 'experimental_hub')
        else:
            print("Experimental hub chromosome doesn't match abstract hub chromosome")   
    
    def get_contact_data(self,
                         distance_cond = None,
                         accepted_conditions = set([0,'Rexhi','Rexlow',48]):
        data = np.empty((0,4))
        
        if len(self.children['experimental_hub']) == 0:
            return data
         
        for hub in self.children['experimental_hub']:
            if hub.attrs['condition'] in accepted_conditions:
                cont_lengths = np.mean(hub.attrs['contacts'][:,2:], axis = 1) - np.mean(hub.attrs['contacts'][:,:2], axis = 1)
                if distance_cond is not None:
                    idxs = []
                    for idx,item in enumerate(cont_lengths):
                        if distance_cond(item):
                            idxs.append(idx)
                    idxs = np.array(idxs)
                    if len(idxs) >0:
                        contact_addition = hub.attrs['contacts'][idxs,:]
                        data = np.append(data, contact_addition, axis = 0)
                else:
                    contact_addition = hub.attrs['contacts']
                    data = np.append(data, contact_addition, axis = 0)
         
         return data

    def get_nonoverlapping_regions(self,
                                   buffer = 1e4,
                                   distance_cond = None,
                                   accepted_conditions = set([0,'Rexhi','Rexlow',48]),
                                   region_type = 'regions'):
        data = self.get_contact_data(distance_cond, accepted_conditions)
        if data.shape[0] == 0:
            return 0
        else:
            if region_type = 'anchors':
                regions = data[:,:2]
            elif region_type = 'contacts':
                regions = data[:,2:]
            else:
                regions = np.append(data[:,:2],data[:,2:],axis = 0)
            for idx, item in enumerate(regions):
                if regions[idx,0] > regions[idx,1]:
                    c0 = regions[idx,0]
                    c1 = regions[idx,1]
                    regions[idx,0] = c1
                    regions[idx,1] = c0
            regions[:,0] -= (buffer*np.ones((regions.shape[0],))).astype('int32')
            regions[:,1] += (buffer*np.ones((regions.shape[0],))).astype('int32')
            
        return non_overlapping(regions)

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
                 region,
                 chromosome,
                 attrs = None,
                 parents = None,
                 children = None):
                         
        super(TAD,self).__init__(self,
                                 'TAD',
                                 fid,
                                 condition,
                                 attrs = attrs,
                                 parents= parents,
                                 children = children)
        self.attrs['region'] = region
        self.attrs['chromosome'] = chromosome
                         
                         
                         
'''
Gene Class
'''
class Gene(Feature):
    def __init__(self,
                 fid,
                 genebody,
                 chromosome,
                 expression,
                 promoter,
                 attrs = None,
                 parents = None,
                 children = None):
        super(Gene,self).__init__(self,
                                 'gene',
                                  fid,
                                  attrs = attrs,
                                  parents= parents,
                                  children = children)
        self.attrs['expression'] = expression
        self.attrs['region'] = genebody
        self.attrs['chromosome'] = chromosome
        self.children['promoter'] = promoter
                         
'''
Promoter Class
'''
class Promoter(Feature):
    def __init__(self,
                 fid,
                 region,
                 chromosome,
                 promoter_type,
                 gene,
                 attrs = None,
                 parents = None,
                 children = None):
        super(Gene,self).__init__(self,
                                 'promoter',
                                  fid,
                                  attrs = attrs,
                                  parents= parents,
                                  children = children)
        self.attrs['region'] = region
        self.attrs['chromosome'] = chromosome
        self.attrs['promoter_type'] = promoter_type
        self.parents['gene'] = gene                 
                         
                         
                         
                         
'''
Enhancer Class
'''
class Enhancer(Feature):
    def __init__(self,
                 fid,
                 region,
                 chromosome,
                 enhancer_type,
                 attrs = None,
                 parents = None,
                 children = None):
        super(Enhancer,self).__init__(self,
                                      'enhancer',
                                      fid,
                                      attrs = attrs,
                                      parents = parents,
                                      children = children)
        self.attrs['region'] = region
        self.attrs['chromosome'] = chromosome
        self.attrs['ehancer_type'] = enhancer_type
                         
        