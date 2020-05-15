import numpy as np

class Feature(object):
    def __init__(self,
                 ftype,
                 fid,
                 chrom,
                 region,
                 parents = None,
                 children = None
                ):
        self.type = ftype
        self.id = fid
        self.chromomsome = chrom
        self.region = region
        if parents is None:
            self.parents = []
        else:
            self.parents = parent_ids
        
        if children is None:
            self.children = []
        else:
            self.children = children
        
        
class Feature_sc(Feature):
    def __init__(self,
                 fid,
                 chrom,
                 region,
                 cell,
                 parents = None,
                 children = None
                ):
        super(Feature_sc, self).__init__(fid, chrom,region)
        self.cell = cell