from ..Feature import Feature

class Tertiary_Struct(Feature):
    '''
    Tertiary Configuration Class - i.e. a structure with just a single, unbroken string
    '''
    def __init__(self,
                 UID,
                 region,
                 contacts = None,
                 bins = None,
                 pos = None
                ):
        '''
        Really all a Tertiary Configuration needs (in the abstract) is some limits - i.e.
        the highest and lowest allowed basepairs.
        Arguments:
        name - The name of the Tertiary configuration
        lims - The limits of the Tertiary configuration (in basepairs)
        bins - If not set then the bins are assumed to be even width and 1 basepair each.
               However, since the DataTrack class allows for non-even binwidths, bins can
               also be an (N+1,) shape array where N is the number of bins we want to 
               bin our tertiary config into.
        pos - 
        '''
        
        
        if contacts is None:
            self.attrs['contacts'] = []
        else:
            self.attrs['contacts'] = contacts
            
        if bins is None:
            #default binSize of 1bp
            self.attrs['bins'] = 1
        else:
            self.attrs['bins'] = bins
            
        if pos is None:
            self.attrs['positions'] = np.empty((0,0,3))
        else:
            self.attrs['positions'] = pos
        