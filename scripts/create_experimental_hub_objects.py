print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__,__name__,str(__package__)))
import pandas as pd
import numpy as np
from ..GenTools.tools.Datatrack import DataTrack_rvp
from ..GenTools.tools.Feature import Experimental_Hub
from ..GenTools.utils.misc import save_obj

from multiprocessing import Pool
from functools import partial

data_root = "/disk1/dh486/newprojects/gentools/tutorial_data"
print("Analysing input file...")
x = pd.read_csv("/disk1/dh486/Data/misc/hubs/hubs.bed".format(data_root),
                sep = "\t",
                header = None, 
                names = ['chr','a_start','a_end','c_start','c_end','cell_idx','cell_number','tpoint'])
x['chr'] = [item[3:] for item in x['chr']]
    
tpoints = {1:'naive',3: 'rexpos',2: 'rexneg',4:'primed'} #Xiaoyan's timepoint convention 1: 0hr, 2: Rexlow, 3: Rexhi, 4: 48hrs
cell_num_ids = {1:{1:'Cell_1_ambig',2:'Cell_2_ambig',3:'Cell_4_ambig',
                    4:'Cell_5_ambig',5:'Cell_6_ambig',6:'P44F12',7:'P44F6',
                    8:'P44H4'},
                3:{1:'P62G7',2:'P62G8',3:'P62H10',4:'P62H13',5:'P62E12',
                    6:'P62E6',7:'P62F11'},
                2:{1:'P63E9',2:'P63F8',3:'P63G10',4:'P63H10',5:'P63H9',
                    6:'P64E11',7:'P63E14',8:'P63G12',9:'P63H14',10:'P63H7',
                    11:'P64E5'},
                4:{1:'P45F10',2:'P46D12',3:'P46G10',4:'P54E14',5:'P54F7',
                    6:'P54G11',7:'P54G12',8:'P54G13',9:'P46D6',
                    10:'P54H12'}
               }

anchor_idxs = {}
for tpoint in tpoints:
    y = x[x['tpoint'] == tpoint]
    anchor_idxs[tpoints[tpoint]] = list(set(y['cell_idx'].values)) 
    
def make_experimental_hubs(tpoint, idx):
    outhubs = []
    y = x[x['tpoint'] == tpoint]
    hubs = y[y['cell_idx'] == idx]
    cells = list(set(hubs['cell_number'].values))
    for cell in cells:
        cell_hub = hubs[hubs['cell_number'] == cell]
        chrom = cell_hub['chr'].values[0]
        anchor_contact_regions = cell_hub[['a_start', 'a_end', 'c_start', 'c_end']].values
        outhubs.append(Experimental_Hub(tpoints[tpoint], 
                                        cell_num_ids[tpoint][cell],
                                        chrom,
                                        anchor_contact_regions,
                                        idx))
    return outhubs

print("Creating hub objects...")
chroms = [str(i+1) for i in np.arange(19)] + ['X']
hub_dict = {chrom: [] for chrom in chroms}
for tpoint in tpoints:
    fn = partial(make_experimental_hubs, tpoint)
    p = Pool()
    temp_outputs = p.imap(fn, (idx for idx in anchor_idxs[tpoints[tpoint]]))
    for temp_output in temp_outputs:
        for item in temp_output:
            chrom = item.attrs['chromosome']
            hub_dict[chrom].append(item)

print("Saving to processed tutorial data...")
save_obj(hub_dict, data_root + '/processed/experimental_hubs')