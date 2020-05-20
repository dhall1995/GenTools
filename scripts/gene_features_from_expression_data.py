print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__,__name__,str(__package__)))
import pandas as pd
import numpy as np
from ..GenTools.tools.Datatrack import DataTrack_rvp
from ..GenTools.tools.Feature import Promoter, Gene
from ..GenTools.utils.dtrack_utils import pairRegionsOverlap
from ..GenTools.utils.misc import save_obj

from multiprocessing import Pool

data_root = "/disk1/dh486/newprojects/gentools/tutorial_data"

#########################################################################################

chroms = [str(i+1) for i in np.arange(19)] + ['X']
chip_paths = {'H3K4me3':'{}/GSEXXXXXLaue_H3K4me3_hapmESC_peaks.npz'.format(data_root),
              'H3K27me3':'{}/GSEXXXXXLaue_H3K27me3_hapmESC_peaks.npz'.format(data_root),
              'H3K27ac':'{}/GSE56098_H3K27ac_ESC_peaks.npz'.format(data_root),
              'H3K36me3':'{}/GSEXXXXXLaue_H3K36me3_hapmESC_peaks.npz'.format(data_root)}

print("Making datatracks")
dtracks = {key: DataTrack_rvp(key).from_npz(chip_paths[key]) for key in chip_paths}

#Active promoters ############################################################################################
#H3K4me3, H3K27ac and no H3K27me3 ############################################################################
active_regions = {chrom: pairRegionsOverlap(pairRegionsOverlap(np.array(dtracks['H3K27ac'].regions[chrom]).astype('int32'), np.array(dtracks['H3K4me3'].regions[chrom]).astype('int32')),np.array(dtracks['H3K27me3'].regions[chrom]).astype('int32'),exclude = True)for chrom in chroms}
active_values = {chrom: np.ones((active_regions[chrom].shape[0],1)).astype('double') for chrom in chroms}

dtracks['active_promoters'] = DataTrack_rvp("active_promoters").from_region_value_id_dicts(active_regions, active_values)
#Bivalent promoters ##########################################################################################
#H3K27me3, H3K4me3 ###########################################################################################
bivalent_regions = {chrom: pairRegionsOverlap(np.array(dtracks['H3K27me3'].regions[chrom]).astype('int32'), np.array(dtracks['H3K4me3'].regions[chrom]).astype('int32')) for chrom in chroms}
bivalent_values = {chrom: np.ones((bivalent_regions[chrom].shape[0],1)).astype('double') for chrom in chroms}

dtracks['bivalent_promoters'] = DataTrack_rvp("bivalent_promoters").from_region_value_id_dicts(bivalent_regions, bivalent_values) 

print("Made datatracks")
# TRANSCRIPTONAL START SITES #################################################################################
TSS_file = "{}/ensembl_TSS_sites_GRCm38_p2.txt".format(data_root)
TSS = pd.read_csv(TSS_file, sep = '\t', index_col = 0)
TSS = TSS[[item in chroms for item in TSS['Chromosome Name']]]

# GENE EXPRESSION DATA #######################################################################################
expression_file = "{}/expression.bed".format(data_root)
exp = pd.read_csv(expression_file, sep = '\t', index_col = 3)
exp['chromosome'] = [item[3:] for item in exp['chromosome']]
chroms = [str(i+1) for i in np.arange(19)] + ['X']
exp = exp[[item in chroms for item in exp['chromosome']]]

# PROMOTER AND GENE OBJECTS ##################################################################################
promoters = {chrom: [] for chrom in chroms}
genes = {chrom: [] for chrom in chroms}
                   
def make_gene_and_prom_obs(ID):
    gene = TSS.loc[ID]
    
    if len(gene.values.shape)>1:
        start = gene['Gene Start (bp)'].values[0]
        end = gene['Gene End (bp)'].values[0]
        strand = gene['Strand'].values[0]
        mychrom = gene['Chromosome Name'].values[0]
        name = gene['Associated Gene Name'].values[0]
    else:
        start = gene['Gene Start (bp)']
        end = gene['Gene End (bp)']
        strand = gene['Strand']
        mychrom = gene['Chromosome Name']
        name = gene['Associated Gene Name']

    
    genebody = [start,end]
    
    if strand == 1:
        tss_reg = [[start - int(4e3), start]]
    else:
        tss_reg = [[end, end + int(4e3)]]
    
    if dtracks['active_promoters'].stats(mychrom,tss_reg)[0]>0:
        if dtracks['H3K36me3'].stats(mychrom,genebody)[0] >0:
            prom_type = 'highly_active'
        else:
            prom_type = 'active'
    elif dtracks['bivalent_promoters'].stats(mychrom,tss_reg)[0]>0:
        prom_type = 'bivalent'
    elif dtracks['H3K4me3'].stats(mychrom, tss_reg)[0]>0:
        prom_type = 'H3K3me3_only'
    else:
        prom_type = 'inactive'
    
    
    myprom = Promoter(ID+"_prom",
                      tss_reg,
                      mychrom,
                      strand,
                      "temp_gene_assignment",
                      attrs = {'promoter_type': {'naive':prom_type},
                               'associated_gene_name': name})

    mygene = Gene(ID,
                  genebody,
                  mychrom,
                  strand,
                  promoter = myprom,
                  attrs = {'name': name})
    
    expr = exp.loc[ID]
    #Xiaoyan's numbering convention again
    expression = expr.values[[3,4,5,6]]
    mygene.attrs['expression'] = {'naive': expression[0],
                                  'rexpos': expression[1],
                                  'regneg':expression[2],
                                  'primed': expression[3]}
    
    myprom.add_parent('gene', mygene)
    mygene.add_child('promoter', myprom)
                   
    return myprom, mygene, mychrom

# MULTIPROCESS THE WHOLE THING SINCE SEPARATE GENES SHOULDN'T AFFECT ONE ANOTHER ########################
myIDs = list(set(TSS.index.values))
p = Pool()
temp_outputs = p.imap(make_gene_and_prom_obs, (ID for ID in myIDs))
for temp_output in temp_outputs: 
    genes[temp_output[2]].append(temp_output[1])
    promoters[temp_output[2]].append(temp_output[0])
    
    

print("Done!")
print("Saving promoter and gene objects to tutorial_data")
out = {'genes': genes, 'promoters': promoters}
save_obj(out, data_root + 'processed/gene_promoter_info_GRCm38_p2')