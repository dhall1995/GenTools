'''
TO DO:
    - Write datatrack_to_bed function
    - Fix datatrack_to_npz function
    - Write bed_to_npz function
'''
from scipy.sparse import coo_matrix
from scipy import sparse
import torch
import pandas as pd
import math
import numpy as np
from numpy import int32

CHR_KEY_SEP = ' '

###############################################################################

def load_npz_contacts(file_path, 
        store_sparse=False,
        display_counts=False,
        normalize = False,
        cut_centromeres = True
        ):
    '''
    Utility function to load a .npz file containing contact information from a Hi-C experiment. 
    
    Arguments:
    
    - file_path: A .npz file generated using the nuc_tools ncc_bin tool. The function assumes a
                 File of this format
    - store_sparse: Boolean determining whether to return the contact matrices in sparse format
    - display_counts: Boolean determining whether to display summary plots of Hi-C counts
    - normalize: Boolean determining whether to normalise all matrix elements to lie between zero or one.
                 If False then raw contact counts are returned instead
    - cut_centromeres: Boolean determining whether to cut out the centromeres from the beginning of each
                       chromosome. Since the centromeres contain repetitive elements, they can't currently
                       be mapped by Hi-C so these rows and columns should be void of Hi-C contacts. This 
                       does affect indexing later on but other functions in this package should accommodate
                       for that
                       
    Returns:
    
    - bin_size: The size of each chromatin bin in basepairs.
    - chromo_limits: Dictionary detailing the start and end basepairs of each chromosome
                     in the contact dictionary. NOTE: the chromosome limits are inclusive
                     i.e. for each CHR_A we should have chromo_limits[CHR_A] = (start_A,
                     end_A) where all basepairs b on this chromsome satisfy:
                                 start_A <= b <= end_A
    - contacts: Dictionary of matrices detailing contacts between chromosome pairs
    '''
    file_dict = np.load(file_path, allow_pickle=True, encoding = 'bytes')
  
    chromo_limits = {}
    contacts = {}
    bin_size, min_bins = file_dict['params']
    bin_size = int(bin_size*1e3)
  
    chromo_hists = {}
    cis_chromo_hists = {}

    pair_keys = [key for key in file_dict.keys() if "cdata" in key]
    nonpair_keys = [key for key in file_dict.keys() if (CHR_KEY_SEP not in key) and (key != 'params')]
  
    for key in nonpair_keys:
        offset, count = file_dict[key]
        chromo_limits[key] = offset*bin_size, (offset+count)*bin_size
        chromo_hists[key] = np.zeros(count)
        cis_chromo_hists[key] = np.zeros(count)

    maxc = 1
    if normalize:
        for key in sorted(pair_keys):
            maxc = np.maximum(maxc, np.max(file_dict[key]))

    for key in sorted(pair_keys):
        chr_a, chr_b, _ = key.split(CHR_KEY_SEP)
        shape = file_dict[chr_a + CHR_KEY_SEP + chr_b + CHR_KEY_SEP + "shape"]
        mtype = "CSR"
        try:
            indices = file_dict[chr_a + CHR_KEY_SEP + chr_b + CHR_KEY_SEP + "ind"]
            indptr = file_dict[chr_a + CHR_KEY_SEP + chr_b + CHR_KEY_SEP + "indptr"]
        except:
            mtype = "COO"
            row = file_dict[chr_a + CHR_KEY_SEP + chr_b + CHR_KEY_SEP + "row"]
            col = file_dict[chr_a + CHR_KEY_SEP + chr_b + CHR_KEY_SEP + "col"]

        if mtype == "CSR":

            mat = sparse.csr_matrix((file_dict[key]/maxc, indices, indptr), shape = shape)
        else:
            mat = sparse.coo_matrix((file_dict[key]/maxc, (row, col)), shape = shape)

        if not store_sparse:
            mat = mat.toarray()
          
        if chr_a == chr_b:
            a, b = mat.shape
            cols = np.arange(a-1)
            rows = cols-1

            if not np.all(mat[rows, cols] == mat[cols, rows]): # Not symmetric
                mat += mat.T
          
        contacts[(chr_a, chr_b)] = mat  
     
    #Chromosomes in our dataset
    chroms = chromo_limits.keys()
    if cut_centromeres:
    #Exclude centromeres of chromosomes where we don't have any contact data
        for chrom in chroms:
            chrmax = chromo_limits[chrom][-1]
            temp = contacts[(chrom, chrom)].indices
            chromo_limits[chrom] = (bin_size*np.min(temp[temp>0]), chrmax)
    
    for pair in contacts:
        s0, s1 = int(chromo_limits[pair[0]][0]/bin_size), int(chromo_limits[pair[1]][0]/bin_size)
        try:
            contacts[pair] = contacts[pair][s0:,s1:]
        except:
            contacts[pair] = contacts[pair].tocsr()[s0:, s1:].tocoo()
  
    if display_counts:
        # A simple 1D overview of count densities
 
        from matplotlib import pyplot as plt

        for chr_a, chr_b in contacts:
            mat = contacts[(chr_a, chr_b)]
            chromo_hists[chr_a] += mat.sum(axis=1)
            chromo_hists[chr_b] += mat.sum(axis=0)
 
            if chr_a == chr_b:
                cis_chromo_hists[chr_a] += mat.sum(axis=1)
                cis_chromo_hists[chr_b] += mat.sum(axis=0)
    
        all_sums = np.concatenate([chromo_hists[ch] for ch in chromo_hists])
        cis_sums = np.concatenate([cis_chromo_hists[ch] for ch in chromo_hists])
 
        fig, ax = plt.subplots()
 
        hist, edges = np.histogram(all_sums, bins=25, normed=False, range=(0, 500))
        ax.plot(edges[1:], hist, color='#0080FF', alpha=0.5, label='Whole genome (median=%d)' % np.median(all_sums))

        hist, edges = np.histogram(cis_sums, bins=25, normed=False, range=(0, 500))
        ax.plot(edges[1:], hist, color='#FF4000', alpha=0.5, label='Intra-chromo/contig (median=%d)' % np.median(cis_sums))
 
        ax.set_xlabel('Number of Hi-C RE fragment ends (%d kb region)' % (bin_size/1e3))
        ax.set_ylabel('Count')
 
        ax.legend()
 
        plt.show()

  
    return bin_size, chromo_limits, contacts


##################################################################################

def save_contacts(out_file_path, matrix_dict, chromo_limits, bin_size, min_bins=0):
    '''
    Save Hi-C a Hi-C contact dictionary to a .npz file. 
    
    Arguments:
    
    - out_file_path: Path to save the contact dictionary to
    - matrix_dict: Dictionary of Hi-C contacts. Dictionary should have keys of the form
                   (CHR_A, CHR_B). Trans contact matrices should be stored in sparse COO
                   format while cis contact matrices should be stored in sparse CSR
                   format.
    - chromo_limits: Dictionary detailing the start and end basepairs of each chromosome
                     in the contact dictionary. NOTE: the chromosome limits are inclusive
                     i.e. for each CHR_A we should have chromo_limits[CHR_A] = (start_A,
                     end_A) where all basepairs b on this chromsome satisfy:
                                 start_A <= b <= end_A
    - bin_size: What is the size of each contact matrix bin in basepairs e.g. 50000
    - min_bins: Minimum number of bins to be included in a contig (read: chromosome) for it
                to be used downstream. 
    
    '''
    contacts = {}
    kb_bin_size = int(bin_size/1e3)
  
    for chr_a, chr_b in matrix_dict:
        pair = chr_a, chr_b
        key = CHR_KEY_SEP.join(pair)
  
        if chr_a == chr_b:
            contacts[key] = sparse.csr_matrix(matrix_dict[pair])
        else:
            contacts[key] = sparse.coo_matrix(matrix_dict[pair])
    
        start_a, end_a = chromo_limits[chr_a]
        start_b, end_b = chromo_limits[chr_b]
    
        min_a = int(start_a/bin_size)
        num_a = int(math.ceil(end_a/bin_size)) - min_a
        min_b = int(start_b/bin_size)
        num_b = int(math.ceil(end_b/bin_size)) - min_b
    
        # Store bin offsets and spans
        contacts[chr_a] = np.array([min_a, num_a])
        contacts[chr_b] = np.array([min_b, num_b])
    
        contacts['params'] = np.array([kb_bin_size, min_bins])    
  
    np.savez_compressed(out_file_path, **contacts) 
    

###################################################################
def rvps_from_npz(file_path,
                  ID = False,
                  values_key = 'values',
                  params = False):
    """Load track data (e.g. ChIp-seq)from Numpy archive (.npz)
    
    Arguments:
    
    - file_path: Path of the data track to be loaded
    - ID: Boolean. Determines if each datatrack region contains a unique ID.
          For example, if the datatrack were genes or transcription then 
          each datatrack region is a specific gene with a specific ensemble
          ID.
    - values_key: Specifies which key to use within the .npz archive as our
                  datatrack values. If values_key doesn't exist in the
                  archive then the datatrack values are set to 1
    - params: Boolean. If true then search the data archive for a 'params'
              key and return that. The params dictionary is used to specify
              any default parameters to be used when binning a datatrack
          
    Returns:
    
    - regions_dict: Dictionary containing chromosomes as keys. Each key
                    value is an (N_CHR,2) shape array where each row is a
                    region and N_CHR is the number of non-zero datatrack
                    regions on that chromsome
    - values_dict: Dictionary containing chromosomes as keys. Each key 
                   value is an (N_CHR,) shape array detailing the
                   datatrack value for each non-zero datatrack region
    - ID_dict: If ID is True, returns a dictionary detailing the unique
               ID for each datatrack region.
    - params: If params is true, try to return the params dictionary from
              the data archive. If 'params' is not a key in archive then
              return an empty dictionary.
    """
    
    if params:
        data_archive = dict(np.load(file_path, allow_pickle = True))
        try:
            return data_archive['params'][()]
        except:
            print("Couldn't extract binning parameters from the file. ")
            return {}
    
    data_archive = np.load(file_path, allow_pickle = True)
    regions_dict = {}
    values_dict = {}
    ID_dict = {}
    params = {}
    
    chromosomes = [str(i+1) for i in np.arange(19)] + ['X']
    for key in data_archive:
        if key != 'params':
            null, key2, track_name, chromo = key.split('/')
        
            if key2 == 'regions':
                regions_dict[chromo] = data_archive[key].astype('int32')
            elif key2 == values_key:
                try:
                    values_dict[chromo] = data_archive[key].astype('float')
                except:
                    reg_key = "/".join([null, 'regions', track_name, chromo])
                    num_regs = data_archive[reg_key].astype('int32').shape[0]
                    values_dict[chromo] = np.zeros((num_regs,1)).astype('float')        
            elif ID and key2 == 'id':
                ID_dict[chromo] = data_archive[key]
    

    return regions_dict, values_dict, ID_dict

###################################################################
def rvps_from_bed(file_path,
                  chrom_col = 0,
                  region_cols = (1,2),
                  value_col = None,
                  ID_col = None,
                  value_fill = 1,
                  header = None,
                  allowed_chroms = None,
                  sep = "\t"):
    """Load track data (e.g. ChIp-seq)from Numpy archive (.npz)
    
    Arguments:
    
    - file_path: Path of the data track to be loaded (bed format)
    - chrom_col: int. Column of the bed file containing the chromosome information
    - region_cols: 2-tuple. Columns of the bed file containing the regions for
                   each value.
    - value_col: int. Column of the bed file containing the value for each region.
                 If this is None then each region is given a score given by the
                 value_fill input argument.
    - ID_col: int. If each region has a specific ID associated with it then
              this is stored in an ID dictionary along with the regions
    - value_fill: float. If value_col is None then we give each region a value
                  according to the value_fill input.
    - header: None. If the bed file has a header then we ignore line 0 and 
              skip to line 1.
    - allowed_chroms: List of chromosomes which we want to include in our datatrack dictionaries.
                        if None then all chromosomes are allowed. 
    - sep: Separating values in the bed file.
                  
          
    Returns:
    
    - regions_dict: Dictionary containing chromosomes as keys. Each key
                    value is an (N_CHR,2) shape array where each row is a
                    region and N_CHR is the number of non-zero datatrack
                    regions on that chromsome
    - values_dict: Dictionary containing chromosomes as keys. Each key 
                   value is an (N_CHR,) shape array detailing the
                   datatrack value for each non-zero datatrack region
    - ID_dict: If ID is True, returns a dictionary detailing the unique
               ID for each datatrack region.
    """
    
    
    x = pd.read_csv(file_path, sep = sep, header = header)
    
    if allowed_chroms is None:
        allowed_chroms = list(set(x[chrom_col].values))
        for idx, item in enumerate(allowed_chroms):
            #use the chromosome naming convention that chromosomes don't start with chr
            if "chr" in item:
                allowed_chroms[idx] = item[3:]
        
    regions_dict = {}
    values_dict = {}
    ID_dict = {}
    for idx in np.arange(x.values.shape[0]):
        chrom = x.loc[idx][0]
        if "chr" in chrom:
            chrom = chrom[3:]
        if chrom not in allowed_chroms:
            continue
            
        start = x.loc[idx][region_cols[0]]
        end = x.loc[idx][region_cols[1]]
        if value_col is not None:
            val = x.loc[idx][value_col]
        else:
            val = value_fill
            
        if ID_col is not None:
            ID = x.loc[idx][ID_col]
    
        if chrom not in regions_dict:
            regions_dict[chrom] = [[start, end]]
            values_dict[chrom] = [[val]]
            if ID_col is not None:
                ID_dict[chrom] = [[ID]]
        else:
            regions_dict[chrom].append([start, end])
            values_dict[chrom].append([val])
            if ID_col is not None:
                ID_dict[chrom].append([ID])
        
    for key in regions_dict:
        regions_dict[key] = np.array(regions_dict[key])
        values_dict[key] = np.array(values_dict[key])
        if ID_col is not None:
            ID_dict[key] = np.array(ID_dict[key])
            
    return regions_dict, values_dict, ID_dict
    
###################################################################
def rvps_to_npz(regions,
                values,
                track_name,
                out_path,
                IDs = None,
                params = None
               ):
    """Save track data (e.g. ChIp-seq) to  Numpy archive (.npz)
    Arguments:
    - regions: dictionary detialing an (N_chrom,2) shape array detailing the
               regions of the datatrack for each chromosome.
    - values: dictionary detialing an (N_chrom,) shape array detailing the values
              associated with each region.
    - track_name: Descriptive name of the datatrack.
    - out_path: The path to save the .npz archive to.
    - IDs: If each regions is associated with a unique ID then save these IDs
           in a dictionary with an (N_chrom,) shape array for each chromosome.
    """
   
    outdict = {}
    
    for chrom in regions:    
        key1 = "dtrack/regions/{}/{}".format(track_name, chrom)
        key2 = "dtrack/values/{}/{}".format(track_name, chrom)
        if IDs is not None:
            key3 = "dtrack/id/{}/{}".format(track_name, chrom)
            
        outdict[key1] = regions[chrom]
        if chrom in values:
            outdict[key2] = values[chrom]
        else:
            print("Couldn't find chromosome {} in values. Assuming ones instead".format(chrom))
            outdict[key2] = np.ones(regions[chrom].shape(0))
        if IDs is not None:
            outdict[key3] = IDs
    
    if params is not None:
        outdict['params'] = params
        
    np.savez(out_path, **outdict, allow_pickle = True)
    
###################################################################
def rvps_to_bed(regions,
                values,
                track_name,
                out_path,
                IDs = None,
                sep = "\t"
               ):
    """Save track data (e.g. ChIp-seq) to  Numpy archive (.npz)
    Arguments:
    - regions: dictionary detialing an (N_chrom,2) shape array detailing the
               regions of the datatrack for each chromosome.
    - values: dictionary detialing an (N_chrom,) shape array detailing the values
              associated with each region.
    - track_name: Descriptive name of the datatrack.
    - out_path: The path to save the .npz archive to.
    - IDs: If each regions is associated with a unique ID then save these IDs
           in a dictionary with an (N_chrom,) shape array for each chromosome.
    """
    if IDs is None:
        IDs = {}
        
    ID_counter = 0
    with open(out_path,'w') as op:
        for chromosome in regions:
            for idx in np.arange(regions[chromosome].shape[0]):
                region = regions[chromosome][idx,:]
                value = values[chromosome][idx]
                try:
                    ID = IDs[chromosome][idx]
                except:
                    ID = "chr{}_{}_{}".format(chromosome,track_name, ID_counter)
                    ID_counter += 1
                
                op.write(sep.join([chromosome, region[0], region[1],value, ID]))
                op.write("\n")
                
        
###################################################################
def bed_to_npz(bed_path,
               out_path,
               track_name,
               chrom_col = 0,
               region_cols = (1,2),
               value_col = None,
               ID_col = None,
               value_fill = 1,
               header = None,
               allowed_chroms = None,
               sep = "\t",
               params = None):
    '''
    convert a BED format region-value-pairs input to a .npz archive
    '''
    regions_dict, values_dict, ID_dict = rvps_from_bed(file_path,
                                                      chrom_col = chrom_col,
                                                      region_cols = region_cols,
                                                      value_col = value_col,
                                                      ID_col = ID_col,
                                                      value_fill = value_fill,
                                                      header = header,
                                                      allowed_chroms = allowed_chroms,
                                                      sep = sep)
    if len(list(ID_dict.keys())) == 0:
        ID_dict = None
    
    rvps_to_npz(regions_dict,
                values_dict,
                track_name,
                out_path,
                IDs = ID_dict,
                params = params)
    
######################################################################
def process_ncc(ncc_file,
                chroms = None,
                contact_region = None
               ):
    if chroms is None:
        chroms = [str(i+1) for i in np.arange(19)] + ['X']
    
    c_dict = {chroms[idx]: {chroms[jdx]: [] for jdx in np.arange(idx, len(chroms))} for idx in np.arange(len(chroms))}
    
    with open(ncc_file, 'r') as ncc:
        for line in ncc:
            linelist = line.split(" ")
            chrom1, start1, end1, chrom2, start2, end2 = linelist[0], linelist[1], linelist[2], linelist[6],linelist[7],linelist[8]
            chrom1 = chrom1[3:]
            chrom2 = chrom2[3:]
            
            start1, start2, end1, end2 = int(start1), int(start2), int(end1), int(end2)
            
            if start1 > end1 and chrom1 == chrom2:
                start1, end1 = end1, start1
            if start2 > end2 and chrom1 == chrom2:
                start2, end2 = end2, start2
                
            pos1 = (0.5)*(int(start1) + int(end1)) 
            pos2 = (0.5)*(int(start2) + int(end2))
            
            if chrom1 not in chroms:
                continue
            if chrom2 not in chroms:
                continue
                
            test_overlaps = dtu.non_overlapping(np.array([[start1, end1], [start2, end2]]).astype('int32'))
            if contact_region is not None:
                contact_region = dtu.non_overlapping(contact_region)
                test_overlaps = dtu.pairRegionsOverlap(test_overlaps, contact_region)
            
            
            if len(test_overlaps) == 0 and chrom1 == chrom2:
                continue
            
                
            if (chrom1 != 'X')&(chrom2 != 'X'):
                chrom1 = int(chrom1)
                chrom2 = int(chrom2)
                if chrom1 > chrom2:
                    chrom1, chrom2 = str(chrom2), str(chrom1)
                    pos1, pos2 = pos2, pos1
            elif chrom1 == 'X':
                chrom1, chrom2 = chrom2, chrom1
                pos1, pos2 = pos2, pos1
            
            chrom1 = str(chrom1)
            chrom2 = str(chrom2)
            c_dict[chrom1][chrom2].append([pos1, pos2])
            
    for chrom1 in c_dict:
        for chrom2 in c_dict[chrom1]:
            c_dict[chrom1][chrom2] = np.array(c_dict[chrom1][chrom2]).astype('int32')
    
    return c_dict
    

######################################################################
import pickle
def save_obj(obj, out_path):
    with open(out_path +'.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(in_path):
    with open(in_path + '.pkl', 'rb') as f:
        return pickle.load(f)