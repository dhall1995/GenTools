from numba import jit
import numpy as np


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def safe_divide(x,y,fill_value = 1):
    out = np.full(x.shape, fill_value)
    
    out[y!=0] = np.divide(x[y!=0],y[y!=0])
    return out


def same_set(set1, set2):
    intersection = set1.intersection(set2)
    
    if len(list(set1)) == len(list(intersection)) and len(list(set2)) == len(list(intersection)):
        return True
    else:
        return False


def round_p_value(p):
    i = 0
    while p < 1:
        p *=10
        i += 1
        if i > 30:
            break

    return i, np.floor(100*p)/100
    

            
def idx_to_bp(idx, chr_lims, binSize, chroms):
    '''
    Utility function to convert from a bin index to basepairs. This
    assumes that chromosomes are concatenated onto each other.
    
    Arguments:
    
    - idx: The index in the concatenated array.
    - chr_lims: Dictionary detailing the start and end basepairs of each chromosome
                in the contact dictionary. NOTE: the chromosome limits are inclusive
                i.e. for each CHR_A we should have chromo_limits[CHR_A] = (start_A,
                end_A) where all basepairs b on this chromsome satisfy:
                                 start_A <= b <= end_A
    - binSize: The size of each chromatin bin.
    - chroms: A list of the chromosomes in the order they have been concatenated
              together (usually this will just be ['1', '2',..., '19', 'X']).
    
    Returns:
    - chrom: The chromosome corresponding to the input index.
    - bp: The basepair of the bin on chromosome chrom.
    
    '''
    
    ordering = {idx: chrom for idx, chrom in enumerate(chroms)}
    clens = {idx: int((chr_lims[ordering[idx]][-1] - chr_lims[ordering[idx]][0])/binSize) for idx in ordering}
    tot = 0
    for idx2 in np.arange(len(chroms)):
        tot += clens[idx2]
        if idx <= tot:
            chrom = ordering[idx2]
            chridx = idx - tot + clens[idx2]
            break
    
    bp = chr_lims[chrom][0] + chridx*binSize
    return chrom, bp           
    
        
def bp_to_idx(bp, chrom, chr_lims, binSize):
    '''
    Utility function to convert from a basepair to a bin index if 
    chromosomes are concatenated together. This assumes a 
    concatenation of in the order '1', '2', ..., '19', 'X'.
    
    Arguments:
    - bp: The input basepair.
    - chrom: The input chromosome (should be a string but can deal
             with integers).
    - chr_lims: Dictionary detailing the start and end basepairs of each chromosome
                in the contact dictionary. NOTE: the chromosome limits are inclusive
                i.e. for each CHR_A we should have chromo_limits[CHR_A] = (start_A,
                end_A) where all basepairs b on this chromsome satisfy:
                                 start_A <= b <= end_A
    - binSize: The size of each chromatin bin.
    '''
    rounded_bp = binSize*np.floor(bp/binSize)
    chr_idx = int((rounded_bp - chr_lims[chrom][0])/binSize)
    tot = 0
    if chrom != 'X':
        for i in np.arange(1,int(chrom)):
            tot += (chr_lims[str(i)][1] - chr_lims[str(i)][0])/binSize
    else:
        for i in np.arange(1,20):
            tot += (chr_lims[str(i)][1] - chr_lims[str(i)][0])/binSize
    
    return int(tot + chr_idx)



def filter_bps(bp_arr, pos_arr, bin_size):
    """
    :: [bp] -> [bp] -> [idx]
    """
    low = pos_arr[0]
    up = pos_arr[-1] + bin_size
    idxs = np.logical_and(bp_arr >= low, bp_arr < up)
    return (idxs)


@jit("float32[:](int32[:], int32[:], double)", nopython=True, nogil=True, cache=True)
def _bps_to_idx(bps, binning):

    idxs = np.empty(bps.shape[0], dtype=np.float32)
    idxs[:] = np.nan
    for i in range(bps.shape[0]):
        bp = bps[i]
        for j in range(binning.shape[0]-1):
            if (bp >= binning[j]) and (bp < binning[j+1]):
                idxs[i] = j
                break
    return (idxs)


def bp_to_idx(bp_arr, binning):
    """
    :: [bp] -> [bp] -> bp -> ([idx], [valid_idx_idx])
    Array of idxs will be float, because int can't be nan.
    Return the array of index values for the bp_arr,
    with nan in invalid entries.
    Also return a list of valid indexes.
    """
    idx_arr = _bps_to_idx(bp_arr, binning)
    valid_input_idxs = ~np.isnan(idx_arr)

    return (idx_arr.astype(np.int32), valid_input_idxs)


