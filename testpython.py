from numpy import array, asarray, cumsum, searchsorted, clip
from numpy import in1d, count_nonzero, ndarray, unique, concatenate
from numpy.random import uniform

ismember = in1d

def probsample_replace(source_array, size, prob_array, return_index=False):
    """Unequal probability sampling; with replacement case.
    Using numpy searchsorted function, suitable for large array"""
    if not isinstance(source_array, ndarray):
        try:
            source_array = asarray(source_array)
        except:
            raise TypeError, "source_array must be of type ndarray"

    if prob_array is None:
        return sample_replace(source_array,size, return_index=return_index)

    if prob_array.sum() == 0:
        raise ValueError, "there aren't non-zero weights in prob_array"

    cum_prob = cumsum(prob_array, dtype='float64')

    sample_prob = uniform(0, cum_prob[-1], size)
    sampled_index = searchsorted(cum_prob, sample_prob)
    sampled_index = sampled_index.astype('int32')
    # due to precision problems, searchsorted could return index = cum_prob.size
    sampled_index = clip(sampled_index, 0, cum_prob.size-1) 
    
    if return_index:
        return sampled_index
    else:
        return source_array[sampled_index]

def probsample_noreplace(source_array, sample_size, prob_array=None,
                         exclude_element=None, exclude_index=None, return_index=False):
    """generate non-repeat random 1d samples from source_array of sample_size, excluding
    indices appeared in exclude_index.

    return indices to source_array if return_index is true.

    source_array - the source array to sample from
    sample_size - scalar representing the sample size
    prob_array - the array used to weight sample
    exclude_element - array representing elements should not appear in resulted array
    exclude_index - array representing indices should not appear in resulted array,
                    which can be used, for example, to exclude current choice from sampling,
                    indexed to source_array
    """
    if sample_size <= 0:
        #logger.log_warning("sample_size is %s. Nothing is sampled." % sample_size)
        if return_index:
            return array([], dtype='i')
        else:
            return array([], dtype=source_array.dtype)
            
    if prob_array is None:
        return sample_replace(source_array,sample_size, return_index=return_index)
    else:
        #make a copy of prob_array so we won't change its original value in the sampling process
        prob_array2 = prob_array.copy()
        if exclude_element is not None:
            prob_array2[ismember(source_array, exclude_element)] = 0.0
            
        if exclude_index is not None:
            index_range = arange(source_array.size, dtype="i")
            if isinstance(exclude_index, numpy.ndarray):
                exclude_index = exclude_index[ismember(exclude_index, index_range)]
                prob_array2[exclude_index] = 0.0
            elif (exclude_index in index_range):
                prob_array2[exclude_index] = 0.0
        
        nzc = count_nonzero(prob_array2)
        if nzc == 0:
            raise ValueError, "The weight array contains no non-zero elements. Check the weight used for sampling."
        if nzc < sample_size:
            logger.log_warning("The weight array contains %s non-zero elements, less than the sample_size %s. Use probsample_replace. " %
                  (nzc, sample_size))
            #sample_size = max
            return probsample_replace(source_array, sample_size, prob_array=prob_array2, return_index=return_index)
        elif nzc == sample_size:
            nonzeroindex = prob_array2.nonzero()[0]
            if return_index:
                return nonzeroindex
            else:
                return source_array[nonzeroindex]

    to_be_sampled = sample_size
    sampled_index = array([], dtype='i')  #initialize sampled_index
    while True:
        proposed_index = probsample_replace(source_array, to_be_sampled, prob_array2, return_index=True)
        valid_index = unique(proposed_index, return_index=False)
        #assert all( logical_not(ismember(valid_index, sampled_index)) )
        #valid_index = valid_index[logical_not(ismember(valid_index, sampled_index))]  #this should not be necessary because we control the prob_array        
        sampled_index = concatenate((sampled_index, valid_index))
        
        to_be_sampled -= valid_index.size
        if to_be_sampled == 0:
            if return_index:
                return sampled_index
            else:
                return source_array[sampled_index]

        prob_array2[proposed_index] = 0.0
        nzc = count_nonzero(prob_array2)
        assert nzc > 0 #given that we have checked and made sure nonzerocounts(prob_array2)>size, it should not run out before we have enough non-repeat samples

def prob2dsample(
                 source_array,
                 nagents, 
                 nalts, 
                 prob_array, 
                 results,
                 exclude_index=None, 
                 replace=False
                ):

    for i in xrange(nagents):
        results[i, :] = probsample_noreplace(source_array,
                                             nalts,
                                             prob_array=prob_array,
                                             return_index=True

                                            )

