
##test cython
##run 
## python setup.py build_ext --inplace
##to compile

import numpy as np
cimport numpy as np
cimport cython

np.import_array()

cdef extern from "randomkit.h":
    ctypedef struct rk_state:
        unsigned long key[624]
        int pos
        int has_gauss
        double gauss

    ctypedef enum rk_error:
        RK_NOERR = 0
        RK_ENODEV = 1
        RK_ERR_MAX = 2

    rk_error rk_randomseed(rk_state *state)
    unsigned long rk_random(rk_state *state)
    double rk_double(rk_state *state)

cdef class Uniform:
    cdef rk_state internal_state
    cdef double loc
    cdef double scale

    def __init__(self, double loc=0, double scale=1):
        cdef rk_error errcode = rk_randomseed(cython.address(self.internal_state))
        self.loc = loc
        self.scale = scale

    cdef double get(self):
        return self.loc + self.scale * rk_double(cython.address(self.internal_state))

    def __call__(self):
        return self.get()       

    def sample(self, int N, np.ndarray[double, ndim=1] out):
        cdef unsigned int i
        for i in xrange(N):
            out[i] = self.get()

cdef extern from "ndarraytypes.h":
    ctypedef enum NPY_SEARCHSIDE:
        NPY_SEARCHLEFT = 0
        NPY_SEARCHRIGHT = 1

def ProbSampleReplaceVec(int n,
                      np.ndarray[double, ndim=1] p, 
                      int size, np.ndarray[int, ndim=1] results):
    cdef int axis = 0
    cdef np.ndarray[double, ndim=1] sample_prob = np.empty(size, np.float64)
    cdef np.ndarray[double, ndim=1] cum_prob = np.empty(size, np.float64)
    #side = np.PyArray_SearchsideConverter(cum_prob, np.NPY_SEARCHLEFT)
    #cdef np.NPY_SEARCHSIDE side = np.NPY_SEARCHLEFT
    cdef NPY_SEARCHSIDE side = NPY_SEARCHRIGHT

    np.PyArray_CumSum(p, axis, np.NPY_FLOAT64, cum_prob)
    u = Uniform(loc=0.0, scale=cum_prob[-1])
    u.sample(size, sample_prob)

    results = np.PyArray_SearchSorted(cum_prob, sample_prob, side)
    return results

@cython.boundscheck(False) # turn of bounds-checking for entire function
def ProbSampleNoReplace(int n, 
                        np.ndarray[double, ndim=1] p, 
                        np.ndarray[double, ndim=1] R,
                        int nans, np.ndarray[int, ndim=1] ans):
    """ 
    weighted sampling without replacement
    This fuction is adapted from R source code src/main/random.c
    """

    cdef np.ndarray index_sorted
    #cdef np.ndarray[double, ndim=1] R
    cdef double rT, mass, totalmass
    cdef int i, j, k, n1
    cdef int axis = 0

    u = Uniform()
    #R = mr.RandomState().uniform(0.0, 1.0, nans)  #passed in from arguments
    index_sorted = np.PyArray_ArgSort(p, axis, np.NPY_QUICKSORT)
    #index_sorted = np.argsort(p)
    p = p[index_sorted]
    #revsort(p, index_sorted, n)
    totalmass = 1 #assume p array is normalized
    n1 = n - 1
    for i in xrange(nans):
        rT = totalmass * u()
        mass = 0
        for j in xrange(n1):
            mass += p[j]
            if rT <= mass:
                break
        ans[i] = index_sorted[j]
        totalmass -= p[j]
        #for k in xrange(j, n1):
            #    p[k] = p[k+1]
            #    index_sorted[k] = index_sorted[k+1]

        p[j:(n1-1)] = p[(j+1):n1]
        index_sorted[j:(n1-1)] = index_sorted[(j+1):n1]

        n1 -= 1

