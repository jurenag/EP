import numpy as np
from test_read_and_format import file_to_numpyarray_test, reformat_density_matrices, realArray_to_complexArray
from mathematical_tests import hermiticity_test, unity_realtrace_test
    
def test_every_density_matrix(density_matrices, N, tolerance):
    '''This function takes:
    - density_matrices: (tridimensional real numpy array) Collection of density matrices which are formatted
    as they are returned from reformat_density_matrices(), i.e. it is a tridimensional numpy array which 
    represents a collection of real arrays (Nx2N). Iterating through the first index gives different density
    matrices.
    - N: (integer) Dimension of the Hilbert space.
    - tolerance: (real array) It is passed to unity_realtrace_test as tolerance.
    
    This function makes every density matrix take hermiticity_test() and unity_real_trace_test() and returns 
    (True, []) (i.e. True and an empty list) if every matrix passed both tests. If any of the density matrices
    failed one test, then the function returns (False, [i1,i2,i3,...]), where i1, i2, i3,... are
    indices of the matrices which failed at least one test.'''
    
    if type(density_matrices)!=type(np.array([])) or type(N)!=type(1) or type(tolerance)!=type(0.0):
        print('test_every_density_matrix(), Err1')
        return -1
    if density_matrices.ndim!=3:
        print('test_every_density_matrix(), Err2')
        return -1
    if np.shape(density_matrices)[1]!=N or np.shape(density_matrices)[2]!=2*N:
        print('test_every_density_matrix(), Err3')
        return -1
    if density_matrices.dtype!=np.empty((1,1), dtype=float).dtype:
        print('test_every_density_matrix(), Err4')
        return -1

    patological_cases = []
    number_of_matrices = np.shape(density_matrices)[0]
    #WARNING: IF YOU ARE GOING TO USE realArray_to_complexArray TO GENERATE A CERTAIN ARRAY, SAY aux, YOU MUST
    #INITIALIZE IT AS dtype=complex. OTHERWISE THE IMAGINARY PARTS ARE DISCARDED!
    aux = np.empty((np.shape(density_matrices)[1],np.shape(density_matrices)[2]), dtype=complex)
    overall_result = True
    for k in range(number_of_matrices):
        aux = realArray_to_complexArray(density_matrices[k,:,:], N)
        if hermiticity_test(aux, N)==False or unity_realtrace_test(aux, N, tolerance)==False:
            patological_cases.append(k)
            overall_result = False
    return overall_result, np.array(patological_cases)
    
def test_every_matrix_file(filepaths, N, rows, cols, tolerance, separator='\t', skipped_rows=0):
    '''This function takes:
    - filepaths: (list of strings) Each one of such strings is the path to a file which contains 
    a set of density matrices which should be formatted as they are read by reformat_density_matrices(),
    i.e. one density matrix per row of the data file.
    - N: (integer) Dimension of the Hilbert. It is given to file_to_numpyarray_test() as N. 
    For more info. on this paramter, check reformat_density_matrices() documentation.
    - rows (resp. cols): (integer) Number of rows (resp. cols) of the unfolded density matrix.
    It is given to reformat_density_matrices() as rows (columns). For more info. on this paramter,
    check reformat_density_matrices() documentation.
    - tolerance: (real scalar) It is given to test_every_density_matrix() as tolerance. This parameter
    is eventually used to check that the trace of the density matrix is equal to 1 within some tolerance.
    
    This function applies test_every_density_matrix() to each density matrices collection (they are stored in each
    file whose path is contained in filepaths) and displays the results of each test.'''
    
    if type(filepaths)!=type([]) or type(N)!=type(1) or type(rows)!=type(1) or type(cols)!=type(1) or type(tolerance)!=type(0.0):
        print('test_every_matrix_file(), Err1')
        return -1
    if N<1 or rows<1 or cols<1 or tolerance<0.0:
        print('test_every_matrix_file(), Err2')
        return -1

    result = True
    for i in range(len(filepaths)):
        print('Testing density matrices within: ', filepaths[i])
        aux_array = file_to_numpyarray_test(filepaths[i], N, separator=separator, rows_to_skip=skipped_rows)
        print('\t -> Testing '+str(np.shape(aux_array)[0])+' matrices')
        aux_array = reformat_density_matrices(aux_array, rows, cols)
        aux_result, _ = test_every_density_matrix(aux_array, rows, tolerance)
        if aux_result == False:
            print('\t -> Test failed')
            return False
        print('\t -> Test passed')
    return True   