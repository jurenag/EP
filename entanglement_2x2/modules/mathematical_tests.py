import numpy as np

def hermiticity_test(input_array, N):
    '''This function takes:
    - input_array: (bidimensional square (NxN) complex numpy array)
    - N: (integer) Dimension of the Hilbert space.
    
    This function checks whether input_array is the matrix representation of an hermitian operator. It returns 
    True if the input_array passed the test, and False otherwise.'''
    
    if type(input_array)!=type(np.array([])) or type(N)!=type(1):
        print('hermiticity_test(), Err1')
        return -1
    if np.shape(input_array)!=(N,N):
        print('hermiticity_test(), Err2')
        return -1
    if input_array.dtype!=np.empty((1,1),dtype=complex).dtype:
        print('hermiticity_test(), Err3')
        return -1


    result = True
    for i in range(N):
        if input_array[i,i].imag!=0.0:
            result = False
        for j in range(i):
            if input_array[i,j]!=np.conjugate(input_array[j,i]):
                result = False
    return result
    
def unity_realtrace_test(input_array, N, tolerance):
    '''This function takes:
    - input_array: (bidimensional square (NxN) complex numpy array)
    - N: (integer) Dimension of the Hilbert space.
    - tolerance: (real scalar).
    
    This function calculates the real trace, TrR(), (i.e. the sum of the real parts of the diagonal elements) of 
    the array and returns True if |1-TrR(input_array)|<tolerance. This test is meant to take place after having 
    passed hermiticity_test() test. I.e. input_array has already been checked to have pure real diagonal elements.
    
    IMPORTANT: Typically, the data received from D.M. passes the hermiticity test unconditionally, and the unity 
    trace test for tolerance>=1e-5, which is of the order of the decimal precision of the received data.'''
    
    if type(input_array)!=type(np.array([])) or type(N)!=type(1) or type(tolerance)!=type(0.0):
        print('unity_realtrace_test, Err1')
        return -1
    if np.shape(input_array)!=(N,N):
        print('unity_realtrace_test, Err2')
        return -1
    if input_array.dtype!=np.empty((1,1),dtype=complex).dtype:
        print('unity_realtrace_test, Err3')
        return -1

    trace = 0.0
    for i in range(N):
        trace = trace + input_array[i,i].real
    if abs(trace-1.0)>tolerance:
        return False
    return True