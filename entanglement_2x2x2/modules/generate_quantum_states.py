import numpy as np
from numpy.random import Generator, PCG64
from scipy import constants as sp
import copy
import math




def generate_one_qubit_state(rng, add_global_phase=True):
    '''This function takes:
    - rng: (a random generator, as initialized by rng = Generator(PCG64(int(seed))). Here, seed is some integer.
    - add_global_phase: (boolean) Whether to add a global phase exp(i*\delta) to the one qubit state.

    This function returns a unidimensional complex numpy array with two complex entries, corresponding to a random 
    one qubit state according to \ket{\psi} = e^{i\delta}(cos(\theta/2)\ket{0}+e^{i\phi}*sin(\theta/2)\ket{1}), 
    where \theta\in[0,\pi], and \phi\in[0,2\pi[. \delta is different from zero only if add_global_phase==True.
    
    This function has been checked to be working as expected.'''

    theta = sp.pi*rng.random()
    phi = 2*sp.pi*rng.random()
    q_state = np.array([np.cos(theta/2.0),np.exp(1.j*phi)*np.sin(theta/2.0)])

    if add_global_phase==True:
        delta = 2*sp.pi*rng.random()
        return np.exp(1.j*delta)*q_state
    else:
        return q_state




def generate_three_qubit_separable_state(rng, add_global_phase=True):
    '''This function takes:
    - rng: (a random generator, as initialized by rng = Generator(PCG64(int(seed))). Here, seed is some integer.
    - add_global_phase: (boolean) This parameter is passed to generate_one_qubit_state() as add_global_phase in 
    one of the three calls that this function makes to generate_one_qubit_state().

    This function returns a unidimensional complex numpy array with eight complex entries, corresponding to a
    random separable three qubits state, according to the product state of three qubit states each of which is
    generated be means of generate_one_qubit_state(). The coordinates of the three qubit state in the product
    space basis, i.e. {|000>,|001>,...,|111>}, is computed using numpy.kron(a,b), which is a numpy function that
    computes the krönecker product of two arrays, a and b. This function operates in such a way that the product
    state is given in the most intuitive way to order the H2xH2xH2 basis, i.e.
    {|000>,|001>,|010>,|011>,|100>,|101>,|110>,|111>}, so that the i-th element of the returned unidimensional
    numpy array matches the coordinate of such quantum state along the axis spanned by the i-th element in the 
    fixed basis.

    This function has been checked to be working as expected.
    '''

    aux = np.kron(generate_one_qubit_state(rng, add_global_phase=False), generate_one_qubit_state(rng, add_global_phase=False))
    return np.kron(aux, generate_one_qubit_state(rng, add_global_phase=add_global_phase))




def negativity(hermitian_matrix, N, transpose_first=True):
    '''This function takes:
    - hermitian_matrix: (bidimensional square complex array) The matrix representation of an hermitian operator. 
    (Remember that such operators have real eigenvalues.)
    - N: (integer) Dimension of the tensor product space.
    
    This function returns the negativity of hermitian_matrix, i.e. the sum of the absolute values of its partial 
    transpose negative eigenvalues.
    
    WARNING: Since we call partial_transpose() within the body of this function, this function is still only
    applicable to the case of H2xH2, i.e. N=4.'''
    
    if type(hermitian_matrix)!=type(np.array([])) or np.ndim(hermitian_matrix)!=2 or type(N)!=type(1):
        print('negativity(), Err1')
    if np.shape(hermitian_matrix)[0]!=np.shape(hermitian_matrix)[1]:
        print('negativity(), Err2')
        
    aux = copy.deepcopy(hermitian_matrix)
    aux = partial_transpose(aux, N, transpose_first_subsystem=transpose_first)
    eigenvalues, _ = np.linalg.eigh(aux, UPLO='L')
    negativity = 0.0
    for i in range(np.shape(eigenvalues)[0]):
        if eigenvalues[i]<0.0:
            negativity += np.abs(eigenvalues[i])
    return negativity




def commuter(a, b):
    return b, a




def partial_transpose(sqc_matrix, N, transpose_first_subsystem=True):
    '''This function takes:
    - sqc_matrix: (bidimensional square complex numpy array) Matrix representation of the density matrix
    - N: (integer) Dimension of the tensor product space
    - transpose_first_subsystem: (boolean) Determines whether the partial transposition is performed over the
    first subsystem or not.

    This function returns its partial transpose (just in case the state is actually separable. Otherwise, the 
    partial transposition is not well defined and the function returns a matrix which has nothing to do with 
    partial transposition, therefore losing the positivity which characterizes a density matrix).
    
    WARNING: THIS FUNCTION ONLY WORKS FOR H2XH2, I.E. THE TENSOR PRODUCT OF TWO QUBITS.'''
    
    if type(sqc_matrix)!=type(np.array([])) or np.ndim(sqc_matrix)!=2 or type(N)!=type(1):
        print('partial_transpose(), Err1')
        return -1
    if np.shape(sqc_matrix)[0]!=N or np.shape(sqc_matrix)[1]!=N:
        print('partial_transpose(), Err2')
        return -1
    
    if transpose_first_subsystem==True:
        sqc_matrix[0,2], sqc_matrix[2,0] = commuter(sqc_matrix[0,2], sqc_matrix[2,0])
        sqc_matrix[0,3], sqc_matrix[2,1] = commuter(sqc_matrix[0,3], sqc_matrix[2,1])
        sqc_matrix[1,2], sqc_matrix[3,0] = commuter(sqc_matrix[1,2], sqc_matrix[3,0])
        sqc_matrix[1,3], sqc_matrix[3,1] = commuter(sqc_matrix[1,3], sqc_matrix[3,1])
    else:
        sqc_matrix[0,1], sqc_matrix[1,0] = commuter(sqc_matrix[0,1], sqc_matrix[1,0])
        sqc_matrix[0,3], sqc_matrix[1,2] = commuter(sqc_matrix[0,3], sqc_matrix[1,2])
        sqc_matrix[2,1], sqc_matrix[3,0] = commuter(sqc_matrix[2,1], sqc_matrix[3,0])
        sqc_matrix[2,3], sqc_matrix[3,2] = commuter(sqc_matrix[2,3], sqc_matrix[3,2])
    return sqc_matrix




def generate_three_qubit_bipartitely_entangled(rng, add_global_phase=True, separable_qubit_label=None):
    '''This function takes:
    - rng: (a random generator, as initialized by rng = Generator(PCG64(int(seed))). Here, seed is some integer.
    - add_global_phase: (boolean) This parameter is passed to generate_one_qubit_state() as add_global_phase in 
    when generating the qubit state that is not entangled with the rest of the two qubits.
    - separable_qubit_label: (1, 2 or 3) The label of the qubit to be disentagled from the rest. If not specified, 
    i.e. separable_qubit_label==None, then such vale is taken randomly in (1,2,3).
    
    This function returns a random bipartitely entangled pure quantum H2xH2xH2 state. The state is generated 
    according to the previously explained algorithm. The label of the qubit which is disentangled from the 
    rest of qubits (i.e. 1, 2 or 3) is assessed randomly.

    This function has been checked to be working as expected.
    
    NOTE_1: This function does not exploits the symmetry condition induced by the hermiticity of the DMs.
    '''

    found_two_qubits_entangled_candidate = False
    while found_two_qubits_entangled_candidate == False:
        modules = rng.random((4,))
        phases = rng.random((4,))
        phases = 2*sp.pi*phases

        density_matrix = np.empty((4,4), dtype=complex)
        for i in range(4):
            for k in range(4):
                density_matrix[i,k] = modules[i]*modules[k]*np.exp(1.j*(phases[i]-phases[k]))
        #Compute density_matrix normalization
        norm = 0.0
        for i in range(4):
            norm += np.power(modules[i], 2)
        #Normalize the density matrix
        density_matrix = density_matrix/norm
        #Assess whether it is actually entangled
        if negativity(density_matrix, 4)>0:
            found_two_qubits_entangled_candidate = True

    #Generate the state vector of the entangled two qubits state
    entangled_two_qubits_state = modules*np.exp(1.j*phases)
    #Normalize the entangled two qubits state
    entangled_two_qubits_state = entangled_two_qubits_state/np.sqrt(np.sum(np.power(modules,2)))
    #Generate the state vector of the separable qubit state
    one_qubit_state = generate_one_qubit_state(rng, add_global_phase=add_global_phase)

    if separable_qubit_label==None:
        #Decide which one is the unentangled qubit. The following statement returns a random number in \{1,2,3\} which matches the disentangled qubit label.
        separable_qubit_label = np.random.randint(1,high=4)

    if separable_qubit_label == 1:
        return np.kron(one_qubit_state, entangled_two_qubits_state)
    elif separable_qubit_label == 2:
        oq = one_qubit_state
        tq = entangled_two_qubits_state
        #Using the krönecker product there is no straightforward way to write the three qubits state according to the fixed
        #basis ordering. The following statement provides the way to combine the bipartitely entangled state coordinates and the
        #disentangled qubit coordinates so that the second qubit is disentangled from the rest of qubits according to the fixed
        #basis ordering.
        return np.array([tq[0]*oq[0], tq[1]*oq[0], tq[0]*oq[1], tq[1]*oq[1], tq[2]*oq[0], tq[3]*oq[0], tq[2]*oq[1], tq[3]*oq[1]])
    elif separable_qubit_label==3:
        return np.kron(entangled_two_qubits_state, one_qubit_state)
    else:
        print('Not allowed value for separable_qubit_label. Returning -1.')
        return -1




def generate_single_qubit_random_unitary_operation(rng):
    '''This function takes:
    - rng: (a random generator, as initialized by rng = Generator(PCG64(int(seed))). Here, seed is some integer.

    This function returns a bidimensional 2x2 complex numpy array which matches the matrix representation of a random
    general single qubit unitary operation as given by Eq. (4.12) in the book by Nielsen and Chuang.
    
    This function has been checked to be working as expected.'''

    #Generate four random angles, alpha, beta, gamma and delta.
    angles = 2*sp.pi*rng.random((4,), dtype=float)
    a = angles[0]; b = angles[1]; g = angles[2]; d = angles[3]; 


    U = np.empty((2,2), dtype=complex)
    U[0,0] = np.exp(1.j*(a-(b/2)-(d/2)))*np.cos(g/2)
    U[0,1] = -1.0*np.exp(1.j*(a-(b/2)+(d/2)))*np.sin(g/2)
    U[1,0] = np.exp(1.j*(a+(b/2)-(d/2)))*np.sin(g/2)
    U[1,1] = np.exp(1.j*(a+(b/2)+(d/2)))*np.cos(g/2)

    return U




def generate_GHZ_three_qubits_state(rng):
    '''This function takes:
    - rng: (a random generator, as initialized by rng = Generator(PCG64(int(seed))). Here, seed is some integer.

    This function returns a unidimensional complex numpy array with eight entries. Such array is the state vector of 
    a genuinely entangled three qubits state that belongs to the GHZ entanglement family, ordered according to the 
    previously fixed H2xH2xH2 basis. To do so, I prepare a random GHZ state according to the standard form given in Eq. 
    (15) in the paper by Dür, and then act over every qubit with a single qubit operation, so that the resulting state 
    belongs to the same entanglement family (GHZ).'''

    #Generate the angles \alpha, \beta and \gamma in [0,pi/2[ according to Dür paper.
    angles = (sp.pi/2.0)*rng.random((3,))
    #Compute its sines and cosines.
    sa = np.sin(angles[0]); ca = np.cos(angles[0])
    sb = np.sin(angles[1]); cb = np.cos(angles[1])
    sg = np.sin(angles[2]); cg = np.cos(angles[2])

    #Generate the delta angle in [0,pi/4] and compute its sine and cosine.
    delta = (sp.pi/4.0)*rng.random()
    sd = np.sin(delta); cd = np.cos(delta)

    #Generate the phi angle in [0,2pi[
    phi = 2*sp.pi*rng.random()

    K = 1.0/(1.0+(2.0*cd*sd*ca*cb*cg*np.cos(phi)))

    phi_A = np.array([ca,sa])
    phi_B = np.array([cb,sb])
    phi_C = np.array([cg,sg])

    GHZ_state = sd*np.exp(1.j*phi)*np.kron(np.kron(phi_A, phi_B), phi_C)
    GHZ_state[0] = GHZ_state[0] + cd
    GHZ_state = np.sqrt(K)*GHZ_state

    #Generate 3 single qubit random unitary operations
    U1 = generate_single_qubit_random_unitary_operation(rng)
    U2 = generate_single_qubit_random_unitary_operation(rng)
    U3 = generate_single_qubit_random_unitary_operation(rng)
    U = np.kron(np.kron(U1, U2), U3)

    return np.matmul(U, GHZ_state)




def generate_W_three_qubits_state(rng):
    '''This function takes:
    - rng: (a random generator, as initialized by rng = Generator(PCG64(int(seed))). Here, seed is some integer.

    This function returns a unidimensional complex numpy array with eight entries. Such array is the state vector of 
    a genuinely entangled three qubits state that belongs to the W entanglement family, ordered according to the 
    previously fixed H2xH2xH2 basis. The way to do so is analogous to that used in generate_GHZ_three_qubits_state().
    In this case, the standard W state form is given by Eq. (19) in the paper by Dür.'''

    parameters = rng.random((4,), dtype=float)
    parameters = parameters/np.sum(parameters)
    a = parameters[0]; b = parameters[1]; c = parameters[2]; d = parameters[3]; 

    W_state = np.zeros((8,), dtype=complex) 
    #In this case it is VITAL that you initialize the W_state as np.zeros((8,)). Then you will overwriten some of those
    #null entries according to Eq. (19) in the paper by Dür.
    W_state[0] = np.sqrt(d) #|000> is the first element in the basis
    W_state[1] = np.sqrt(a) #|001> is the second element in the basis
    W_state[2] = np.sqrt(b) #|010> is the third element in the basis
    W_state[4] = np.sqrt(c) #|100> is the fifth element in the basis

    #Generate 3 single qubit random unitary operations
    U1 = generate_single_qubit_random_unitary_operation(rng)
    U2 = generate_single_qubit_random_unitary_operation(rng)
    U3 = generate_single_qubit_random_unitary_operation(rng)
    U = np.kron(np.kron(U1, U2), U3)

    return np.matmul(U, W_state)




def generate_random_quantum_states_dataset(N, rng, outpath, entanglement_type='separable', sep='\t'):
    '''This function takes:
    - N: (integer) Number of random quantum states to 
    - rng: (a random generator, as initialized by rng = Generator(PCG64(int(seed))). Here, seed is some integer.
    - outpath: (string) Filepath to store the generated dataset.
    - entanglement_type: (string or list of strings) The string or all of the strings contained in the list must take 
    one of the following values: \{'separable', 'bipartite_entanglement', 'GHZ', 'W'\}.
    - sep: (string) Is passed to write_one_quantum_state() as sep. Separation to be inserted in the output file 
    between consecutive real entries.

    If entanglement_type is a string, then this function generates a dataset of N random quantum states whose 
    entanglement type matches entanglement_type. If entanglement_type is a list of strings, then this function 
    generates a dataset which contains N quantum states, so that the first N/len(entanglement_type) states are 
    entangled (or not) according to entanglement_type[0], the following N/len(entanglement_type) states are 
    entangled (or not) accordint to entanglement_type[1], and so on. Such dataset is written to the file whose 
    file path is given by outpath, so that every line matches one of the generated random quantum states. Since 
    each state comprises eight complex entries, each line has 16 columns, each of them storing one real value. 
    The split of eight complex entries into sixteen real values is done such that: 
    a1+ib1, a2+ib2, ..., a8+ib9 --> a1, b1, a2, b2, ..., a8, b8.
    Please note that in case entanglement_type is a list, the generated quantum states are written to the output
    file block-wise. In order to feed a neural network with this dataset, you may need to consider pre-shuffling
    the dataset.
    
    NOTE_1: For the moment, it is not possible to specify which qubit is disentangled from the rest in the case of
    entanglement_type='bipartite_entanglement' when calling this function. However, it is easy to implement, since
    such functionality is already present in generate_three_qubit_bipartitely_entangled().'''

    output_file = open(outpath, mode='w')

    if type(entanglement_type)==type(''):
        if entanglement_type=='separable':
            for i in range(N):
                state_vector_holder = generate_three_qubit_separable_state(rng)
                write_one_quantum_state(output_file, state_vector_holder, 8, sep=sep)
        elif entanglement_type=='bipartite_entanglement':
            for i in range(N):
                state_vector_holder = generate_three_qubit_bipartitely_entangled(rng)
                write_one_quantum_state(output_file, state_vector_holder, 8, sep=sep)
        elif entanglement_type=='GHZ':
            for i in range(N):
                state_vector_holder = generate_GHZ_three_qubits_state(rng)
                write_one_quantum_state(output_file, state_vector_holder, 8, sep=sep)
        elif entanglement_type=='W':
            for i in range(N):
                state_vector_holder = generate_W_three_qubits_state(rng)
                write_one_quantum_state(output_file, state_vector_holder, 8, sep=sep)
        else:
            print('Unrecognized entanglement type. Returning -1.')
            return -1

    elif type(entanglement_type)==type([]):
        howManyOfEachType = int(math.ceil(N/len(entanglement_type)))
        for j in range(len(entanglement_type)):
            if entanglement_type[j]=='separable':
                for i in range(howManyOfEachType):
                    state_vector_holder = generate_three_qubit_separable_state(rng)
                    write_one_quantum_state(output_file, state_vector_holder, 8, sep=sep)
            elif entanglement_type[j]=='bipartite_entanglement':
                for i in range(howManyOfEachType):
                    state_vector_holder = generate_three_qubit_bipartitely_entangled(rng)
                    write_one_quantum_state(output_file, state_vector_holder, 8, sep=sep)
            elif entanglement_type[j]=='GHZ':
                for i in range(howManyOfEachType):
                    state_vector_holder = generate_GHZ_three_qubits_state(rng)
                    write_one_quantum_state(output_file, state_vector_holder, 8, sep=sep)
            elif entanglement_type[j]=='W':
                for i in range(howManyOfEachType):
                    state_vector_holder = generate_W_three_qubits_state(rng)
                    write_one_quantum_state(output_file, state_vector_holder, 8, sep=sep)
            else:
                print(str(j)+'-th element of entanglement type is unrecognized. Returning -1.')
                return -1            

    output_file.close()
    return


def write_one_quantum_state(output_file, quantum_state, M, insert_eol=True, sep='\t'):
    '''This function takes:
    - output_file: A file variable which must have been opened in write mode. For example, as returned by
    open(outpath, mode='w').
    - quantum_state: (unidimensional complex numpy array) This array comprises M complex entries.
    - M: (integer) Number of complex entries in quantum_state.
    - inster_eol: (boolean) Whether to write and end of line after writing the quantum state, or not. It
    is set to True by default.
    - sep: (string) Separator to insert in output_file between consecutive real entries. It is a tabulator
    ('\t') by default.

    This function writes quantum_state to output_file according to the format specified in
    generate_random_quantum_states_dataset() documentation. I.e.:
    a1+ib1, a2+ib2, ..., a8+ib9 --> a1, b1, a2, b2, ..., a8, b8.'''

    for i in range(M-1):
        output_file.write(str(np.real(quantum_state[i]))+sep+str(np.imag(quantum_state[i]))+sep)
    #I do not want to introduce a tabulator before ending the line.
    output_file.write(str(np.real(quantum_state[M-1]))+sep+str(np.imag(quantum_state[M-1])))
    if insert_eol==True:
        output_file.write('\n')
    return
