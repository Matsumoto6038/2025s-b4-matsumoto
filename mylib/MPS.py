import numpy as np

def random_MPS_mixed_cannonical(phys_dim,bond_dim,L,center):
    """ create a random MPS in mixed canonical form
    Args:
        phys_dim (int): physical dimension of the local Hilbert space
        bond_dim (int): maximum bond dimension
        L (int): number of sites in the MPS
        center (int): index of the center site, must be in the range [0, L-1]
    Returns:
        A (list): list of tensors representing the MPS in mixed canonical form"""
    if L < 2:
        raise ValueError("L must be at least 2.")
    # Check input parameters
    if center < 0 or center >= L:
        raise ValueError("Center must be in the range [0, L-1].")
    A = []
    Bond_list = []
    for i in range(L-1):
        Bond_list.append(min(bond_dim, phys_dim**(i+1), phys_dim**(L-i-1))) 
    Q, _ = np.linalg.qr(np.random.rand(phys_dim, Bond_list[0]))
    A.append(Q)  # (d,D[0])
    for i in range(1, min(center+1, L-1)):
        B = (np.random.rand(Bond_list[i-1], phys_dim, Bond_list[i])).reshape(Bond_list[i-1]*phys_dim, Bond_list[i])  # (D[i-1],d,D[i])
        Q, _ = np.linalg.qr(B)
        A.append(Q.reshape(Bond_list[i-1], phys_dim, Bond_list[i]))  # (D[i-1],d,D[i])
    for i in range(center+1, L-1):
        B = (np.random.rand(Bond_list[i-1], phys_dim, Bond_list[i])).reshape(Bond_list[i-1], phys_dim*Bond_list[i])  # (D[i-1],d,D[i])
        Q, _ = np.linalg.qr(B.T)
        A.append((Q.T).reshape(Bond_list[i-1], phys_dim, Bond_list[i]))
    Q, _ = np.linalg.qr(np.random.rand(phys_dim, Bond_list[L-2]))
    A.append(Q.T) # (D[L-2], d)

    # insert the orthogonality center matrix
    if center != (L-1):
        S = np.random.rand(Bond_list[center])
        S = S / np.linalg.norm(S) 
        S = np.diag(S)  # make it diagonal
        A[center] = np.tensordot(A[center], S, axes=([-1],[0])) 
    else:
        S = np.random.rand(Bond_list[L-2])
        S = S / np.linalg.norm(S) 
        S = np.diag(S) 
        A[L-1] = np.tensordot(S, A[L-1], axes=([-1],[0]))
        
    return A, Bond_list

def inner_product(mps_1,mps_2):
    """ compute the inner product of two MPS
    Args:
        A (list): first MPS in mixed canonical form
        B (list): second MPS in mixed canonical form
    Returns:
        float: inner product <A|B>
    """
    if len(mps_1) != len(mps_2):
        raise ValueError("MPS must have the same number of tensors.")
    L = len(mps_1)
    
    Ah = [] # conjugate transpose of A
    for i in range(L):
        Ah.append(mps_1[i].conjugate())

    Transfer = []
    for i in range(L):
        Transfer.append(np.einsum('ijk,ljm->ilkm',mps_2[i],Ah[i]))

    value = Transfer[0]
    for i in range(1, L):
        value = np.einsum('abcd,cdef->abef',value,Transfer[i])

    return (value.reshape(1))

# MPSのbond dimensionを取得する関数
def get_bondinfo(mps):
    L = len(mps)
    D = []
    D.append(1)
    for i in range(L):
        D.append(mps[i].shape[2])
    return D