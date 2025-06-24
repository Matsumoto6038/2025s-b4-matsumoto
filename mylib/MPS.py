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

""" mpsの初期状態を生成する関数群 """
def plus(L):
    mps = []
    for i in range(L):
        mps.append(np.array([1/np.sqrt(2),1/np.sqrt(2)]).reshape(1,2,1))
    return mps

def all_up(L):
    mps = []
    for i in range(L):
        mps.append(np.array([1,0]).reshape(1,2,1))
    return mps

def all_down(L):
    mps = []
    for i in range(L):
        mps.append(np.array([0,1]).reshape(1,2,1))
    return mps

# 1/sqrt(2)(|0...0>+|1...1>)
def GHZ_unnormalized(L):
    mps = []
    ket_0 = np.array([1,0])
    ket_1 = np.array([0,1])
    zero = np.array([0,0])
    A = np.array([[ket_0,ket_1]]).transpose(0,2,1)  
    mps.append(A)  
    for i in range(1,L-1):
        A = np.array([[ket_0,zero],[zero,ket_1]]).transpose(0,2,1)
        mps.append(A)
    A = np.array([[ket_0],[ket_1]]).transpose(0,2,1)
    mps.append(A)  # 最後のサイト
    return mps

def mps_random(L, D):
    # 与えられたbond dimension Dに基づいてランダムなMPSを生成
    if len(D) != L + 1:
        raise ValueError("Length of D must be L + 1.")
    if D[0] != 1 or D[L] != 1:
        raise ValueError("First and last bond dimensions must be 1.")
    for i in range(1, L):
        if D[i] > (2 ** min(i, L-i)):
            raise ValueError("Bond dimensions must not exceed 2^min(i, L-1-i).")
    mps = []
    for i in range(L):
        A = np.random.rand(D[i], 2, D[i+1])+ 1j * np.random.rand(D[i], 2, D[i+1])
        mps.append(A)
    return mps

# MPSのbond dimensionを取得する関数
def get_bondinfo(mps):
    L = len(mps)
    D = []
    D.append(1)
    for i in range(L):
        D.append(mps[i].shape[2])
    return D

""" MPOの関数群 """