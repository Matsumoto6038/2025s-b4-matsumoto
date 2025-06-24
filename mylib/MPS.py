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
# 長さLも必要ない。
def mpo_ising_transverse(L,h,J):
    mpo = []
    sigma_z = np.array([[1, 0], [0, -1]])
    sigma_x = np.array([[0, 1], [1, 0]])
    I = np.eye(2)
    zero = np.zeros((2, 2))
    O = np.array([[h*sigma_x, J*sigma_z, I]]).transpose(0,2,3,1)
    mpo.append(O)  # mpo[0]
    for i in range(1, L-1):
        O = np.array([[I,zero,zero],[sigma_z,zero,zero],[h*sigma_x,J*sigma_z,I]]).transpose(0,2,3,1)
        mpo.append(O)
    O = np.array([[I],[sigma_z],[h*sigma_x]]).transpose(0,2,3,1)
    mpo.append(O)  # mpo[L-1]
    return mpo

""" canonical form の関数群 """

def right_canonical(mps):
    # mpsをright canonical form に変形する
    L = len(mps)
    D = get_bondinfo(mps)
    # LQ分解
    for i in range(L-1, 0, -1):
        mps[i] = mps[i].reshape(D[i],D[i+1]*2)
        q, r = np.linalg.qr(mps[i].T)
        mps[i] = q.T.reshape(D[i],2,D[i+1])
        mps[i-1] = np.einsum('ijk,kl->ijl', mps[i-1], r.T)
    # 規格化
    mps[0] = mps[0] / np.sqrt(np.einsum('ijk,ijk->',mps[0] , mps[0].conj()))
    return mps

def check_right_canonical(mps):
    # mpsがright canonical formになっているか確認する
    L = len(mps)
    D = get_bondinfo(mps)
    for i in range(1,L):
        A = mps[i].reshape(D[i],D[i+1]*2)
        # print(A @ A.conj().T)
        val = np.linalg.norm(np.eye(D[i]) - A @ A.conj().T)
        # print(val)
        if val > 1e-10:
            return False
    return True
