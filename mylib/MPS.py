import numpy as np
import copy
from mylib import TDVP,TEBD

""" 足の順番についての注意 """
# mpoの足の順番は、左下上右の順。
# 縮約はできるだけ以下のルールでかく。
# ボンドはi,j,k,l,,,で、物理系の足はa,b,c,d,,,で書く。
# np.einsum('iabj,kbl->ikajl',mpo[i],mps[i]) bra-operator-ketの順番で書く
# np.einsum('iabj,jcdk->iacbdk',mpo[i],mpo[i+1])
# np.einsum('iaj,kabl->ikbjl',mps[i].conj().T,mpo[i])

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

""" 内積、期待値の計算 """
# 内積
def inner_product(mps1, mps2):
    # <mps1|mps2>
    if len(mps1) != len(mps2):
        raise ValueError("mps1 and mps2 must have the same length.")
    for i in range(len(mps1)):
        if i == 0:
            inner = np.einsum('iaj,iak->jk', mps1[i].conj(), mps2[i])
        else:
            inner = np.einsum('ij,jak->iak', inner, mps2[i])
            inner = np.einsum('iaj,iak->jk', mps1[i].conj(), inner)
    return inner.reshape(1)

# 全ビットの期待値を計算するための環境テンソル
def env(mps,center):
    #mixed canonical formの環境行列
    Left = []
    Right = []
    L = len(mps)
    for i in range(center,0,-1):
        if i == center:
            R = np.einsum('acd,bcd->ab', mps[i], mps[i].conj())
        else:
            R = np.einsum('abc,cd->abd', mps[i], R)
            R = np.einsum('acd,bcd->ab', R, mps[i].conj())
        Right.append(R)
    for i in range(center,L-1):
        if i == center: 
            L = np.einsum('abc,abd->cd', mps[i], mps[i].conj())
        else:
            L = np.einsum('da,dbc->abc', L, mps[i])
            L = np.einsum('abc,abd->cd', L, mps[i].conj())
        Left.append(L)
    #for i in range(len(Left)):
    #    print(f'Left[{i}].shape: {Left[i].shape}')
    #for i in range(len(Right)):
    #    print(f'Right[{i}].shape: {Right[i].shape}')
    return Left, Right

# 期待値
def expval(operator,mps,center):
    if operator == 'z':
        O = np.array([[1,0],[0,-1]])
    elif operator == 'x':
        O = np.array([[0,1],[1,0]])
    else:
        raise ValueError("operator must be 'z' or 'x'")
    left_env, right_env = env(mps, center)
    z_list = []
    L = len(mps)
    # 左端から中心までの期待値の計算
    for i in range(0,center):
        z_i = np.einsum('adc,db->abc',mps[i],O)
        z_i = np.einsum('abc,abd->cd',z_i, mps[i].conj())
        z_i = np.einsum('ab,ab->', z_i, right_env[center-i-1])
        z_list.append(float(z_i.real))
    # 中心の期待値の計算  
    z_i = np.einsum('adc,db->abc', mps[center], O)
    z_i = np.einsum('aec,bed->abcd', z_i, mps[center].conj())
    z_i = np.einsum('aabb->', z_i)
    z_list.append(float(z_i.real))
    # 中心から右端までの期待値の計算
    for i in range(center+1,L):
        z_i = np.einsum('adc,db->abc', mps[i], O)
        z_i = np.einsum('acd,bcd->ab', z_i, mps[i].conj())
        z_i = np.einsum('ab,ab->', left_env[i-center-1], z_i)
        z_list.append(float(z_i.real))
    return z_list

# 入力'x'などに対して対応する演算子を返す関数
def build_operator(
    operator: str
):
    valid = {
        'z': np.array([[1, 0], [0, -1]]),
        'x': np.array([[0, 1], [1, 0]]),
        'y': np.array([[0, -1j], [1j, 0]])
    }
    if operator in valid:
        return valid[operator]
    else:
        raise ValueError(f"Operator '{operator}' is not supported.")

# 相関
def correlation(mps, i, j, operator='z'):
    copy_mps = copy.deepcopy(mps)
    op = build_operator(operator)
    mps_i = np.einsum('ab,ibj->iaj', op, copy_mps[i])
    mps_j = np.einsum('ab,ibj->iaj', op, copy_mps[j])
    copy_mps[i] = mps_i 
    z_i = inner_product(mps, copy_mps)
    copy_mps[i] = mps[i]  # 元の状態に戻す
    copy_mps[j] = mps_j
    z_j = inner_product(mps, copy_mps)
    copy_mps[i] = mps_i  
    z_ij = inner_product(mps, copy_mps)
    return z_ij - z_i * z_j # <z_i z_j> 

# エネルギーも測定
def energy(mps, mpo):
    Right = TDVP.initial_right_env(mps, mpo)
    Left = TDVP.initial_left_env(len(mpo))
    TDVP.update_left_env(mps, mpo, Left, 0)
    TDVP.update_left_env(mps, mpo, Left, 1)
    energy = np.einsum('abi,iab->',Left[2],Right[0])
    return -energy
