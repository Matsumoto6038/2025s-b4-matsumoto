import numpy as np
import scipy as sp
import time
from mylib import TEBD 

""" TDVPの注意点 """

# L > mps > D > maxbond > h > J > T,dt,n_steps > cutoff > operator
# 0番目から数える。

""" 足の順番についての注意 """
# mpoの足の順番は、左下上右の順。
# 縮約はできるだけ以下のルールでかく。
# ボンドはi,j,k,l,,,で、物理系の足はa,b,c,d,,,で書く。
# np.einsum('iabj,kbl->ikajl',mpo[i],mps[i]) bra-operator-ketの順番で書く
# np.einsum('iabj,jcdk->iacbdk',mpo[i],mpo[i+1])
# np.einsum('iaj,kabl->ikbjl',mps[i].conj().T,mpo[i])

""" 状態や演算子の準備 """

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

# MPSのbond dimensionを取得する関数
def get_bondinfo(mps):
    L = len(mps)
    D = []
    D.append(1)
    for i in range(L):
        D.append(mps[i].shape[2])
    return D

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

""" TDVPの関数群 """

# Enviroment
def initial_right_env(mps, mpo):
    # mpsはright canonical form
    if len(mps) != len(mpo):
        raise ValueError("Length of mps and mpo must be the same.")
    # if check_right_canonical(mps) == False:
    #     raise ValueError("mps must be in right canonical form.")
    length = len(mps)
    Right = []
    R = np.array([[[1]]]) # 左下上の順
    Right.append(R)  # R_n
    # R_n-1, ..., R_2,
    for i in range(length-1, 1, -1):
        R = np.einsum('iabj,jcd->iacbd',mpo[i],R)
        R = np.einsum('iacbd,kbd->ikac',R,mps[i])
        R = np.einsum('jac,ikac->ijk',mps[i].conj(),R)
        Right.append(R)
    Right.reverse() 
    # R_2, R_3, ..., R_n
    # R_i にアクセスする時は R[i-2]となる。
    
    return Right

def update_right_env(mps, mpo, Right, i):
    # R{i} (R[i-2])を更新する。
    # iは2,3,...,n-1の範囲で指定する。
    if i < 2 or i >= len(mps):
        raise ValueError("i must be in the range [2, len(mps) - 1].")
    Right[i-2] = np.einsum('iabj,jcd->iacbd', mpo[i], Right[i-1])
    Right[i-2] = np.einsum('iacbd,kbd->ikac', Right[i-2], mps[i])
    Right[i-2] = np.einsum('jac,ikac->ijk', mps[i].conj(), Right[i-2])

def initial_left_env(L):
    # L_-1, L_0, L_1, ..., L_{n-3}
    # L_iにアクセスするときは L[i+1]となる。
    # 足は下上右の順
    return [np.array([[[1]]])] * (L - 1)

def update_left_env(mps, mpo, Left, i):
    # L_{i}(L[i+1])を更新する。
    # iは0,1,...,L-3の範囲で指定する。
    if i < 0 or i > len(mps) - 3:
        raise ValueError("i must be in the range [0, len(mps) - 3].")
    Left[i+1] = np.einsum('abi,icdj->acbdj', Left[i], mpo[i]) # L_{i-1}とmpo[i]をかける
    Left[i+1] = np.einsum('acbdj,bdl->aclj', Left[i+1], mps[i])
    Left[i+1] = np.einsum('ack,aclj->klj', mps[i].conj(), Left[i+1])

# l->rとr->lをまとめて行う
def sweep(
    mps: list, 
    mpo: list, 
    maxbond: int, 
    dt: float, 
    Left: list, # Left　environment
    Right: list # Right environment
):
    if len(mps) != len(mpo):
        raise ValueError("Length of mps and mpo must be the same.")
    L = len(mps)
    alpha = -1/2 * dt * 1.j
    
    for i in range(L-1):
        # mps[i]とmps[i+1]の縮約をとり整形する
        T = np.einsum('iaj,jbk->iabk',mps[i],mps[i+1])
        shape = T.shape # 足の情報
        T = T.reshape(np.prod(shape))
        """ sparse """
        Heff2 = make_Heff2(mpo, Left, Right, alpha, i)
        Heff = sp.sparse.linalg.LinearOperator(
            shape = (np.prod(shape), np.prod(shape)),
            matvec = Heff2,
            rmatvec = Heff2  # 省略可だけどあると安全 
        )
        T = sp.sparse.linalg.expm_multiply(Heff, T)
        T = T.reshape(shape[0]*shape[1], shape[2]*shape[3])
        U_svd, S_svd, Vh_svd = np.linalg.svd(T, full_matrices=False)
        # トランケーション
        if len(S_svd) > maxbond:
            idx = maxbond
            U_svd = U_svd[:, :idx]
            S_svd = S_svd[:idx]
            Vh_svd = Vh_svd[:idx, :]
            S_svd /= np.linalg.norm(S_svd)
        else:
            idx = len(S_svd)
        mps[i] = U_svd.reshape(shape[0], shape[1], idx)
        mps[i+1] = (Vh_svd).reshape(idx, shape[2], shape[3])
        mps[i+1] = np.einsum('ij,jak->iak', np.diag(S_svd), mps[i+1])  
        if i != L - 2:
            update_left_env(mps, mpo, Left, i)
            mps[i+1] = mps[i+1].reshape(idx*shape[2]*shape[3])
            """ sparse """
            Heff1 = make_Heff1(mpo, Left, Right, -alpha, i+1)
            Heff = sp.sparse.linalg.LinearOperator(
                shape = (np.prod(idx*shape[2]*shape[3]), np.prod(idx*shape[2]*shape[3])),
                matvec = Heff1,
                rmatvec = Heff1  # 省略可だけどあると安全 
            )
            mps[i+1] = sp.sparse.linalg.expm_multiply(Heff, mps[i+1])
            mps[i+1] = mps[i+1].reshape(idx, shape[2], shape[3])
            
    for i in range(L-1, 0, -1):
        T = np.einsum('iaj,jbk->iabk', mps[i-1], mps[i])
        shape = T.shape  # 足の情報
        T = T.reshape(np.prod(shape))
        """ sparse """
        Heff2 = make_Heff2(mpo, Left, Right, alpha, i-1)
        Heff = sp.sparse.linalg.LinearOperator(
            shape = (np.prod(shape), np.prod(shape)),
            matvec = Heff2,
            rmatvec = Heff2  # 省略可だけどあると安全 
        )
        T = sp.sparse.linalg.expm_multiply(Heff, T)
        T = T.reshape(shape[0]*shape[1], shape[2]*shape[3])
        U_svd, S_svd, Vh_svd = np.linalg.svd(T, full_matrices=False)
        # トランケーション
        if len(S_svd) > maxbond:
            idx = maxbond
            U_svd = U_svd[:, :idx]
            S_svd = S_svd[:idx]
            Vh_svd = Vh_svd[:idx, :]
            S_svd /= np.linalg.norm(S_svd)
        else:
            idx = len(S_svd)
        mps[i] = Vh_svd.reshape(idx, shape[2], shape[3])
        mps[i-1] = (U_svd).reshape(shape[0], shape[1], idx)
        mps[i-1] = np.einsum('iaj,jk->iak', mps[i-1], np.diag(S_svd))  
        if i != 1:
            update_right_env(mps, mpo, Right, i)
            mps[i-1] = mps[i-1].reshape(shape[0]*shape[1]*idx)
            """ sparse """
            Heff1 = make_Heff1(mpo, Left, Right, -alpha, i-1)
            Heff = sp.sparse.linalg.LinearOperator(
                shape = (np.prod(idx*shape[0]*shape[1]), np.prod(idx*shape[0]*shape[1])),
                matvec = Heff1,
                rmatvec = Heff1  # 省略可だけどあると安全 
            )
            mps[i-1] = sp.sparse.linalg.expm_multiply(Heff, mps[i-1])
            mps[i-1] = mps[i-1].reshape(shape[0], shape[1], idx)

def tdvp(
    mps: list,
    mpo: list,
    maxbond: int,
    T: float,
    n_steps: int,
):
    Left = initial_left_env(len(mps))
    Right = initial_right_env(mps, mpo)
    dt = T / n_steps
    Time = []
    Time.append(0)
    Magnetization = []
    Magnetization.append(np.sum(TEBD.expval('x', mps, 0)))
    for i in range(n_steps):
        sweep(mps, mpo, maxbond, dt, Left, Right)
        Time.append((i+1) * dt)
        Magnetization.append(np.sum(TEBD.expval('x', mps, 0)))
        progress(i, n_steps)

    return Time, Magnetization

def progress(i, n_steps):
    # 進捗状況の表示
    percent = (i + 1) / n_steps * 100
    print(f"進捗状況: {percent:.2f}%", end="\r")
    time.sleep(0.00000001)

# 有効ハミルトニアンH_effを作る
def make_Heff2(mpo, Left, Right, alpha, l):
    # l,l+1に作用するので、
    if l < 0 or l >= len(mpo) - 1:
        raise ValueError("l must be in the range [0, len(mpo) - 2].")
    # L_{l-1}, mpo[l], mpo[l+1], R_{l+2}で構成される。
    # (Left[l], mpo[l], mpo[l+1], Right[l])
    shape = (Left[l].shape[1],mpo[l].shape[2],mpo[l+1].shape[2],Right[l].shape[2])
    def Heff2(v):
        v = v.reshape(shape)
        v = np.einsum('cik,iabj->cabjk', Left[l], v)
        v = np.einsum('cabjk,kdal->cdbjl', v, mpo[l])
        v = np.einsum('cdbjl,lebm->cdejm', v, mpo[l+1])
        v = np.einsum('cdejm,mfj->cdef', v, Right[l])
        v = alpha * v.reshape(np.prod(shape))
        return v
    return Heff2

def make_Heff1(mpo, Left, Right, alpha, l):
    # H_{eff}^{l}
    if l < 1 or l >= len(mpo)-1:
        raise ValueError("l must be in the range [1, len(mpo) - 2].")
    # L_{l-1}, mpo[l], R_{l+1}で構成される。
    # (Left[l], mpo[l], Right[l-1])
    shape = (Left[l].shape[1], mpo[l].shape[2], Right[l-1].shape[2])
    def Heff1(v):
        v = v.reshape(shape)
        v = np.einsum('bik,iaj->bajk', Left[l], v)
        v = np.einsum('bajk,kcal->bcjl', v, mpo[l])
        v = np.einsum('bcjl,ldj->bcd', v, Right[l-1])
        v = alpha * v.reshape(np.prod(shape))
        return v
    return Heff1