import numpy as np
import time
import copy
from mylib import TDVP, MPS

# 変数の並びのルール
# mps > D,L > maxbond > h > J > T,dt,n_steps > cutoff > operator
# 左bond-物理-右bondの順に足を並べる

def U_X(h,dt):
    sigma_x = np.array([[0,1],[1,0]])
    S, U = np.linalg.eig(sigma_x)
    alpha = 1j * dt * h * S
    U_X = U @ np.diag(np.exp(alpha)) @ U.conj().T
    return U_X

def U_ZZ(J, dt):
    beta = 1j * dt * J
    return np.diag(np.exp(beta * np.array([1,-1,-1,1])))

# ZZにXを吸収させたユニタリを定義
def U_ZZX(h, J, dt):
    # U_ZX = exp(dt * (J * sigma_z + h * sigma_x))
    sigma_z = np.array([[1, 0], [0, -1]])
    sigma_x = np.array([[0, 1], [1, 0]])
    H = J * np.kron(sigma_z, sigma_z) + h * np.kron(sigma_x, np.eye(2)) + h * np.kron(np.eye(2), sigma_x)
    S, U = np.linalg.eigh(H)
    gamma = 1j * dt * S
    U_H = U @ np.diag(np.exp(gamma)) @ U.conj().T
    return U_H

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

# TEBDで用いるユニタリを構築する関数
def build_twosite_unitary(
    h: float,
    J: float,
    dt: float,
    operator: str
):
    valid1 = {
        'ZZ': U_ZZ
    }
    valid2 = {
        'ZX': U_ZZX
    }
    if operator in valid1:
        U_twosite = valid1[operator](J, dt)
    elif operator in valid2:
        U_twosite = valid2[operator](h, J, dt)
    else:
        raise ValueError(f"Operator '{operator}' is not supported.")
    return U_twosite

""" TEBDの関数群 """

def onsite(mps, h, dt):
    U = U_X(h, dt)
    for i in range(len(mps)):
        mps[i] = np.einsum('ijk,jl->ilk',mps[i],U)

# 一般の近接2サイト相互作用を適用する関数
def apply_bond_layer(
    mps,
    D: list,                    # 各サイトのボンド次元のリスト
    maxbond: int,               # 最大ボンド次元
    h: float,                   # 磁場の強さ
    J: float,                   # 相互作用定数
    dt: float,                  # このstepの時間幅（係数込み）
    operator: str,              # 'ZZ' or 'JZZhXhX' or ...
    bond_type: str,             # 'even' or 'odd'
    direction: str,             # 'lr'（左→右）または 'rl'（右→左）
    cutoff: float = 1e-10       # SVDの閾値
):  
    L = len(mps)
    U_twosite = build_twosite_unitary(h, J, dt, operator)
    if direction == 'lr':
        if bond_type == 'even':
            start = 0
        elif bond_type == 'odd':
            start = 1
        for i in range(start, L-1, 2):
            # contract mps[i] and mps[i+1]
            T = np.einsum('ijk,klm->ijlm', mps[i], mps[i + 1])
            T = T.reshape(D[i],4,D[i+2])
            T = np.einsum('ijk,jl->ilk', T, U_twosite)
            T = T.reshape(D[i]*2, 2*D[i+2])
            U, S, Vh = np.linalg.svd(T, full_matrices=False)
            # truncate SVD
            indices = np.where(S < cutoff)[0]
            if len(indices) > 0:
                idx = min(indices[0], maxbond)
            else:
                idx = min(len(S), maxbond)
            U = U[:, :idx]
            S = S[:idx]
            Vh = Vh[:idx, :]
            S /= np.linalg.norm(S) # normalize S
            D[i+1] = idx
            # reshape and update mps tensors to mixed canonical form
            if i < L-2:
                mps[i] = U.reshape(D[i], 2, D[i+1])
                mps[i+1] = S[:, None] * Vh
                if i == L-3: 
                    mps[i+1] = mps[i+1].reshape(D[i+1], 2, D[i+2])
                    #print(f'center is {i+1}')
                else:
                    mps[i+1] = mps[i+1].reshape(D[i+1]*2, D[i+2])
                    Q, R = np.linalg.qr(mps[i+1])
                    mps[i+1] = Q.reshape(D[i+1], 2, D[i+2])
                    mps[i+2] = np.einsum('ij,jkl->ikl', R, mps[i+2])
                    #print(f'center is {i+2}')  
            else: # if i == L-2
                mps[i] = U * S[None,:]
                mps[i] = mps[i].reshape(D[i], 2, D[i+1])
                mps[i+1] = Vh.reshape(D[i+1], 2, D[i+2])
                #print(f'center is {i}')
    elif direction == 'rl':
        if bond_type == 'odd':
            start = L-1 if (L-1) % 2 == 0 else L-2
        elif bond_type == 'even':
            start = L-2 if (L-1) % 2 == 0 else L-1
        targets = range(start, 1, -2)
        # if bond_type is odd, targets = [... ,4,2]
        # hamilonian is applied to {[1,2],[3,4],...}
        # if bond_type is even, targets = [... ,3,1]
        # hamilonian is applied to {[0,1],[2,3],...}
        for i in targets:
            # contract mps[i-1] and mps[i]
            T = np.einsum('ijk,klm->ijlm', mps[i-1], mps[i])
            T = T.reshape(D[i-1],4,D[i+1])
            T = np.einsum('ijk,jl->ilk', T, U_twosite)
            T = T.reshape(D[i-1]*2, 2*D[i+1])
            U, S, Vh = np.linalg.svd(T, full_matrices=False)
            # truncate SVD
            indices = np.where(S < cutoff)[0]
            if len(indices) > 0:
                idx = min(indices[0], maxbond)
            else:
                idx = min(len(S), maxbond)
            U = U[:, :idx]
            S = S[:idx]
            Vh = Vh[:idx, :]
            S /= np.linalg.norm(S)
            D[i] = idx
            # reshape and update mps tensors to mixed canonical form
            if 1 < i: 
                mps[i] = Vh.reshape(D[i], 2, D[i+1])
                mps[i-1] = U * S[None,:]
                if i == 2:
                    mps[i-1] = mps[i-1].reshape(D[i-1], 2, D[i])
                    #print(f'center is {i-1}')
                else:
                    mps[i-1] = (U * S[None,:]).reshape(D[i-1], 2*D[i])
                    Q, R = np.linalg.qr(mps[i-1].T)
                    mps[i-1] = Q.T.reshape(D[i-1], 2, D[i])
                    mps[i-2] = np.einsum('ijk,kl->ijl',mps[i-2],R.T)
                    #print(f'center is {i-2}')
            else: # if i == 1
                mps[i] = [S[:, None] * Vh].reshape(D[i], 2, D[i+1])
                mps[i-1] = U.reshape(D[i-1], 2, D[i])
                #print(f'center is {i}')      

def tebd1(
    mps,
    D: list,                    # 各サイトのボンド次元のリスト
    maxbond: int,               # 最大ボンド次元
    h: float,                   # 磁場の強さ
    J: float,                   # 相互作用定数
    T: float,                   # 時間
    n_steps: int,               # 時間ステップ数
    cutoff: float = 1e-10,      # SVDの閾値
    operator: str = 'z'         # 演算子の種類 ('z' or 'x')
):  
    dt = T / n_steps
    Time = []
    Magnetization = []
    Time.append(0)
    Magnetization.append(np.sum(expval(operator, mps, 1)))
    for step in range(n_steps): 
        if step % (n_steps // 10) == 0:
            cleaned = [int(x) for x in D]
            #print(cleaned)
        onsite(mps, h, dt)
        apply_bond_layer(mps, D, maxbond, h, J, dt, 'ZZ', 'even', 'lr', cutoff)
        apply_bond_layer(mps, D, maxbond, h, J, dt, 'ZZ', 'odd', 'rl', cutoff)
        
        Time.append((step+1) * dt)
        M = np.sum(expval(operator, mps, 1))
        Magnetization.append(M)

        # # 進捗状況の表示
        percent = (step + 1) / n_steps * 100
        print(f"進捗状況: {percent:.2f}%", end="\r")
        time.sleep(0.00000001)
        
    return Time, Magnetization

def tebd2(
    mps,
    h: float,                   # 磁場の強さ
    J: float,                   # 相互作用定数
    T: float,                   # 時間
    n_steps: int,               # 時間ステップ数
    maxbond: int = 100,         # 最大ボンド次元
    cutoff: float = 1e-10,      # SVDの閾値
    operator: str = 'z'         # 演算子の種類 ('z' or 'x')
):  
    D = MPS.get_bondinfo(mps)
    mpo = TDVP.mpo_ising_transverse(len(mps), h, J)
    dt = T / n_steps
    Time = []
    Magnetization = []
    Energy = []
    Time.append(0)
    Magnetization.append(np.sum(expval(operator, mps, 1)))
    Energy.append(energy(mps,mpo))
    for step in range(n_steps): 
        if step % (n_steps // 10) == 0:
            cleaned = [int(x) for x in D]
            #print(cleaned)
        onsite(mps, h, dt/2)
        apply_bond_layer(mps, D, maxbond, h, J, dt, 'ZZ', 'even', 'lr', cutoff)
        apply_bond_layer(mps, D, maxbond, h, J, dt, 'ZZ', 'odd', 'rl', cutoff)
        onsite(mps, h, dt/2)
        
        Time.append((step+1) * dt)
        M = np.sum(expval(operator, mps, 1))
        Magnetization.append(M)
        Energy.append(energy(mps,mpo))

        # # 進捗状況の表示
        percent = (step + 1) / n_steps * 100
        print(f"進捗状況: {percent:.2f}%", end="\r")
        time.sleep(0.00000001)
        
    return Time, Magnetization, Energy

""" 物理量の計算 """

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

# single spin-flip
