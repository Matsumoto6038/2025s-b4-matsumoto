import numpy as np
import time
from mylib import TDVP, MPS
import copy

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
    h: float,                   # 磁場の強さ
    J: float,                   # 相互作用定数
    dt: float,                  # このstepの時間幅（係数込み）
    operator: str,              # 'ZZ' or 'JZZhXhX' or ...
    bond_type: str,             # 'even' or 'odd'
    direction: str,             # 'lr'（左→右）または 'rl'（右→左）
    maxbond: int = 100,               # 最大ボンド次元
    cutoff: float = 1e-10       # SVDの閾値
):  
    D = MPS.get_bondinfo(mps)
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
    h: float,                   # 磁場の強さ
    J: float,                   # 相互作用定数
    T: float,                   # 時間
    n_steps: int,               # 時間ステップ数
    maxbond: int = 100,         # 最大ボンド次元
    cutoff: float = 1e-10,      # SVDの閾値
    output_type: str = 'energy',# 演算子の種類 ('z' or 'x')
    clone = False                # mpsをコピーするかどうか
):  
    mps_copy = copy.deepcopy(mps) if clone else mps
    D = MPS.get_bondinfo(mps_copy)
    dt = T / n_steps
    mpo = MPS.mpo_ising_transverse(len(mps_copy), h, J)
    
    Result = []
    exp_func = output(output_type, mpo)
    Result.append(exp_func(mps_copy))
    
    for step in range(n_steps): 
        if step % (n_steps // 10) == 0:
            cleaned = [int(x) for x in D]
            #print(cleaned)
        onsite(mps, h, dt)
        apply_bond_layer(mps_copy, h, J, dt, 'ZZ', 'even', 'lr', maxbond, cutoff)
        apply_bond_layer(mps_copy, h, J, dt, 'ZZ', 'odd', 'rl', maxbond, cutoff)
        
        Result.append(exp_func(mps_copy))

        # # 進捗状況の表示
        percent = (step + 1) / n_steps * 100
        print(f"進捗状況: {percent:.2f}%", end="\r")
        time.sleep(0.00000001)
        
    return Result

def tebd2(
    mps,
    h: float,                   # 磁場の強さ
    J: float,                   # 相互作用定数
    T: float,                   # 時間
    n_steps: int,               # 時間ステップ数
    maxbond: int = 100,         # 最大ボンド次元
    cutoff: float = 1e-10,      # SVDの閾値
    output_type: str = 'energy',# 出力の種類 ('z' or 'x')
    clone = False                # mpsをコピーするかどうか
):  
    mps_copy = copy.deepcopy(mps) if clone else mps
    D = MPS.get_bondinfo(mps_copy)
    mpo = MPS.mpo_ising_transverse(len(mps_copy), h, J)
    dt = T / n_steps
    
    exp_func = output(output_type, mpo)
    Result = []
    Result.append(exp_func(mps_copy))
    
    for step in range(n_steps): 
        onsite(mps_copy, h, dt/2)
        apply_bond_layer(mps_copy, h, J, dt, 'ZZ', 'even', 'lr',maxbond, cutoff)
        apply_bond_layer(mps_copy, h, J, dt, 'ZZ', 'odd', 'rl', maxbond, cutoff)
        onsite(mps_copy, h, dt/2)
        
        Result.append(exp_func(mps_copy))

        # # 進捗状況の表示
        percent = (step + 1) / n_steps * 100
        print(f"進捗状況: {percent:.2f}%", end="\r")
        time.sleep(0.00000001)
        
    return Result

# tebdの出力を選択する関数
def output(
    output_type : str,
    mpo : list,
    ):
    if output_type == 'energy':
        def energy(mps):
            result = MPS.energy(mps, mpo)
            return result
        return energy
    elif output_type == 'M_x':
        def M_x(mps,center=1):
            result = np.sum(MPS.expval('x', mps, center))
            return result
        return M_x
    elif output_type == 'M_z':
        def M_z(mps,center=1):
            result = np.sum(MPS.expval('z', mps, center))
            return result
        return M_z
    else:
        raise ValueError(f"Output type '{output_type}' is not supported.")
            
    