import numpy as np

""" スピングラスの全数探索 """

# 自然数を2進数に変換し、さらに、0->1、1->-1 とした配列を返す。
# 0 -> [1,...,1], 1 -> [1,...,1,-1], ...
def state_to_bit(state: int, nx: int, ny: int):
    if state > 2**(nx*ny) - 1:
        raise ValueError(f"state must be in the range [0, {2**(nx*ny) - 1}].")
    bit = bin(state)[2:].zfill(nx*ny)  # 2進数に変換し、L*L桁に0埋め
    return np.array([1 - int(b) * 2 for b in bit])

# スピングラスのコスト関数(負符号込み)
def make_cost_func(nx, ny, bias, J_holiz, J_vert):
    def cost_func(
        state
    ):
        bit_list = state_to_bit(state, nx, ny)
        nnx_prod = bit_list[:nx*ny-1] * bit_list[1:]
        nny_prod = bit_list[:nx*ny-nx] * bit_list[nx:]
        cost = np.dot(nnx_prod, J_holiz[:nx*ny-1]) + np.dot(nny_prod, J_vert[:nx*ny-nx])
        cost += bias * np.sum(bit_list)
        return -cost
    return cost_func


""" 厳密対角化によるアニーリングの追跡を行うための関数 """
def weight_func(i, n_steps):
    return np.sin((np.pi * i) / (2 * n_steps)) ** 2

# スピングラスハミルトニアン(対角化)
def diag_factor(
    nx,         # 格子の横のサイズ
    ny,         # 格子の縦のサイズ
    bias,       # バイアス
    J_holiz,    # 横方向の結合定数
    J_vert      # 縦方向の結合定数
):  
    cost_func = make_cost_func(nx, ny, bias, J_holiz, J_vert)
    diag = np.array([])
    for state in range(2**(nx*ny)):
        diag = np.append(diag, cost_func(state))
    # 順番を逆にする。0 <-> 0000 <-> 
    diag = np.flip(diag) 
    return diag

# 各サイトの横磁場項を生成
def x_onsite(i, nx, ny):
    sigma_x = np.array([[0,1],[1,0]])
    identity = np.eye(2)
    mat = np.eye(1)
    for j in range(nx*ny):
        if j == i:
            mat = np.kron(mat, sigma_x)
        elif j != i:
            mat = np.kron(mat, identity)
    return mat

# annealing 過程のエネルギー固有値(weightが時間の情報を含む)
def annealing_energy(nx, ny, bias, h, J_holiz, J_vert, weight):
    Hamiltonian = np.zeros((2**(nx*ny), 2**(nx*ny)))
    for i in range(nx*ny):
        Hamiltonian += (1-weight) * h * x_onsite(i, nx, ny)
    Hamiltonian += weight * np.diag(diag_factor(nx, ny, bias, J_holiz, J_vert))
    eig_vals, _ = np.linalg.eigh(Hamiltonian)
    return eig_vals

