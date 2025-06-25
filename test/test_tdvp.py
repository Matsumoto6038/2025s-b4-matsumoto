import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mylib import TDVP, TEBD, MPS
import time

""" mpo_ising_transverseのテスト """
# mpo = TDVP.mpo_ising_transverse(4, 1, 1)
# name = ['bond_left', 'phys_bra', 'phys_ket', 'bond_right']
# for i in range(len(mpo)):
#     output = []
#     for j in range(len(name)):
#         output.append(f"{name[j]}:{mpo[i].shape[j]}")
#     print(f'mpo[{i}]: {output}")

""" GHZ_unnormalizedのテスト """
# mps = TDVP.GHZ_unnormalized(4)
# for i in range(len(mps)):
#     print(f"mps[{i}]: {mps[i].shape}")
# mps = TDVP.mps_random(4, [1, 2, 3, 2, 1])
# for i in range(len(mps)):
#     print(f"mps_random[{i}]: {mps[i].shape}")

""" ボンド情報の取得 """
# mps = TDVP.GHZ_unnormalized(4)
# bond_info = TDVP.get_bondinfo(mps) 
# print(f"Bond dimensions: {bond_info}")

""" QR分解のテスト """
# A = np.random.rand(4, 3)
# q,r = np.linalg.qr(A)
# print(q.conj().T @ q)  # Check orthogonality
# print(A - q @ r)  # Check if A is reconstructed correctly

""" LQ分解のテスト"""
# A = np.random.rand(4, 3)
# q, r = np.linalg.qr(A.T)
# print(q.T @ q.conj())  # Check orthogonality
# print(A - r.T @ q.T)  # Check if A is reconstructed correctly

""" ランダムMPSとright_canonicalのテスト """
# L = 6
# D = [1,2,4,5,4,2,1]
# mps = TDVP.mps_random(L, D)
# mps = TDVP.right_canonical(mps)
# print(TDVP.check_right_canonical(mps))
# print(np.einsum('ijk,ijk->', mps[0], mps[0]))  

""" initial right environment """
# L = 10
# mps = TDVP.mps_random(L, [1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1])
# mps = TDVP.right_canonical(mps)
# mpo = TDVP.mpo_ising_transverse(L, 1, 1)
# Right = TDVP.initial_right_env(mps, mpo)
# for i in range(len(Right)):
#     print(f"Right[{i+2}]: {Right[i].shape}")

""" initial left environment """
# L = 4
# Left = TDVP.initial_left_env(L)
# print(Left[0].shape)

""" H_effのテスト """
# L = 10
# mpo = TDVP.mpo_ising_transverse(L, 1, 1)
# mps = TDVP.GHZ_unnormalized(L)
# mps = TDVP.right_canonical(mps)
# Left = TDVP.initial_left_env(L)
# Right = TDVP.initial_right_env(mps, mpo)
# H_eff = TDVP.H_eff2(mpo, Left, Right, 0)
# print(f'H_eff.shape: {H_eff.shape}')

""" np.linalg.eigh の実行時間 """
# start = time.time()
# np.linalg.eigh(np.random.rand(10**2*4, 10**2*4))  
# np.linalg.eigh(np.random.rand(10*4, 10*4))
# end = time.time()
# print(f"time taken: {20*(end - start):.2f} seconds")

""" TDVPのテスト """
L = 10
mps = MPS.all_up(L)
mpo = MPS.mpo_ising_transverse(L, h=1, J=1)
# mpo = MPS.mpo_xxz(L, h=1, Delta=2)
start = time.time()
n_steps = 100
T = 10
Time = np.linspace(0, T, n_steps + 1)
M_x = TDVP.tdvp(mps, mpo, T=T, n_steps=n_steps, output_type='M_x', cutoff = 1e-6,clone = True)
end = time.time()
print(f"TDVP: {end - start:.2f} seconds")
plt.plot(Time, M_x, label='TDVP')

""" TEBDとの比較 """
# start = time.time()
# M_x = TEBD.tebd2(mps, h=1, J=1, T=T, n_steps=n_steps, output_type='M_x')
# plt.plot(Time, M_x, label='TEBD')
# end = time.time()
# print(f"TEBD: {end - start:.2f} seconds")

""" グラフの表示設定 """
plt.xlabel('Time')
plt.ylabel('Magnetization')
plt.legend()
plt.title('Magnetization vs Time')
plt.grid()
plt.show()

