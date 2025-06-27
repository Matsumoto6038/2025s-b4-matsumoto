import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import numpy as np
import matplotlib.pyplot as plt
from mylib import MPS, TDVP
import time


""" spin_glass_annealingのテスト """
# L = 3
# h = 2
# n_steps = 100

# J_holiz = np.array([2]*L*L)
# J_vert = np.array([0]*L*L)
# weight = 0.5
# mpo = MPS.spin_glass_annealing(L, h=h, J_holiz=J_holiz, J_vert=J_vert, weight=weight)

# mps = MPS.all_up(L*L)
# rate = float(1/n_steps)
# Left = TDVP.initial_left_env(len(mps))
# Right = TDVP.initial_right_env(mps, mpo)

# timer = np.linspace(0, 10, n_steps+1)
# result = TDVP.tdvp(mps, mpo, 10, n_steps, output_type='M_x', cutoff=1e-6, clone=True)
# plt.plot(timer, result, label='annealing')

# # 以下のTDVPと同じ挙動
# L = 9
# mps = MPS.all_up(L)
# mpo = MPS.mpo_ising_transverse(L, h=1, J=1)
# n_steps = 100
# T = 10
# Time = np.linspace(0, T, n_steps + 1)
# M_x = TDVP.tdvp(mps, mpo, T=T, n_steps=n_steps, output_type='M_x', cutoff = 1e-6,clone = True)
# plt.plot(Time, M_x, label='TDVP')


# plt.legend()
# plt.grid()
# plt.show()

""" spin_glass """
# L = 2
# h = 1
# i = 1
# J_holiz, J_vert = MPS.generate_J_array(L=L, seed=1234)
# mpo = MPS.spin_glass_annealing(L=L, h=h, J_holiz=J_holiz, J_vert=J_vert, weight=0)
# print(f"mpo[i]: {mpo[i].transpose(0,3,1,2)}")
# mpo = MPS.spin_glass_annealing(L=L, h=h, J_holiz=J_holiz, J_vert=J_vert, weight=0.5)
# print(f"mpo[i]: {mpo[i].transpose(0,3,1,2)}")

""" annealing """
L = 2
h = 0.2
n_steps = 1000
total_time = 500
rate = float(total_time/n_steps)

J_holiz, J_vert = MPS.generate_J_array(L=L, seed=12345)
print(f"J_holiz: {J_holiz}")
print(f"J_vert: {J_vert}")
J_holiz = np.array([1]*L*L)  # 水平方向の相互作用は定数に設定
J_vert = np.array([1]*L*L)  # 垂直方向の相互作用はゼロに設定
for i in range(L):
    J_holiz[i * L - 1] = 0  
    J_vert[L * (L-1) + i] = 0
mpo = MPS.spin_glass_annealing(L=L, h=h, J_holiz=J_holiz, J_vert=J_vert, weight=0)
mps = MPS.plus(L*L)

Left = TDVP.initial_left_env(len(mps))
Right = TDVP.initial_right_env(mps, mpo)

result = []

start = time.time()
for i in range(0,n_steps):
    weight = (np.sin(np.pi * i / (2*n_steps)))**2  
    # weight = i * rate  
    # weight = (np.tanh((i - n_steps / 2) / (n_steps / 10)) + 1) / 2
    mpo = MPS.spin_glass_annealing(L, h=h, J_holiz=J_holiz, J_vert=J_vert, weight=weight)
    result.append(MPS.energy(mps, mpo))  
    for j in range(4):
        TDVP.sweep(mps=mps, mpo=mpo, dt=rate/4, Left=Left, Right=Right, maxbond=30, cutoff=0)
        result.append(MPS.energy(mps, mpo))
    TDVP.progress(i, n_steps)
    if i % (n_steps // 10) == 0:
        cleaned = [int(x) for x in MPS.get_bondinfo(mps)]
        print(f"Bond dimensions: {cleaned}")
end = time.time()
print(f"Annealing time: {end - start:.2f} seconds")
# 各ビットの期待値を計算し、小数点以下3桁で表示
print("Expectation values of each bit:")
for i in range(L*L):
    print(f"bit{i}:{MPS.expval('z', mps, 0)[i]:.3f}")

timer = np.linspace(0, total_time, len(result))
plt.plot(timer, result, label='annealing')
plt.grid()
plt.show()

""" 全数探索 """
# L = 3
# def state_to_bit(state: int, L: int):
#     if state > 2**(L*L) - 1:
#         raise ValueError(f"state must be in the range [0, {2**(L*L) - 1}].")
#     bit = bin(state)[2:].zfill(L*L)  # 2進数に変換し、L*L桁に0埋め
#     return np.array([1 - int(b) * 2 for b in bit])

# def make_cost_func(J_holiz, J_vert, bias, L):
#     def cost_func(
#         state
#     ):
#         bit_list = state_to_bit(state, L)
#         nnx_prod = bit_list[:L*L-1] * bit_list[1:]
#         nny_prod = bit_list[:L*L-L] * bit_list[L:]
#         cost = np.dot(nnx_prod, J_holiz[:L*L-1]) + np.dot(nny_prod, J_vert[:L*L-L])
#         cost += bias * np.sum(bit_list)
#         return cost
#     return cost_func

# J_holiz, J_vert = MPS.generate_J_array(L=L, seed=12345)
# print(f"J_holiz: {J_holiz}")
# print(f"J_vert: {J_vert}")
# cost_func = make_cost_func(J_holiz, J_vert, 1/L/L, L)

# min_energy = -L*L
# min_state = 0
# for i in range(2**(L*L)):
#     if min_energy <= cost_func(i):
#         min_energy = cost_func(i)
#         min_state = i
#         print(f"state: {i}, min_energy: {-min_energy}")
# print(f"min_state: {min_state}, min_energy: {-min_energy}")
# print(f"min_state: {state_to_bit(178,L)}")
# print(f"min_state bit: {state_to_bit(269, L)}")

""" L=3, seed=12345 の場合
J_holiz: [-1  1  0  1 -1  0 -1 -1  0]
J_vert: [-1  1  1 -1  1  1  0  0  0]で、
基底は2重に縮退しており、
[ 1 -1  1 -1 -1  1  1 -1  1]
[-1  1  1  1  1 -1 -1  1 -1]
の2つが最小エネルギー状態。
アニーリングの結果は、エネルギーはこれと一致するが、得られる状態はこれらの重ね合わせである。
[-0.238, 0.238, 0.999, 0.238, 0.238, -0.238, -0.238, 0.238, -0.238]
第二ビットだけ共通なので、期待値は1である。
"""