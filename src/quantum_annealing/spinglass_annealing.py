import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import numpy as np
import matplotlib.pyplot as plt
from mylib import MPS, TDVP, annealing

""" スピングラスの量子アニーリングを行い、エネルギーの時間変化をプロットする。 """

# 各種パラメータの設定
nx = 3              # スピングラスのx方向のサイト数
ny = 3              # スピングラスのy方向のサイト数
h = 0.5             # 横磁場の大きさ
bias = 0.1            # バイアス磁場の大きさ
seed = 12345        # 乱数のシード

n_steps = 100       # 時間分割数
total_time = 50    # アニーリングの総時間

# 相互作用定数を乱数から生成
rate = float(total_time / n_steps)
J_holiz, J_vert = MPS.generate_J_array(nx=nx, ny=ny, seed=12345)

# 初期状態を用意
mps = MPS.plus(nx*ny)
mpo = MPS.spin_glass_annealing(nx=nx, ny=ny, h=h, J_holiz=J_holiz, J_vert=J_vert, weight=0, bias=bias)

# 環境行列を初期化
Left = TDVP.initial_left_env(nx*ny)
Right = TDVP.initial_right_env(mps, mpo)

# 量子アニーリング中のエネルギーを保存するリスト
results = []

# Annealingの実行
results.append(MPS.energy(mps, mpo))  
for i in range(n_steps):
    weight = annealing.weight_func(i, n_steps)
    mpo = MPS.spin_glass_annealing(nx, ny, h=h, J_holiz=J_holiz, J_vert=J_vert, weight=weight, bias=bias)
    for j in range(4):
        TDVP.sweep(mps=mps, mpo=mpo, dt=rate/4, Left=Left, Right=Right, maxbond=30, cutoff=0)
        results.append(MPS.energy(mps, mpo))
    TDVP.progress(i, n_steps)
    
# 量子アニーリング中のエネルギーをプロット
time = np.linspace(0, total_time, len(results))
plt.plot(time, results, label='Annealing Energy')

# 対角化の固有値の保存先のパスを設定
base = os.path.dirname(os.path.dirname(os.getcwd()))
folder = os.path.join(base, "results", "annealing")
filepath = os.path.join(folder, f"exact_diag_nx={nx}_ny={ny}_h={h}_bias={bias}_seed={seed}.txt")

# 対角化により計算した基底状態と励起状態のエネルギーをプロット
data = np.loadtxt(filepath, dtype=float)
time = np.linspace(0, total_time, len(data[:,0]))
plt.plot(time, data[:,0], label='ground state')
plt.plot(time, data[:,1], label='first excited')
plt.plot(time, data[:,2], label='second excited')

plt.xlabel('Time')
plt.ylabel('Energy')
plt.legend()
plt.grid()

# エネルギーの時間変化のグラフの保存先のパスを設定
base = os.path.dirname(os.path.dirname(os.getcwd()))
folder = os.path.join(base, "results", "annealing")
filepath = os.path.join(folder, f"annealing_energy_nx={nx}_ny={ny}_h={h}_bias={bias}_seed={seed}.png")

# グラフを保存
plt.savefig(filepath)