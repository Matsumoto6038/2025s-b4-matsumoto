import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mylib import TDVP, MPS
import time

# 各種パラメータの設定
L = 10
h = 1.0
J = 1.0
total_time = 10.0
n_steps = 1000

# mpsの初期状態の生成
mps = MPS.all_up(L)

# mpoの生成
mpo = MPS.mpo_ising_transverse(L, h=h, J=J)

# TDVPの実行とx方向の磁化の計算
M_x = TDVP.tdvp(mps, mpo, T=total_time, n_steps=n_steps, output_type='M_x', cutoff=1e-10, maxbond=32, clone = True)
M_z = TDVP.tdvp(mps, mpo, T=total_time, n_steps=n_steps, output_type='M_z', cutoff=1e-10, maxbond=32, clone = True)

# 計算結果のプロット
Time = np.linspace(0, total_time, n_steps + 1)
plt.plot(Time, M_x, label='TDVP M_x')
plt.plot(Time, M_z, label='TDVP M_z')

# グラフの表示設定
plt.xlabel('Time')
plt.ylabel('Magnetization')
plt.legend()
plt.title('Magnetization vs Time')
plt.grid()

# グラフの保存先のパスを設定
base = os.path.dirname(os.path.dirname(os.getcwd()))
folder = os.path.join(base, "results", "ising_transverse")
filepath = os.path.join(folder, f"M_tdvp_h={h}_J={J}_L={L}_T={total_time}_n_steps={n_steps}.png")

# グラフを保存
plt.savefig(filepath)