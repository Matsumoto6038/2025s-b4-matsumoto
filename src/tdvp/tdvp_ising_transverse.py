import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mylib import TDVP, MPS
import time

L = 10
T = 10
n_steps = 100

# mpsの初期状態の生成
mps = MPS.all_up(L)

# mpoの生成
mpo = MPS.mpo_ising_transverse(L, h=1, J=1)

# 実行時間の測定
start = time.time()

# TDVPの実行とx方向の磁化の計算
M_x = TDVP.tdvp(mps, mpo, T=T, n_steps=n_steps, output_type='M_x', cutoff=1e-10, maxbond=32, clone = True)

# 実行時間の測定終了
end = time.time()
print(f"TDVP: {end - start:.2f} seconds")

# 計算結果のプロット
Time = np.linspace(0, T, n_steps + 1)
plt.plot(Time, M_x, label='TDVP')

# グラフの表示設定
plt.xlabel('Time')
plt.ylabel('Magnetization')
plt.legend()
plt.title('Magnetization vs Time')
plt.grid()
plt.show()
