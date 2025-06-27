import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import numpy as np
import matplotlib.pyplot as plt

nx = 3
ny = 3
h = 0.5
bias = 0
seed = 12345
n_steps = 200
total_time = 100

level = 10  # 取得する励起状態の数

# 対角化の結果の読み込み
base = os.path.dirname(os.path.dirname(os.getcwd()))
folder = os.path.join(base, "results", "annealing")
filepath = os.path.join(folder, f"exact_diag_nx={nx}_ny={ny}_h={h}_bias={bias}_seed={seed}.txt")

# 基底状態と第一励起状態のエネルギーをプロット
data = np.loadtxt(filepath, dtype=float)
time = np.linspace(0, total_time, len(data[:,0]))
for i in range(level):
    plt.plot(time, data[:,i], label=f'{i} state')

plt.xlabel('Time')
plt.ylabel('Energy')
# plt.legend()
plt.grid()
plt.show()