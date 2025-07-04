import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import numpy as np
import matplotlib.pyplot as plt 
from mylib import TEBD, MPS

# 各種パラメータの設定
L = 10
h = 1.0
J = 1.0
total_time = 10.0
n_steps = 1000


# グラフの保存先のパスを設定
base = os.path.dirname(os.path.dirname(os.getcwd()))
folder = os.path.join(base, "results", "ising_transverse")
filepath = os.path.join(folder, f'M_tebd_h={h}_J={J}_L={L}_T={total_time}_n_steps={n_steps}.png')


mps = MPS.all_up(L)
M_x = TEBD.tebd2(mps, h=h, J=J, T=total_time, n_steps=n_steps, output_type = 'M_x', clone=True)
M_z = TEBD.tebd2(mps, h=h, J=J, T=total_time, n_steps=n_steps, output_type = 'M_z', clone=True)
T = np.linspace(0, total_time, n_steps + 1)
plt.plot(T, M_x, label='tebd2 M_x')
plt.plot(T, M_z, label='tebd2 M_z')

# グラフの表示設定
plt.xlabel('Time')
plt.ylabel('M')
plt.title('Time Evolution of M')
plt.legend()
plt.grid()


# グラフを保存
plt.savefig(filepath)
