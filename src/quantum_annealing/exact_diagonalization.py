import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from mylib import annealing, MPS

#各種パラメータの設定
nx = 2          # スピングラスのx方向のサイト数
ny = 2          # スピングラスのy方向のサイト数
h = 0.2         # 横磁場の大きさ
bias = 0.5      # バイアス磁場の大きさ
seed = 12345    # 乱数のシード

n_steps = 100   # 取得データ数

# 相互作用定数を生成
J_holiz, J_vert = MPS.generate_J_array(nx, ny, seed)

base = os.path.dirname(os.path.dirname(os.getcwd()))
folder = os.path.join(base, "results", "annealing")
filepath = os.path.join(folder, f"exact_diag_nx={nx}_ny={ny}_h={h}_bias={bias}_seed={seed}.txt")

with open(filepath, "w") as f:
    for i in range(n_steps + 1):
        weight = annealing.weight_func(i, n_steps)
        data = annealing.annealing_energy(nx, ny, bias, h, J_holiz, J_vert, weight)
        line = "".join(map(str,data))
        f.write(line + "\n")
