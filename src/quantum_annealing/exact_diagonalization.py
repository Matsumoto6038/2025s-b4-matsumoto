import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from mylib import annealing, MPS

""" 量子アニーリングに用いたハミルトニアンを各時刻で対角化し、固有エネルギーを出力する。 """

#各種パラメータの設定
nx = 3          # スピングラスのx方向のサイト数
ny = 3          # スピングラスのy方向のサイト数
h = 0.5         # 横磁場の大きさ
bias = 0        # バイアス磁場の大きさ
seed = 12345    # 乱数のシード

n_steps = 100   # 取得データ数

# 相互作用定数を生成
J_holiz, J_vert = MPS.generate_J_array(nx, ny, seed)

# 対角化の固有値の保存先のパスを設定
base = os.path.dirname(os.path.dirname(os.getcwd()))
folder = os.path.join(base, "results", "annealing")
filepath = os.path.join(folder, f"exact_diag_nx={nx}_ny={ny}_h={h}_bias={bias}_seed={seed}.txt")

# 対角化の実行と結果の書き込み
with open(filepath, "w") as f:
    for i in range(n_steps + 1):
        weight = annealing.weight_func(i, n_steps)
        data = annealing.annealing_energy(nx, ny, bias, h, J_holiz, J_vert, weight)
        line = " ".join(map(str,data))
        f.write(line + "\n")
