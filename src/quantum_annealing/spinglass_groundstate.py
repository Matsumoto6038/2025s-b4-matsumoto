import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import numpy as np
import matplotlib.pyplot as plt
from mylib import MPS, TDVP, annealing
from collections import Counter

nx = 3              # スピングラスのx方向のサイト数
ny = 3              # スピングラスのy方向のサイト数
h = 0.5             # 横磁場の大きさ
bias = 0.1            # バイアス磁場の大きさ
seed = 12345        # 乱数のシード

n_steps = 200       # 時間分割数
total_time = 100   # アニーリングの総時間

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
for i in range(n_steps):
    weight = annealing.weight_func(i, n_steps)
    mpo = MPS.spin_glass_annealing(nx, ny, h=h, J_holiz=J_holiz, J_vert=J_vert, weight=weight, bias=bias)
    for j in range(4):
        TDVP.sweep(mps=mps, mpo=mpo, dt=rate/4, Left=Left, Right=Right, maxbond=30, cutoff=0)
    TDVP.progress(i, n_steps)

# 全ビットを測定する
shots = 1000
threshold = 0.05  # ヒストグラムに表示するための閾値
mps = MPS.right_canonical(mps)  
results = [MPS.measure_all_bits(mps, check=False) for _ in range(shots)]

# 結果をヒストグラムで出力
counts = Counter(results)
total = sum(counts.values())
new_counter = Counter()
other_count = 0

for bitstring, count in counts.items():
    if count / total < threshold:
        other_count += count
    else:
        new_counter[bitstring] = count

if other_count > 0:
    new_counter["others"] = other_count

labels = sorted(new_counter)
values = [new_counter[label] for label in labels]

plt.bar(labels, values)
plt.xlabel('Bitstring')
plt.ylabel('Counts')
plt.title(f'Measurement Results for {shots} Shots')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()  
