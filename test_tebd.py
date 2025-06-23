import numpy as np
import matplotlib.pyplot as plt
import time
from mylib import TEBD, TDVP

""" U_ZXのテスト """
# U_ZX = TEBD.U_ZX(1, 1)
# print("U_ZZ:", U_ZZ)
# print(U_ZZ @ U_ZZ.conj().T)  # ユニタリ性の確認

""" apply_bond_layerのテスト """
# L = 20
# mps, D = TEBD.all_up(L)
# TEBD.apply_bond_layer(mps, D, 100, 0.1, 1, 'odd', 'lr', 0)
# #TEBD.apply_bond_layer(mps, D, 100, 0.1, 1, 'even', 'lr', 0)
# #TEBD.apply_bond_layer(mps, D, 100, 0.1, 1, 'odd', 'rl', 0)
# #TEBD.apply_bond_layer(mps, D, 100, 0.1, 1, 'even', 'rl', 0)

""" envのテスト """
# L = 10
# mps, D = TEBD.all_up(L)
# mps, D = TEBD.all_down(L)
# mps, D = TEBD.plus(L)
# print(TEBD.expectation_z(mps,1))

""" inner_productのテスト """
# L = 10
# mps1 = TDVP.GHZ_unnormalized(L)
# mps2 = TDVP.GHZ_unnormalized(L)
# inner = TEBD.inner_product(mps1, mps2)
# print("Inner product:", inner)

""" EPRの相関 """
# mps = TDVP.GHZ_unnormalized(2)
# mps = TDVP.right_canonical(mps)
# print(TEBD.correlation(mps,0,1, 'x'))
# mps = TEBD.plus(2)[0]
# print(TEBD.correlation(mps,0,1, 'z'))

""" M_xの時間依存性テスト """
L = 20
# mps, D = TEBD.all_up(L)
# T, M_x,_,_ = TEBD.tebd2(mps, D, 32, 1, 1, 10, 100, 1e-10, 'x')
# plt.plot(T, M_x, label='tebd2 M_x')
mps, D = TEBD.all_up(L)
T, M_x, Energy = TEBD.tebd2_ver2(mps, D, 32, 1, 1, 10, 1000, 1e-10, 'x')
plt.plot(T, M_x, label='tebd2_ver2 M_x')
# mps, D = TEBD.all_up(L)
# T, M_x = TEBD.tebd1(mps, D, 32, 1, 1, 10, 100, 1e-10, 'x')
plt.plot(T, M_x, label='tebd1 M_x')

""" M_zの時間依存性テスト """
# L = 10
# mps, D = TEBD.all_up(L)
# T, M_x,_,_ = TEBD.tebd2(mps, D, 32, 1, 1, 10, 100, 1e-10, 'z')
# plt.plot(T, M_x, label='tebd2 M_z')
# mps, D = TEBD.all_up(L)
# T, M_x, Energy = TEBD.tebd2_ver2(mps, D, 32, 1, 1, 10, 100, 1e-10, 'z')
# plt.plot(T, M_x, label='tebd2_ver2 M_z')
# mps, D = TEBD.all_up(L)
# T, M_x = TEBD.tebd1(mps, D, 32, 1, 1, 10, 100, 1e-10, 'z')
# plt.plot(T, M_x, label='tebd1 M_z')

""" 相関のテスト """
# L = 10
# mps, D = TEBD.all_up(L)
# T,_,corr,_ = TEBD.tebd2(mps, D, 32, 1, 1, 20, 1000, 1e-10, 'x')
# plt.plot(T, corr, label='Correlation')

""" エネルギー保存 """
# L = 10
# mps, D = TEBD.all_up(L)
# T,_,_,Energy = TEBD.tebd2(mps, D, 32, 1, 1, 10, 1000, 1e-10, 'x')
# plt.plot(T, Energy, label='Energy')
# mps, D = TEBD.all_up(L)
# T,_,Energy = TEBD.tebd2_ver2(mps, D, 32, 1, 1, 10, 1000, 1e-10, 'x')
# plt.plot(T, Energy, label='Energy ver2')


""" TEBDのトロッター誤差のテスト """
# L = 10
# for n_steps in [20, 30, 50, 100]:
#     mps, D = TEBD.all_up(L)
#     T, M = TEBD.tebd1(mps, D, 40, 1, 1, 10, n_steps, 1e-12, 'x')
#     plt.plot(T, M, label=f'n_steps={n_steps}')

""" TEBD2のトロッター誤差のテスト """
# L = 10
# for n_steps in [10, 20, 30, 100]:
#     mps, D = TEBD.all_up(L)
#     T, M, _, _ = TEBD.tebd2(mps, D, 40, 1, 1, 10, n_steps, 1e-12, 'x')
#     plt.plot(T, M, label=f'n_steps={n_steps}')

""" トランケーション誤差のテスト """
# L = 10
# for maxbond in [4, 8, 16, 32]:
#     mps, D = TEBD.all_up(L)
#     T, M, _, _ = TEBD.tebd2(mps, D, maxbond, 1, 1, 10, 1000, 1e-10, 'x')
#     plt.plot(T, M, label=f'maxbond={maxbond}')

""" グラフの表示設定 """
plt.xlabel('Time')
plt.ylabel('Magnetization')
plt.title('Magnetization vs Time')
# plt.ylabel('correlation')
# plt.title('Correlation vs Time')
plt.grid()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout() 
plt.show()




