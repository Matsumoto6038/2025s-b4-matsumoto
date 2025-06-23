import numpy as np
from scipy.sparse.linalg import LinearOperator, expm_multiply
import time
from scipy.linalg import eigh_tridiagonal


# パラメータ
N = 1000                 # 内部点の数
h = 1.0 / (N + 1)       # 格子幅
# tridiagonal の対角と非対角成分を指定
d = 2.0 / h**2 * np.ones(N)       # 対角成分
e = -1.0 / h**2 * np.ones(N - 1)  # 非対角（上下同じ）
# 固有値・固有ベクトルを計算（scipy の tridiagonal solver 使用）
start = time.time()
eigvals, eigvecs = eigh_tridiagonal(d, e)
end = time.time()
print(f"Eigenvalues calculated in {end - start:.6f} seconds")


N = 1000
# A @ x を定義する関数（ここでは対角が1の単位行列とする例）
def matvec(v):
    return -np.roll(v, -1) + 2*v - np.roll(v, 1)
# A = I として、exp(I)x = e * x
# matvec が 1次元にも2次元にも対応できるようにしておくと安全
A_op = LinearOperator(
    shape=(N, N),
    matvec=matvec,              # 必須！
    rmatvec=matvec              # （省略可だけどあると安全）
)

v0 = np.ones(N)

# 行列指数作用を計算
start = time.time()
traceA = 2 * N  # 例えばラプラシアンなら対角は 2/h^2
result = expm_multiply(A_op, v0, traceA = traceA)
end = time.time()

print(f"Time taken: {end - start:.6f} seconds")

v = np.random.rand(5*2*2*5)
L = np.random.rand(5, 5, 3)
R = np.random.rand(3, 5, 5)
H = np.random.rand(3, 2, 2, 3) 
def H_eff2(v):
    v = v.reshape(5, 2, 2, 5)
    v = np.einsum('cik,iabj->cabjk', L, v)
    v = np.einsum('cabjk,kdal->cdbjl', v, H)
    v = np.einsum('cdbjl,lebm->cdejm', v, H)
    v = np.einsum('cdejm,mfj->cdef', v, R)
    return v.reshape(5,2,2,5)

H_op = LinearOperator(
    shape=(5*2*2*5, 5*2*2*5),
    matvec = H_eff2,
    rmatvec = H_eff2  # 省略可だけどあると安全
)
result = expm_multiply(H_op, v)
print(result)
