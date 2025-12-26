from impl.tank import Tank
import numpy as np
import matplotlib.pyplot as plt

tank = Tank(0.2, 0.5, 0.1, tank_size=20)
tank.run(20001, "test.log", 100)

#step = np.load("./log/log_step_200_lt.npy")
#plt.imshow(step[:, :, 0], cmap="viridis")
#plt.colorbar(label="value")
#plt.axis("off")
#plt.show()

# molecule.py の directions と同じ順番
DIR8 = np.array([[-1, -1], [-1, 0], [-1, 1],
                 [ 0,  1], [ 1, 1], [ 1, 0],
                 [ 1, -1], [ 0, -1]], dtype=float)

SOAP, WATER, AIR = 1, 2, 3

def plot_lattice_with_soap_arrows(step, every=1, arrow_scale=0.6, origin="lower"):
    """
    step: (H,W,2) で step[...,0]=kind, step[...,1]=dir
    every: 間引き（重い/見づらい時に 2,3...）
    """
    kind = step[..., 0].astype(int)
    d = step[..., 1].astype(int)

    H, W = kind.shape
    yy, xx = np.mgrid[0:H, 0:W]

    # soap だけ
    m = (kind == SOAP)

    # 間引き
    if every > 1:
        m = m & ((yy % every == 0) & (xx % every == 0))

    # quiver 用の成分（imshow座標に合わせる）
    # DIR8 は (drow, dcol) = (dy, dx) なので
    # U=dx, V=dy に入れる。ただし origin="lower" ならそのままでOK。
    dx = DIR8[d, 1]
    dy = DIR8[d, 0]

    U = np.where(m, dx, np.nan)
    V = np.where(m, dy, np.nan)

    plt.figure(figsize=(6, 6))
    plt.imshow(kind, cmap="viridis", origin=origin)
    plt.colorbar(label="kind (Soap=1, Water=2, Air=3)")
    plt.quiver(xx, yy, U, V,
               angles="xy", scale_units="xy", scale=1/arrow_scale,
               width=0.006, headwidth=3.5, headlength=4.5, facecolor="red")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# 例:
step = np.load("./log/log_step_20000_lt.npy")   # (H,W,2)
plot_lattice_with_soap_arrows(step, every=1, arrow_scale=0.7)
