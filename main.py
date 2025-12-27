from impl.tank import Tank
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

#soap_ratio = 0.2
#water_ratio = 0.25
#tank_size = 50
#save_step_num = 10
#hot_step_num = 200
#hot_tmp = 15.0
#cold_step_num = 100
#cold_tmp = 0.1
#loop_num = 50
#
#print("loop0 hot")
#tank = Tank(soap_ratio, water_ratio, hot_tmp, tank_size=tank_size, seed=0)
#tank.run(hot_step_num+1, "loop0_hot", save_step_num)
#print("loop0 cold")
#restart = np.load("./log/loop0_hot_step_{}.npy".format(hot_step_num))
#tank = Tank(soap_ratio, water_ratio, hot_tmp, tank_size=tank_size, seed=1, restart=restart)
#tank.run(cold_step_num+1, "loop0_cold", save_step_num)
#
#for loop_idx in range(1, loop_num+1):
#    print("loop{} hot".format(loop_idx))
#    restart = np.load("./log/loop{}_cold_step_{}.npy".format(loop_idx-1, cold_step_num))
#    tank = Tank(soap_ratio, water_ratio, hot_tmp, tank_size=tank_size, seed=2*loop_idx, restart=restart)
#    tank.run(hot_step_num+1, "loop{}_hot".format(loop_idx), save_step_num)
#    print("loop{} cold".format(loop_idx))
#    restart = np.load("./log/loop{}_hot_step_{}.npy".format(loop_idx, hot_step_num))
#    tank = Tank(soap_ratio, water_ratio, cold_tmp, tank_size=tank_size, seed=2*loop_idx+1, restart=restart)
#    tank.run(cold_step_num+1, "loop{}_cold".format(loop_idx), save_step_num)

#tank = Tank(0.3, 0.35, 1000.0, tank_size=50)
#tank.run(1001, "isa_first", 100)
#restart = np.load("./log/isa_first_step_100.npy")
#tank = Tank(0.3, 0.35, 1.0, tank_size=50, restart=restart)
#tank.run(1001, "isa_second", 100)
#restart = np.load("./log/isa_second_step_100.npy")
#tank = Tank(0.3, 0.35, 0.2, tank_size=50, restart=restart)
#tank.run(1001, "isa_third", 100)
#restart = np.load("./log/isa_third_step_100.npy")
#tank = Tank(0.3, 0.35, 0.1, tank_size=50, restart=restart)
#tank.run(1001, "isa_fourth", 100)

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

    # quiver 用の成分
    dx = DIR8[d, 1]
    dy = DIR8[d, 0]

    U = np.where(m, dx, np.nan)
    V = np.where(m, dy, np.nan)
    #V = -V  # imshow 座標補正

    # === ここが色指定の本体 ===
    # index: 0は未使用（ダミー）
    cmap = ListedColormap([
        "black",        # 0 (unused)
        "orange",       # 1 = SOAP
        "#7ec8e3",       # 2 = WATER (水色)
        "#b0b0b0",       # 3 = AIR (灰色)
    ])
    norm = BoundaryNorm([0, 1, 2, 3, 4], cmap.N)

    plt.figure(figsize=(6, 6))
    plt.imshow(kind, cmap=cmap, norm=norm, origin=origin)
    plt.colorbar(
        ticks=[1, 2, 3],
        label="kind (Soap=1, Water=2, Air=3)"
    )

    plt.quiver(
        xx, yy, U, V,
        angles="xy", scale_units="xy", scale=1/arrow_scale,
        width=0.006, headwidth=3.5, headlength=4.5,
        color="red"
    )

    plt.axis("off")
    plt.tight_layout()
    plt.show()

#def plot_lattice_with_soap_arrows(step, every=1, arrow_scale=0.6, origin="lower"):
#    """
#    step: (H,W,2) で step[...,0]=kind, step[...,1]=dir
#    every: 間引き（重い/見づらい時に 2,3...）
#    """
#    kind = step[..., 0].astype(int)
#    d = step[..., 1].astype(int)
#
#    H, W = kind.shape
#    yy, xx = np.mgrid[0:H, 0:W]
#
#    # soap だけ
#    m = (kind == SOAP)
#
#    # 間引き
#    if every > 1:
#        m = m & ((yy % every == 0) & (xx % every == 0))
#
#    # quiver 用の成分（imshow座標に合わせる）
#    # DIR8 は (drow, dcol) = (dy, dx) なので
#    # U=dx, V=dy に入れる。ただし origin="lower" ならそのままでOK。
#    dx = DIR8[d, 1]
#    dy = DIR8[d, 0]
#
#    U = np.where(m, dx, np.nan)
#    V = np.where(m, dy, np.nan)
#    V = -V
#
#    plt.figure(figsize=(6, 6))
#    plt.imshow(kind, cmap="viridis", origin=origin)
#    plt.colorbar(label="kind (Soap=1, Water=2, Air=3)")
#    plt.quiver(xx, yy, U, V,
#               angles="xy", scale_units="xy", scale=1/arrow_scale,
#               width=0.006, headwidth=3.5, headlength=4.5, facecolor="red")
#    plt.axis("off")
#    plt.tight_layout()
#    plt.show()


# 例:
#for i in range(0, 51, 10):
step = np.load("./log/loop44_cold_step_{}.npy".format(100))   # (H,W,2)
plot_lattice_with_soap_arrows(step, every=1, arrow_scale=1.5)
