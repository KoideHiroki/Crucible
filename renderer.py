import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# ===== 設定 =====
LOG_DIR = "./log"
OUT_DIR = "./png"
SCALE = 6        # 1セル何px相当にするか
ARROW_SCALE = 0.4

os.makedirs(OUT_DIR, exist_ok=True)

# 8方向（あなたの定義と一致）
DIRS = np.array([
    [-1, -1], [-1, 0], [-1, 1],
    [ 0, 1],  [ 1, 1],  [ 1, 0],
    [ 1,-1],  [ 0,-1]
])

# 種類 → 色
COLORS = {
    1: np.array([1.0, 0.5, 0.0]),  # Soap : orange
    2: np.array([0.2, 0.4, 1.0]),  # Water: blue
    3: np.array([0.95,0.95,0.95])  # Air  : white
}

def render(arr, out_path):
    H, W, _ = arr.shape

    img = np.zeros((H, W, 3))

    for k, col in COLORS.items():
        img[arr[:,:,0] == k] = col

    fig, ax = plt.subplots(figsize=(W/SCALE, H/SCALE), dpi=200)
    ax.imshow(img, interpolation="nearest")

    # Soap の向き描画
    ys, xs = np.where(arr[:,:,0] == 1)
    for y, x in zip(ys, xs):
        d = int(arr[y, x, 1])
        dy, dx = DIRS[d]
        ax.arrow(x, y, dx*ARROW_SCALE, dy*ARROW_SCALE,
                 head_width=0.8, head_length=0.8,
                 fc="black", ec="black", linewidth=1.0)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, W-0.5)
    ax.set_ylim(H-0.5, -0.5)

    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=200)
    plt.close()


# ===== 実行部 =====
files = sorted(glob.glob(os.path.join(LOG_DIR, "*_step_*.npy")))
print(len(files))
for f in files:
    name = os.path.splitext(os.path.basename(f))[0]
    #out = os.path.join(OUT_DIR, name + ".png")
    out = os.path.join(OUT_DIR, "out_frame_{}.png".format(int(name.split("_")[-1])//10))
    arr = np.load(f)
    print("render:", name)
    render(arr, out)

print("done.")
