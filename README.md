# 🧪 Soap–Water–Air Lattice Simulation  
**Local-rule-based self-organization of amphiphilic molecules**

---

## 概要

[movie.mp4](https://private-user-images.githubusercontent.com/251115363/532700267-8583112b-4d3f-407e-8bea-43244e1dc05a.mp4?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Njc3Njk0NDAsIm5iZiI6MTc2Nzc2OTE0MCwicGF0aCI6Ii8yNTExMTUzNjMvNTMyNzAwMjY3LTg1ODMxMTJiLTRkM2YtNDA3ZS04YmVhLTQzMjQ0ZTFkYzA1YS5tcDQ_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwMTA3JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDEwN1QwNjU5MDBaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT01ODlmMGY2M2RiYmFhYzBlMTQzNWZhNmZiYzEwOThkMmExOWFiMWMyMTUwNmZkYmQ4NDZmODBmNDU0Mzg4Yzg1JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.k4Ylpw7wc8uZWISFCoZfe_od6dgwzzJDOa5HE4xgnLI)

step 0
<img width="4000" height="4000" alt="Image" src="https://github.com/user-attachments/assets/ce7466dd-4423-462f-a858-9d58b8079a04">

step 1000
<img width="4000" height="4000" alt="Image" src="https://github.com/user-attachments/assets/8399e306-7052-4680-8a76-082713fa5dea">

本プロジェクトは、**石鹸分子・水分子・空気**からなる2次元格子モデルに対して、  
**局所エネルギールールとMCMC（Metropolis法）だけ**を用いて、  
ミセル様構造・界面配向・曲率構造が自発的に形成されるかを検証するシミュレーションです。

特徴は：

- 連続ポテンシャルなし  
- 力・距離なし  
- 分子は「種類」と「向き」だけを持つ  
- 相互作用は**意味論的ルールのみ**

という**極端に抽象化された人工化学モデル**である点です。

「石鹸っぽさ」を論理として書いたとき、それだけで構造は出るのか？  
という問いに対する実験コードです。

---

## モデルの特徴

### ✔ 分子種

| 種類 | 説明 | 自由度 |
|------|------|--------|
Soap | 両親媒性分子 | 位置 + 向き（8方向）  
Water | 水 | 位置のみ  
Air | 空気 | 位置のみ  

各格子点に必ず1分子が存在します（密充填モデル）。

---

### ✔ 石鹸分子の設計

石鹸は 0〜7 の8方向を向いており、

- 水を向く  
- 空気を向く  
- 石鹸同士で曲率を作る  
- 接線方向に揃う  

といった**配置の意味**に応じてエネルギーが与えられます。

---

### ✔ 相互作用はすべてローカル

エネルギーはすべて

- 3×3 近傍  
- 石鹸の向き  
- 隣接方向

だけから決まります。

距離ポテンシャル・連続角度・ベクトル力場は一切使っていません。

---

## シミュレーション手法

- 2次元周期境界格子
- 自由度：
  - 分子の位置交換（swap）
  - 石鹸の向き回転（±45°）
- 更新：
  - 局所7×7領域に対する Metropolis MCMC
- 温度パラメータあり

```
(1) 中心セルと近傍を交換
(2) 中心が石鹸なら向きを回転
(3) エネルギー差で採択
```

---

## コード構成

```
.
├── impl/
│   └── molecule.py   # 分子モデル・相互作用・エネルギー・MCMC
│   └── tank.py   # 全体格子・初期化・時間発展・ログ出力
└── log/              # 出力（.npy）
```

---

## 主要コンポーネント

### molecule.py

- 分子定義（Soap / Water / Air）
- 向き依存相互作用の意味論
- 局所エネルギー計算
- MCMCユーティリティ

設計上の中核です。  
**「どんな配置を石鹸らしいと見なすか」**はすべてここに書かれています。

---

### tank.py

- 格子生成
- 乱数初期化
- MCMCループ
- 周期境界処理
- ログ保存

純粋に「系を回す」ための管理層です。

---

## 使い方（例）

```python
from tank import Tank

tank = Tank(
    soap_ratio=0.3,
    water_ratio=0.35,
    temp_scale=0.1,
    tank_size=120,
    seed=0
)

tank.run(
    loop_num=1001,
    out_prefix="exe",
    save_step_num=100
)
```

出力は `log/exe_step_xxxxx.npy` として保存されます。

形式：

```
[x, y] = [molecule_kind, soap_direction]
```

---

## このプロジェクトでアピールしている点

- ✔ 物理・化学系モデリングの設計力  
- ✔ 局所ルールから秩序を作る設計  
- ✔ MCMC・エネルギーベースモデルの実装力  
- ✔ 数値シミュレーションの構造設計  
- ✔ 抽象モデル構築力（力を使わない化学）  

---

## 応用・発展アイデア

- ミセル率・曲率分布の定量化
- 相図探索（温度・比率）
- 構造クラスタリング
- 3D拡張
- 連続角度モデル化
- 機械学習による有効相互作用抽出

---

## 作者

Koide Hiroki  
Ph.D. in Biophysics / Numerical Simulation  
Python / Rust / C++

---

## ライセンス

MIT