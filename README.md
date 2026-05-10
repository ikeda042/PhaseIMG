
## 概要

本リポジトリは、Nikon ND2 形式のタイムラプス顕微鏡画像から TIFF フレームを抽出し、Canny 系の輪郭抽出条件を探索しながら、細胞の輪郭面積および細胞長を定量するための解析パイプラインである。主な処理は以下の 3 段階からなる。

1. ND2 タイムラプス画像の TIFF 化およびスケールバー付与
2. Canny threshold パラメータの探索と可視化
3. 輪郭に基づく細胞長推定

`main.py` は、`utils` にまとめた処理を呼び出す入口として機能する。


## Canny パラメータ探索結果

`result.gif` は、Canny threshold を変化させたときの輪郭抽出結果、輪郭面積の推移、元画像上の輪郭 overlay、および注目領域の拡大表示をまとめたアニメーションである。GIF の開始フレームは `canny param = 50` としている。

![Canny parameter search animation](docs/canny_param_search.gif)

各フレームは 2x2 パネルで構成される。

- 左上: 抽出された輪郭
- 右上: Canny threshold に対する総輪郭面積
- 左下: 元画像への輪郭 overlay
- 右下: 細胞が比較的多い領域の拡大表示

面積グラフの縦軸は、全 threshold における最大面積を基準として固定している。これにより、フレーム間で面積変化を視覚的に比較しやすくしている。

## 入力データと抽出方法

ND2 ファイルは `nd2.ND2File` により読み込む。多次元タイムラプスデータのうち、位置軸 `P` が `position_index = 0` であるフレームのみを抽出対象とする。

各フレーム $I_n$ は、8-bit 画像へ変換される。入力配列を $X$、整数型の最大値を $M$ とすると、整数型画像では次の正規化を行う。

$$
I_n = \left\lfloor \frac{X_n}{M} \cdot 255 \right\rfloor
$$

浮動小数または既に 8-bit である画像については、$[0, 255]$ へ clipping したうえで `uint8` として保存する。

スケールバーは、画素サイズ

$$
s = 0.108\ \mu\mathrm{m/pixel}
$$

に基づいて描画する。長さ $L = 10\ \mu\mathrm{m}$ のスケールバーに対応するピクセル幅は、

$$
w_{\mathrm{bar}} = \frac{L}{s}
$$

であり、本解析では約 $92.6$ pixel である。TIFF 化後の画像は `nd2totiff_processed/` に保存され、動画確認用に `timelapse_5fps.avi` も生成される。

## 輪郭抽出アルゴリズム

輪郭抽出は `utils/search_canny_param_component.py` の `SearchCannyParamComponent` が担当する。入力 RGB/BGR 画像をグレースケール化した後、Canny threshold 値 $t$ により二値化する。

$$
B_t(x, y) =
\begin{cases}
255 & \text{if } G(x, y) \geq t \\
0 & \text{otherwise}
\end{cases}
$$

ここで $G(x, y)$ はグレースケール画像である。二値化後、画像右下に描画された `10 um` スケールバーと文字領域は解析対象外として黒塗りする。これにより、スケールバーや注釈文字が細胞輪郭として誤検出されることを防ぐ。

その後、OpenCV の Canny edge detector を適用する。

$$
E_t = \mathrm{Canny}(B_t; 0, 150)
$$

輪郭は `cv2.findContours(..., cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)` により抽出し、面積が 3 pixel 未満の微小輪郭を除外する。

$$
\mathcal{C}_t =
\{ C_i \mid \mathrm{Area}(C_i) \geq 3 \}
$$

各 threshold に対して総輪郭面積を

$$
A(t) = \sum_{C_i \in \mathcal{C}_t} \mathrm{Area}(C_i)
$$

として算出する。探索範囲は

$$
t \in \{1, 2, \ldots, 254\}
$$

である。

## Canny 探索 GIF の構成

`SearchCannyParamComponent.search_canny_param()` は、各 threshold について以下の画像を生成する。

- `contour_{t}.png`: 輪郭のみの描画
- `contour_sum_{t}.png`: $A(t)$ の時系列プロット
- `contour_overlay_{t}.png`: 元画像への輪郭重畳
- `combined_{t}.png`: 2x2 パネル化した統合画像

GIF への追加順序は、現在の観察に合わせて次のように設定している。

$$
50, 51, \ldots, 254, 1, 2, \ldots, 49
$$

これにより `result.gif` のスタート地点は `canny param = 50` となる。GIF 書き出しでは `.result.tmp.gif` に一度出力し、成功後に `result.gif` へ置換する。これは途中停止時に既存の `result.gif` が壊れることを避けるためである。

## 細胞長推定アルゴリズム

細胞長推定は `utils/celllength.py` の `CellLengthCalculator` および `CellLengthAnalysisComponent` が担当する。解析対象の輪郭を

$$
C = \{(x_i, y_i)\}_{i=1}^{N}
$$

とする。

### 1. 輪郭内部画素のラスタライズ

まず輪郭を局所 bounding box 内に移動し、`cv2.fillPoly` により細胞内部領域をラスタライズする。内部画素集合を

$$
P = \{(x_j, y_j)\}_{j=1}^{M}
$$

とする。

### 2. PCA による主軸推定

内部画素の座標から共分散行列を計算する。

$$
\Sigma =
\begin{pmatrix}
\mathrm{Var}(x) & \mathrm{Cov}(x,y) \\
\mathrm{Cov}(y,x) & \mathrm{Var}(y)
\end{pmatrix}
$$

固有値分解

$$
\Sigma q_k = \lambda_k q_k
$$

を行い、最大固有値に対応する固有ベクトルを細胞の長軸方向とみなす。補助的な PCA 長は、中心化輪郭点 $\tilde{p}_i$ の長軸方向射影を用いて

$$
L_{\mathrm{PCA}} =
\max_i(\tilde{p}_i \cdot q_1)
-
\min_i(\tilde{p}_i \cdot q_1)
$$

として出力する。

### 3. PCA 座標系への変換

内部画素と輪郭点を、PCA によって得られた局所座標系 $(u_1, u_2)$ に変換する。ここで $u_1$ を細胞長軸方向、$u_2$ を短軸方向とみなす。

### 4. 中心線の多項式近似

内部画素集合に対し、短軸方向座標を長軸方向座標の関数として近似する。

$$
u_2 = f(u_1)
$$

本実装では最大 4 次の多項式

$$
f(u_1) = a_0 u_1^d + a_1 u_1^{d-1} + \cdots + a_d
$$

を最小二乗法で推定する。Vandermonde 行列を $W$、観測ベクトルを $y$ とすると、係数ベクトル $a$ は

$$
a = (W^\mathsf{T} W)^{-1} W^\mathsf{T} y
$$

で求める。正則でない場合は疑似逆行列を用いる。

### 5. 中心線長の積分

推定された中心線 $f$ に対し、輪郭端点を中心線へ射影し、積分範囲 $[u_{\min}, u_{\max}]$ を決定する。細胞長は中心線の弧長として

$$
L_{\mathrm{arc}} =
\int_{u_{\min}}^{u_{\max}}
\sqrt{1 + \left(\frac{df}{du}\right)^2}\,du
$$

で定義する。実装ではこの積分を離散的な台形則で近似している。

## 出力

細胞長解析の出力先は `celllength_results/` である。デフォルト設定では `nd2totiff_processed/0.tif` に対して `canny=85` を適用し、以下を生成する。

- `celllength_canny85_frame0.csv`
- `celllength_canny85_frame0.json`
- `celllength_canny85_frame0_overlay.png`

CSV/JSON には、各輪郭について以下の値を記録する。

- `area_px2`: 輪郭面積
- `perimeter_px`: 輪郭周長
- `bbox_x`, `bbox_y`, `bbox_w`, `bbox_h`: bounding box
- `poly_arc_length_px`: 多項式中心線に基づく弧長
- `pca_length_px`: PCA 主軸方向の射影長

## 実行方法

依存関係をインストールする。

```bash
pip install -r requirements.txt
```

ND2 抽出と細胞長解析を実行する。

```bash
python main.py
```

Canny threshold 探索のみを実行する場合は、互換ラッパーから次を実行できる。

```bash
python search_canny_param_component.py
```

## 実装構成

```text
main.py
utils/
  __init__.py
  timelapse.py
  search_canny_param_component.py
  celllength.py
celllength.py
search_canny_param_component.py
```

`celllength.py` と `search_canny_param_component.py` は後方互換性のための薄いラッパーであり、実装本体は `utils/` 配下に集約している。
