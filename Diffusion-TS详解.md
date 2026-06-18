# Diffusion-TS 详解：可解释的扩散模型用于通用时间序列生成

> 论文：**Diffusion-TS: Interpretable Diffusion for General Time Series Generation**（Xinyu Yuan, Yan Qiao, ICLR 2024）
> 论文链接：<https://openreview.net/forum?id=4h1apFjO99>
> 本文结合**原始论文**与**本仓库源码**逐步讲解，公式均与代码一一对应，便于边读边查。

---

## 目录
1. [一句话概览：它要解决什么问题](#1-一句话概览它要解决什么问题)
2. [预备知识：扩散模型（DDPM）直觉](#2-预备知识扩散模型ddpm直觉)
3. [核心创新一：直接重建 x₀，而不是预测噪声](#3-核心创新一直接重建-x而不是预测噪声)
4. [核心创新二：可解释的解码器（趋势 + 季节分解）](#4-核心创新二可解释的解码器趋势--季节分解)
5. [核心创新三：傅里叶域损失与训练目标](#5-核心创新三傅里叶域损失与训练目标)
6. [采样：从噪声到时间序列](#6-采样从噪声到时间序列)
7. [条件生成：预测、插补、多分类（无需改模型）](#7-条件生成预测插补多分类无需改模型)
8. [实验结果](#8-实验结果)
9. [如何上手运行](#9-如何上手运行)
10. [小结](#10-小结)

---

## 1. 一句话概览：它要解决什么问题

**一句话**：Diffusion-TS 是一个基于扩散模型的框架，能够生成**高质量、可解释**的多变量时间序列，并且**同一个模型**既能做无条件生成，也能做预测（forecasting）和插补（imputation）。

### 时间序列生成难在哪？

生成图像时，我们只关心"看起来真不真"。但生成时间序列时，我们往往还希望它**可解释**——能看出哪部分是长期**趋势**（trend）、哪部分是周期性**季节波动**（seasonality）、哪部分是随机**残差**（residual）。传统扩散模型像个"黑盒"，直接吐出一串数字，无法告诉你这些语义结构。

Diffusion-TS 的核心思路是：**把"可解释的时间序列分解"直接嵌入扩散模型的去噪网络中**。去噪网络不再盲目地输出一段序列，而是显式地输出"趋势项 + 季节项 + 残差项"，然后相加得到结果。这样既保证了真实度，又自带可解释性。

### 整体架构

<p align="center">
  <img src="figures/fig1.jpg" alt="Diffusion-TS 整体架构">
  <br>
  <b>图 1</b>：Diffusion-TS 整体架构。一个序列编码器（Encoder）+ 一个可解释解码器（Decoder）。解码器把序列分解为季节部分（Fourier 三角表示）和趋势部分（多项式回归 + 各层输出均值）。
</p>

它由两大部分组成（对应源码 `Models/interpretable_diffusion/transformer.py` 中的 `Transformer` 类）：

- **序列编码器（Encoder）**：从带噪输入中挖掘细粒度的时序信息，作为条件（condition）。
- **可解释解码器（Decoder）**：把表示解耦（disentangle）为**趋势**和**季节**两部分，分别用多项式回归和傅里叶级数建模。

三个关键创新点（本文第 3、4、5 节逐一展开）：

| 创新点 | 做了什么 | 好处 |
|---|---|---|
| ① 直接重建 $x_0$ | 去噪网络输出干净样本 $\hat x_0$，而非噪声 $\epsilon$ | 可在数据域上做可解释分解 |
| ② 可解释解码器 | 显式分解为趋势/季节/残差 | 语义清晰、可解释 |
| ③ 傅里叶域损失 | 在频域额外约束重建 | 抓住周期性，提升真实度 |

---

## 2. 预备知识：扩散模型（DDPM）直觉

如果你已熟悉 DDPM，可跳到第 3 节。这里用最直观的方式快速建立基础。

扩散模型的想法非常朴素：

- **前向过程（加噪）**：往一段真实数据里一点点加高斯噪声，加 $T$ 步后变成纯噪声。这一步**不需要学习**。
- **反向过程（去噪）**：训练一个神经网络，学会把噪声一点点"擦掉"，从纯噪声还原出数据。生成时，就从随机噪声出发、反复去噪即可。

记一条真实时间序列为 $x_0$，加噪 $t$ 步后为 $x_t$。

### 2.1 前向加噪过程

单步加噪定义为：

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(x_t;\ \sqrt{1-\beta_t}\,x_{t-1},\ \beta_t \mathbf{I}\right)
$$

其中 $\beta_t \in (0,1)$ 是第 $t$ 步的"噪声强度"，由**噪声调度（beta schedule）**给出。本仓库提供两种调度（`gaussian_diffusion.py:15-32`）：

- **linear**：$\beta_t$ 线性增长；
- **cosine**（默认）：$\bar\alpha_t = \dfrac{\cos^2\!\big(\frac{t/T+s}{1+s}\cdot\frac{\pi}{2}\big)}{\cos^2\!\big(\frac{s}{1+s}\cdot\frac{\pi}{2}\big)}$，噪声在两端变化更平缓，对生成质量更友好。

令 $\alpha_t = 1-\beta_t$，$\bar\alpha_t = \prod_{i=1}^{t}\alpha_i$。一个非常关键的性质是：**可以一步到位**地从 $x_0$ 直接采样出任意 $x_t$，无需逐步迭代：

$$
\boxed{\ q(x_t \mid x_0) = \mathcal{N}\!\left(x_t;\ \sqrt{\bar\alpha_t}\,x_0,\ (1-\bar\alpha_t)\mathbf{I}\right)\ }
$$

写成可微的"重参数化"形式（$\epsilon \sim \mathcal N(0,\mathbf I)$）：

$$
x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon
$$

这正是源码中的 `q_sample`：

```python
# gaussian_diffusion.py:240-245
def q_sample(self, x_start, t, noise=None):
    noise = default(noise, lambda: torch.randn_like(x_start))
    return (
        extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +          # √ᾱ_t · x_0
        extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise     # √(1-ᾱ_t) · ε
    )
```

其中 `sqrt_alphas_cumprod` $=\sqrt{\bar\alpha_t}$、`sqrt_one_minus_alphas_cumprod` $=\sqrt{1-\bar\alpha_t}$ 都是初始化时预先算好的缓冲区（`gaussian_diffusion.py:102-103`）。

### 2.2 反向去噪过程

反向过程也是一系列高斯分布，由网络参数 $\theta$ 决定：

$$
p_\theta(x_{t-1}\mid x_t) = \mathcal{N}\!\left(x_{t-1};\ \mu_\theta(x_t,t),\ \Sigma_t\right)
$$

理论基石是：**给定 $x_0$ 时，真实后验 $q(x_{t-1}\mid x_t, x_0)$ 有闭式解**，是一个高斯分布：

$$
q(x_{t-1}\mid x_t, x_0) = \mathcal{N}\!\left(x_{t-1};\ \tilde\mu_t(x_t,x_0),\ \tilde\beta_t \mathbf I\right)
$$

$$
\tilde\mu_t(x_t,x_0) = \underbrace{\frac{\beta_t\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_t}}_{\text{coef1}}\,x_0 + \underbrace{\frac{(1-\bar\alpha_{t-1})\sqrt{\alpha_t}}{1-\bar\alpha_t}}_{\text{coef2}}\,x_t,
\qquad
\tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t
$$

这三项系数正是源码里预先注册的缓冲区（`gaussian_diffusion.py:110-120`），对应函数 `q_posterior`：

```python
# gaussian_diffusion.py:138-145
def q_posterior(self, x_start, x_t, t):
    posterior_mean = (
        extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +   # coef1 · x_0
        extract(self.posterior_mean_coef2, t, x_t.shape) * x_t          # coef2 · x_t
    )
    posterior_variance = extract(self.posterior_variance, t, x_t.shape) # = β̃_t
    ...
```

**关键洞察**：从上面这个式子可以看出，只要我们能在每一步**估计出 $x_0$**，就能算出反向分布的均值，从而完成一步去噪。这就引出了 Diffusion-TS 的第一个核心创新。

---

## 3. 核心创新一：直接重建 x₀，而不是预测噪声

### 3.1 两种参数化方式

在去噪的每一步，网络需要"猜"出一些东西，帮助我们恢复 $x_{t-1}$。有两种常见选择：

- **预测噪声 $\epsilon$（传统 DDPM 做法）**：网络输出 $\epsilon_\theta(x_t,t)$，再反推 $x_0$。
- **直接预测干净样本 $x_0$（Diffusion-TS 做法）**：网络直接输出 $\hat x_0 = f_\theta(x_t, t)$。

两者数学上可互相转换。由 $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$ 可解出：

$$
\hat x_0 = \frac{1}{\sqrt{\bar\alpha_t}}x_t - \sqrt{\frac{1}{\bar\alpha_t}-1}\;\epsilon,
\qquad
\hat\epsilon = \frac{\frac{1}{\sqrt{\bar\alpha_t}}x_t - \hat x_0}{\sqrt{\frac{1}{\bar\alpha_t}-1}}
$$

源码中这两个互转就是 `predict_start_from_noise` 和 `predict_noise_from_start`：

```python
# gaussian_diffusion.py:126-136
def predict_noise_from_start(self, x_t, t, x0):                # 由 x̂_0 反推 ε̂
    return (
        (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
        extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
    )

def predict_start_from_noise(self, x_t, t, noise):            # 由 ε 推 x̂_0
    return (
        extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
        extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
    )
```

其中 `sqrt_recip_alphas_cumprod` $=\sqrt{1/\bar\alpha_t}$，`sqrt_recipm1_alphas_cumprod` $=\sqrt{1/\bar\alpha_t-1}$。

### 3.2 为什么 Diffusion-TS 选择直接重建 x₀？

因为 Diffusion-TS 想要**可解释性**：它要把输出分解成趋势、季节、残差。而趋势和季节这些语义概念，**只有在干净的数据域上才有意义**——你没法对"噪声"做趋势分解。所以网络必须直接输出数据域里的 $\hat x_0$，然后在 $\hat x_0$ 上施加分解结构。

在源码中，网络输出的"干净样本"由 `output` 函数给出，它就是趋势项与季节项之和：

```python
# gaussian_diffusion.py:147-150
def output(self, x, t, padding_masks=None):
    trend, season = self.model(x, t, padding_masks=padding_masks)
    model_output = trend + season      # x̂_0 = 趋势 + 季节(含残差)
    return model_output
```

这个 `trend + season` 就是下一节要展开的可解释分解。

---

## 4. 核心创新二：可解释的解码器（趋势 + 季节分解）

这是 Diffusion-TS 最有特色的部分。去噪网络是一个**编码器-解码器 Transformer**（`transformer.py` 中的 `Transformer` 类），它把重建结果显式拆成：

$$
\boxed{\ \hat x_0 \;=\; \underbrace{V_{\text{tr}}}_{\text{趋势 Trend}} \;+\; \underbrace{\sum_{i} S_i}_{\text{季节 Season}} \;+\; \underbrace{R}_{\text{残差 Residual}}\ }
$$

### 4.1 数据流总览

参考 `Transformer.forward`（`transformer.py:422-438`）：

```python
def forward(self, input, t, padding_masks=None, return_res=False):
    emb = self.emb(input)                                    # 1) 输入卷积嵌入 (Conv_MLP)
    inp_enc = self.pos_enc(emb)                              #    + 位置编码
    enc_cond = self.encoder(inp_enc, t, ...)                 # 2) 编码器 → 条件表示

    inp_dec = self.pos_dec(emb)
    output, mean, trend, season = self.decoder(inp_dec, t, enc_cond, ...)   # 3) 解码器

    res = self.inverse(output)                               # 4) 残差回到数据域
    res_m = torch.mean(res, dim=1, keepdim=True)
    season_error = self.combine_s(season.transpose(1,2)).transpose(1,2) + res - res_m  # 季节 + 残差
    trend = self.combine_m(mean) + res_m + trend                                       # 趋势 + 各层均值
    return trend, season_error    # 二者相加即 x̂_0
```

可以看到，最终 `trend`（趋势）与 `season_error`（季节 + 残差）相加，正是第 3 节的 `output = trend + season`。

下面分别看趋势、季节如何建模。

### 4.2 趋势项：多项式回归 + 各层均值聚合

**直觉**：趋势是缓慢、平滑的整体走向，天然适合用**低阶多项式**来拟合（比如一条略微弯曲的曲线）。

每个解码器块（`DecoderBlock`）里有一个 `TrendBlock`（`transformer.py:12-34`）。它用一个小卷积网络回归出多项式系数 $C$，再乘以一组固定的**多项式基**：

$$
V_{\text{tr}}^{(i)} = \mathbf{C}^{(i)} \cdot \mathbf{P}, \qquad
\mathbf{P} = \big[\,p^1,\ p^2,\ p^3\,\big],\quad p = \left[\tfrac{1}{n+1}, \tfrac{2}{n+1}, \dots, \tfrac{n}{n+1}\right]
$$

这里多项式阶数 `trend_poly = 3`（即用 $p, p^2, p^3$ 三个基），$n$ 为序列长度。代码：

```python
# transformer.py:26-34
lin_space = torch.arange(1, out_dim + 1, 1) / (out_dim + 1)
self.poly_space = torch.stack([lin_space ** float(p + 1) for p in range(trend_poly)], dim=0)  # 多项式基
...
def forward(self, input):
    x = self.trend(input).transpose(1, 2)                       # 回归多项式系数
    trend_vals = torch.matmul(x.transpose(1, 2), self.poly_space.to(x.device))   # 系数 × 基
    return trend_vals.transpose(1, 2)
```

此外，每个解码器块还会提取自身输出的**均值** `m`（代表该层捕捉到的"水平偏移"），所有层的均值经一个 1×1 卷积 `combine_m` 聚合后也并入趋势：

```python
# transformer.py:331-332  (DecoderBlock.forward 末尾)
m = torch.mean(x, dim=1, keepdim=True)
return x - m, self.linear(m), trend, season       # x-m 去均值后继续传递；m 单独汇总到趋势
```

```python
# transformer.py:433
trend = self.combine_m(mean) + res_m + trend      # 趋势 = 各层均值聚合 + 残差均值 + 多项式趋势
```

> 仓库里还提供了一个基于**移动平均**的趋势分解备选 `MovingBlock`/`series_decomp`（`transformer.py:37-49`、`model_utils.py:160-190`），默认用多项式版本。

### 4.3 季节项：傅里叶级数 / 频域 top-k 外推

**直觉**：季节性是周期性波动，最自然的工具就是**傅里叶变换**——任何周期信号都能写成不同频率正余弦波的叠加。

默认季节模块是 `FourierLayer`（`transformer.py:52-97`），流程是：

1. 对序列做实数 FFT：$X_f = \mathrm{rFFT}(x)$；
2. 选出**幅值最大的 top-k 个频率**（`top_k = factor·log(length)`），其余丢弃（去噪、保留主周期）；
3. 用这些主频做**逆变换式的外推**重建：

$$
S(t) = \sum_{k \in \text{top-}K} A_k \cos\!\big(2\pi f_k\, t + \phi_k\big)
$$

其中 $A_k=|X_{f_k}|$ 是幅值、$\phi_k=\angle X_{f_k}$ 是相位、$f_k$ 是频率。代码：

```python
# transformer.py:79-97
def extrapolate(self, x_freq, f, t):
    x_freq = torch.cat([x_freq, x_freq.conj()], dim=1)   # 补共轭，保证实数输出
    f = torch.cat([f, -f], dim=1)
    amp   = rearrange(x_freq.abs(),   'b f d -> b f () d')   # A_k 幅值
    phase = rearrange(x_freq.angle(), 'b f d -> b f () d')   # φ_k 相位
    x_time = amp * torch.cos(2 * math.pi * f * t + phase)    # A_k cos(2π f_k t + φ_k)
    return reduce(x_time, 'b f t d -> b t d', 'sum')

def topk_freq(self, x_freq):
    top_k = int(self.factor * math.log(length))
    values, indices = torch.topk(x_freq.abs(), top_k, dim=1, ...)   # 取幅值最大的 K 个频率
```

> 仓库还提供另一个等价思路 `SeasonBlock`（`transformer.py:100-120`）：直接用一组固定的正余弦基 $\{\cos(2\pi p\,\tau), \sin(2\pi p\,\tau)\}$ 与回归系数相乘。默认 `DecoderBlock` 用的是 `FourierLayer`（`transformer.py:310`）。

### 4.4 时间步如何注入：AdaLayerNorm

去噪网络必须知道"现在是第几步 $t$"（噪声有多大）。Diffusion-TS 用 **AdaLayerNorm**（自适应层归一化）把 $t$ 注入每个 Transformer 块（`model_utils.py:259-274`）：

$$
\text{AdaLN}(x, t) = \text{LayerNorm}(x)\cdot\big(1+\text{scale}(t)\big) + \text{shift}(t)
$$

其中 $\text{scale}(t),\text{shift}(t)$ 由时间步的正弦嵌入经一个小 MLP 生成。多分类任务下，还可叠加类别嵌入 `label_emb`：

```python
# model_utils.py:267-274
def forward(self, x, timestep, label_emb=None):
    emb = self.emb(timestep)                              # 时间步的正弦位置嵌入
    if label_emb is not None:
        emb = emb + label_emb                             # 可选：注入类别条件
    emb = self.linear(self.silu(emb)).unsqueeze(1)
    scale, shift = torch.chunk(emb, 2, dim=2)
    x = self.layernorm(x) * (1 + scale) + shift           # 自适应缩放/平移
    return x
```

### 4.5 编码器 / 解码器结构

- **Encoder**（`transformer.py:204-268`）：标准 Transformer 块（AdaLayerNorm + 多头自注意力 + MLP），输出条件表示 `enc_cond`。
- **Decoder**（`transformer.py:271-378`）：每块包含**自注意力** + **交叉注意力**（query 来自解码器、key/value 来自编码器条件），随后分流出趋势 `TrendBlock` 与季节 `FourierLayer`，并逐块累加：

```python
# transformer.py:370-378
for block_idx in range(len(self.blocks)):
    x, residual_mean, residual_trend, residual_season = self.blocks[block_idx](...)
    season += residual_season      # 季节项跨层累加
    trend  += residual_trend       # 趋势项跨层累加
    mean.append(residual_mean)     # 各层均值收集
```

至此，可解释解码器完整地构造出了 $\hat x_0 = \text{Trend} + \text{Season} + \text{Residual}$。

---

## 5. 核心创新三：傅里叶域损失与训练目标

### 5.1 时域重建损失

既然网络直接输出 $\hat x_0$，最朴素的训练目标就是让它逼近真实 $x_0$。本仓库默认用 **L1 损失**（也可选 L2，`gaussian_diffusion.py:231-238`、配置 `loss_type: 'l1'`）：

$$
\mathcal{L}_{\text{time}} = \big\| \hat x_0 - x_0 \big\|
$$

### 5.2 傅里叶域损失（关键加项）

只在时域上对齐，模型容易忽略周期性细节。Diffusion-TS 额外在**频域**施加约束：把预测和真实序列都做 FFT，比较它们的实部和虚部：

$$
\mathcal{L}_{\text{fourier}} = \sum_{i} \Big( \big\|\,\mathrm{Re}\,\mathrm{FFT}(\hat x_0)_i - \mathrm{Re}\,\mathrm{FFT}(x_0)_i \big\| + \big\|\,\mathrm{Im}\,\mathrm{FFT}(\hat x_0)_i - \mathrm{Im}\,\mathrm{FFT}(x_0)_i \big\| \Big)
$$

```python
# gaussian_diffusion.py:257-264
fourier_loss = torch.tensor([0.])
if self.use_ff:
    fft1 = torch.fft.fft(model_out.transpose(1, 2), norm='forward')
    fft2 = torch.fft.fft(target.transpose(1, 2), norm='forward')
    fft1, fft2 = fft1.transpose(1, 2), fft2.transpose(1, 2)
    fourier_loss = self.loss_fn(torch.real(fft1), torch.real(fft2), reduction='none') \
                 + self.loss_fn(torch.imag(fft1), torch.imag(fft2), reduction='none')
    train_loss += self.ff_weight * fourier_loss
```

权重 `ff_weight` 默认取 $\sqrt{\text{seq\_length}}/5$（`gaussian_diffusion.py:63`）。

### 5.3 按时间步重加权

不同噪声水平 $t$ 的样本，对训练的贡献应该不同。Diffusion-TS 给每个样本乘上一个与 $t$ 相关的权重：

$$
w_t = \frac{\sqrt{\alpha_t}\,\sqrt{1-\bar\alpha_t}}{\beta_t}\cdot\frac{1}{100}
$$

```python
# gaussian_diffusion.py:124
register_buffer('loss_weight', torch.sqrt(alphas) * torch.sqrt(1. - alphas_cumprod) / betas / 100)
```

### 5.4 完整训练目标与流程

综合起来，第 $t$ 步、单个样本的损失为：

$$
\boxed{\ \mathcal{L}(t) = w_t \cdot \Big( \mathcal{L}_{\text{time}} \;+\; \lambda_{\text{ff}}\,\mathcal{L}_{\text{fourier}} \Big)\ }
$$

对应 `_train_loss`（`gaussian_diffusion.py:247-268`）。训练主循环非常简洁（`forward`，`gaussian_diffusion.py:270-274`）：

```python
def forward(self, x, **kwargs):
    t = torch.randint(0, self.num_timesteps, (b,), device=device).long()   # 随机采时间步
    return self._train_loss(x_start=x, t=t, **kwargs)
```

**训练伪代码**：

```
重复直到收敛：
    从数据集采一个 batch x_0
    随机采时间步 t ~ Uniform(1..T)，随机采噪声 ε ~ N(0, I)
    加噪：x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε        # q_sample
    预测：x̂_0 = trend(x_t,t) + season(x_t,t)     # 可解释解码器
    L = w_t · ( ||x̂_0 - x_0|| + λ_ff · ||FFT(x̂_0) - FFT(x_0)|| )
    反向传播，更新 θ（并用 EMA 维护一份平滑权重）
```

---

## 6. 采样：从噪声到时间序列

训练好后，生成只需从纯噪声 $x_T\sim\mathcal N(0,\mathbf I)$ 出发，反复去噪。

### 6.1 标准反向采样（DDPM）

每一步：网络预测 $\hat x_0$ → 用后验公式算出均值 → 加一点噪声 → 得到 $x_{t-1}$。

$$
x_{t-1} = \tilde\mu_t(\hat x_0, x_t) + \sigma_t z, \quad z\sim\mathcal N(0,\mathbf I),\ \sigma_t^2=\tilde\beta_t
$$

```python
# gaussian_diffusion.py:170-190
def p_sample(self, x, t, ...):
    model_mean, _, model_log_variance, x_start = self.p_mean_variance(x=x, t=batched_times, ...)
    noise = torch.randn_like(x) if t > 0 else 0.        # t=0 时不再加噪
    pred_series = model_mean + (0.5 * model_log_variance).exp() * noise
    return pred_series, x_start

def sample(self, shape):
    img = torch.randn(shape, device=device)             # 从纯噪声开始
    for t in reversed(range(0, self.num_timesteps)):    # T-1, T-2, ..., 0
        img, _ = self.p_sample(img, t)
    return img
```

注意 `p_mean_variance`（`gaussian_diffusion.py:162-168`）里会把预测的 $\hat x_0$ 裁剪到 $[-1,1]$（数据被归一化到该区间，见配置 `neg_one_to_one: True`）。

### 6.2 DDIM 加速采样

标准采样要走完全部 $T$ 步（如 500/1000 步），较慢。Diffusion-TS 支持 **DDIM** 快速采样：只在 $T$ 步中取若干个子步（`sampling_timesteps < timesteps` 时自动启用，`gaussian_diffusion.py:90`）。更新公式：

$$
x_{t_{\text{next}}} = \sqrt{\bar\alpha_{t_{\text{next}}}}\;\hat x_0 \;+\; c\,\hat\epsilon \;+\; \sigma\, z
$$

$$
\sigma = \eta\sqrt{\frac{(1-\bar\alpha_t/\bar\alpha_{t_{\text{next}}})(1-\bar\alpha_{t_{\text{next}}})}{1-\bar\alpha_t}}, \qquad c = \sqrt{1-\bar\alpha_{t_{\text{next}}}-\sigma^2}
$$

```python
# gaussian_diffusion.py:212-219
alpha      = self.alphas_cumprod[time]
alpha_next = self.alphas_cumprod[time_next]
sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
c     = (1 - alpha_next - sigma ** 2).sqrt()
noise = torch.randn_like(img)
img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise
```

超参 $\eta$（`eta`）控制随机性：$\eta=0$ 时为确定性 DDIM（最快、可复现），$\eta=1$ 时退化为 DDPM。生成入口统一为 `generate_mts`（`gaussian_diffusion.py:223-229`），会根据是否开启快速采样自动选择 `sample` 或 `fast_sample`。

---

## 7. 条件生成：预测、插补、多分类（无需改模型）

Diffusion-TS 的一大卖点：**同一个无条件训练好的模型，不用改结构、不用重训**，就能做预测和插补。

<p align="center">
  <img src="figures/fig4.jpg" alt="时间序列插补与预测可视化">
  <br>
  <b>图 2</b>：Diffusion-TS 在时间序列插补（imputation）与预测（forecasting）上的可视化效果。
</p>

### 7.1 插补 / 预测：带引导的 in-filling

**设定**：序列里有一部分是已知的（`partial_mask` 标出的"观测位置"），要补全其余部分。预测就是插补的特例（已知前面、补全后面）。

核心机制（`sample_infill` / `p_sample_infill`，`gaussian_diffusion.py:320-365`）分两件事：

1. **已知位置直接回填**：每一步都把观测值按当前噪声水平加噪后，强制覆盖到已知位置上：

   ```python
   # gaussian_diffusion.py:362-363
   target_t = self.q_sample(target, t=batched_times)   # 把观测值加噪到第 t 步
   pred_img[partial_mask] = target_t[partial_mask]      # 强制覆盖已知位置
   ```

2. **Langevin 引导（`langevin_fn`，`gaussian_diffusion.py:367-416`）**：在未知位置上，做几步基于梯度的优化，让网络重建的 $\hat x_0$ 在已知位置尽量贴合观测、同时保持与去噪均值一致。优化目标为：

   $$
   \mathcal{L}_{\text{guide}} = \underbrace{\text{coef}\cdot\frac{\|\mu - x\|^2}{\sigma}}_{\text{贴近去噪均值}} + \underbrace{\frac{\|\hat x_0[\text{obs}] - x_{\text{obs}}\|^2}{\bar\sigma}}_{\text{贴合观测}}
   $$

   迭代步数 $K$ 随噪声水平自适应（噪声越大步数越多，`gaussian_diffusion.py:380-389`），用 Adagrad 优化器更新未知位置。

### 7.2 分类器引导：多分类生成（2025 更新）

仓库新增了 **Classifier Guidance**，支持按类别生成（如不同状态的 EEG）。思路是用一个额外训练的分类器 $p(y\mid x_t)$ 的梯度，把采样轨迹"推"向目标类别 $y$：

- `cond_fn`（`model_utils.py:68-88`）计算 $\nabla_{x}\log p(y\mid x)$；
- `condition_mean`（`gaussian_diffusion.py:418-431`，Sohl-Dickstein 等 2015 策略）：

  $$
  \tilde\mu = \mu_\theta + s\cdot\Sigma_t\,\nabla_{x}\log p(y\mid x_t)
  $$

- `condition_score`（`gaussian_diffusion.py:433-450`，Song 等 2020 的 score 修正策略），用于 DDIM 路径。

其中 $s$ 是 `classifier_scale`，控制类别引导强度。这部分**完全外挂**，主模型权重无需改动。

---

## 8. 实验结果

论文在 Stocks、ETTh1、Energy、fMRI、Sines、MuJoCo 等数据集上做了大量定量与定性评测，指标包括：

- **Context-FID**：衡量生成与真实分布的距离（越低越好，`Utils/context_fid.py`）；
- **Discriminative Score**：训练一个判别器区分真假，越接近 0 越好（`Utils/discriminative_metric.py`）；
- **Predictive Score**：用生成数据训练、真实数据测试的预测误差（`Utils/predictive_metric.py`）；
- **Correlational Score**：特征间相关结构的保持程度（`Utils/cross_correlation.py`）。

<p align="center">
  <b>表 1</b>：24 长度时间序列生成结果。
  <br>
  <img src="figures/fig2.jpg" alt="标准时间序列生成结果">
</p>

<p align="center">
  <b>表 2</b>：长序列时间序列生成结果。
  <br>
  <img src="figures/fig3.jpg" alt="长序列生成结果">
</p>

结论：Diffusion-TS 在标准长度与长序列生成上均达到 **SOTA**，尤其在长序列和复杂多变量场景下相对 GAN/VAE 类与其他扩散方法优势明显。

---

## 9. 如何上手运行

### 9.1 环境

```bash
pip install -r requirements.txt
```

需要一块支持 CUDA 的 GPU。数据集放入 `./Data` 目录（见 `README.md` 的下载链接）。

### 9.2 训练

```bash
python main.py --name {name} --config_file Config/{config}.yaml --gpu 0 --train
```

### 9.3 采样

```bash
# 无条件生成
python main.py --name {name} --config_file {cfg} --gpu 0 --sample 0 --milestone {ckpt}

# 插补（缺失补全）
python main.py --name {name} --config_file {cfg} --gpu 0 --sample 1 --milestone {ckpt} \
               --mode infill --missing_ratio {ratio}

# 预测
python main.py --name {name} --config_file {cfg} --gpu 0 --sample 1 --milestone {ckpt} \
               --mode predict --pred_len {len}
```

### 9.4 关键超参（以 `Config/etth.yaml` 为例）

| 参数 | 含义 | 示例值 |
|---|---|---|
| `seq_length` | 序列长度（窗口） | 24 |
| `feature_size` | 变量数（通道数） | 7 |
| `n_layer_enc` / `n_layer_dec` | 编码器/解码器层数 | 3 / 2 |
| `d_model` | 隐藏维度（= n_heads × head_dim） | 64 |
| `timesteps` | 扩散步数 $T$ | 500 |
| `sampling_timesteps` | 采样步数（< T 则启用 DDIM 加速） | 500 |
| `loss_type` | 时域损失（l1 / l2） | l1 |
| `beta_schedule` | 噪声调度（linear / cosine） | cosine |
| `n_heads` | 注意力头数 | 4 |

> 训练无条件生成时，配置中 `proportion`（训练比例）设为 `1.0`（用全部数据）；做条件生成时需切分数据，设为 `< 1`。

更多细节与可视化、评测代码见仓库中的 `Tutorial_*.ipynb` 与 `Experiments/` 目录。

---

## 10. 小结

Diffusion-TS 用三个互相配合的设计，让扩散模型在时间序列上既"真"又"可解释"：

1. **直接重建 $x_0$** —— 输出落在数据域，才有可能做语义分解（第 3 节）。
2. **可解释解码器** —— 用多项式回归建模**趋势**、用傅里叶级数建模**季节**，显式给出 $\hat x_0=\text{Trend}+\text{Season}+\text{Residual}$（第 4 节）。
3. **傅里叶域损失 + 时间步重加权** —— 在频域额外约束，抓住周期性，提升真实度（第 5 节）。

再加上 **DDIM 加速采样** 和 **无需改模型的条件生成**（in-filling 引导 + 分类器引导），让它成为一个统一、实用、可解释的通用时间序列生成框架。

**适用场景**：合成数据增强（隐私保护）、缺失值插补、时序预测、异常检测的正常样本建模、可解释的趋势/周期分析等。

---

### 参考

- 论文：Xinyu Yuan, Yan Qiao. *Diffusion-TS: Interpretable Diffusion for General Time Series Generation*. ICLR 2024. <https://openreview.net/forum?id=4h1apFjO99>
- 核心源码：
  - 扩散过程与训练/采样：[`Models/interpretable_diffusion/gaussian_diffusion.py`](Models/interpretable_diffusion/gaussian_diffusion.py)
  - 可解释 Transformer：[`Models/interpretable_diffusion/transformer.py`](Models/interpretable_diffusion/transformer.py)
  - 基础组件（AdaLayerNorm、分解、位置编码）：[`Models/interpretable_diffusion/model_utils.py`](Models/interpretable_diffusion/model_utils.py)

```bibtex
@inproceedings{yuan2024diffusionts,
  title={Diffusion-{TS}: Interpretable Diffusion for General Time Series Generation},
  author={Xinyu Yuan and Yan Qiao},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=4h1apFjO99}
}
```
