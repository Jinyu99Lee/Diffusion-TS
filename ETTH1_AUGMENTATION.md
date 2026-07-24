# ETTh1 数据增强运行手册

以下命令均从 `Diffusion-TS` 仓库根目录运行。生成器只读取 ETTh1
的前 8640 个训练点，不读取官方 validation/test。

Sonnet 的 ETTh raw/forecast loader 改为 weather-style `target last` 后，
这里不需要修改 Diffusion-TS dataloader，也不需要重建已有输入 NPZ。原因是
两者负责不同层次的切分：Sonnet 用 target endpoint 决定 forecast 样本属于
train/val/test；Diffusion-TS 则先把纯 train 的 8640 点变成 `(N,T,7)`，再在
样本轴上无放回地切 generator train/val。Sonnet 的 target-last 改动不会改变
真实 train 窗口总数 N。

## 1. 转换成显式 train/val NPZ

```bash
python Data/etth1_to_npz.py \
  --csv /data/jinyuli/.darts/datasets/ETTh1.csv \
  --output-dir Data/datasets/etth1 \
  --lookback 336 \
  --horizons 96 192 336 720 \
  --val-sample-start-ratio 0.70 \
  --val-sample-end-ratio 0.85 \
  --stride 1 \
  --overwrite
```

转换器先对 8640 个点完整滑窗，再将样本轴上的半开区间
`[floor(0.70N), floor(0.85N))` 放入 generator-val，其余样本放入
generator-train。完整窗口不重复分配；stride-1 相邻窗口共享原始时间点是预期行为。

| H | T=336+H | 全部 N | train | val | 文件目录 |
|---:|---:|---:|---:|---:|---|
| 96 | 432 | 8209 | 6978 | 1231 | `Data/datasets/etth1/T432/` |
| 192 | 528 | 8113 | 6896 | 1217 | `Data/datasets/etth1/T528/` |
| 336 | 672 | 7969 | 6774 | 1195 | `Data/datasets/etth1/T672/` |
| 720 | 1056 | 7585 | 6447 | 1138 | `Data/datasets/etth1/T1056/` |

## 2. HPO

先给任一命令添加 `--dry-run` 做快速配置检查；确认输出的 `T`、`D=7`
和 4 个候选任务正确后，再移除 `--dry-run` 启动正式训练。每块 GPU 只运行
一个候选，下面的正式命令不会被转换器或其他脚本自动执行。四档命令使用
相同的 GPU 0/2，因此必须依次运行；若要并行启动不同 H，需改成互不重叠的
GPU 集合。

### H=96，T=432

```bash
mkdir -p Logs/hpo/etth1/H96

OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 nohup python hpo_grid_search.py \
  --base-config Config/etth1_npz.yaml \
  --train-npz Data/datasets/etth1/T432/etth1_T432_p96_train.npz \
  --val-npz Data/datasets/etth1/T432/etth1_T432_p96_val.npz \
  --d-model 64 96 \
  --base-lr 1e-5 \
  --batch-size 64 128 \
  --max-epochs 18000 --save-cycle 1800 --val-num-repeats 1 \
  --gpu-slots "0:1,2:1" \
  --output-root outputs/hpo/etth1/H96 \
  --experiment-name etth1_T432_p96 \
  > "Logs/hpo/etth1/H96/etth1_T432_p96_$(date +%F_%H-%M-%S).log" 2>&1 &
```

### H=192，T=528

```bash
mkdir -p Logs/hpo/etth1/H192

OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 nohup python hpo_grid_search.py \
  --base-config Config/etth1_npz.yaml \
  --train-npz Data/datasets/etth1/T528/etth1_T528_p192_train.npz \
  --val-npz Data/datasets/etth1/T528/etth1_T528_p192_val.npz \
  --d-model 64 96 \
  --base-lr 1e-5 \
  --batch-size 64 128 \
  --max-epochs 18000 --save-cycle 1800 --val-num-repeats 1 \
  --gpu-slots "0:1,2:1" \
  --output-root outputs/hpo/etth1/H192 \
  --experiment-name etth1_T528_p192 \
  > "Logs/hpo/etth1/H192/etth1_T528_p192_$(date +%F_%H-%M-%S).log" 2>&1 &
```

### H=336，T=672

```bash
mkdir -p Logs/hpo/etth1/H336

OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 nohup python hpo_grid_search.py \
  --base-config Config/etth1_npz.yaml \
  --train-npz Data/datasets/etth1/T672/etth1_T672_p336_train.npz \
  --val-npz Data/datasets/etth1/T672/etth1_T672_p336_val.npz \
  --d-model 64 96 \
  --base-lr 1e-5 \
  --batch-size 64 128 \
  --max-epochs 18000 --save-cycle 1800 --val-num-repeats 1 \
  --gpu-slots "0:1,2:1" \
  --output-root outputs/hpo/etth1/H336 \
  --experiment-name etth1_T672_p336 \
  > "Logs/hpo/etth1/H336/etth1_T672_p336_$(date +%F_%H-%M-%S).log" 2>&1 &
```

### H=720，T=1056

```bash
mkdir -p Logs/hpo/etth1/H720

OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 nohup python hpo_grid_search.py \
  --base-config Config/etth1_npz.yaml \
  --train-npz Data/datasets/etth1/T1056/etth1_T1056_p720_train.npz \
  --val-npz Data/datasets/etth1/T1056/etth1_T1056_p720_val.npz \
  --d-model 64 96 \
  --base-lr 1e-5 \
  --batch-size 64 128 \
  --max-epochs 18000 --save-cycle 1800 --val-num-repeats 1 \
  --gpu-slots "0:1,2:1" \
  --output-root outputs/hpo/etth1/H720 \
  --experiment-name etth1_T1056_p720 \
  > "Logs/hpo/etth1/H720/etth1_T1056_p720_$(date +%F_%H-%M-%S).log" 2>&1 &
```

不要为这些显式切分文件传 `--data-npz`、`--split-method` 或
`--valid-perc`。

## 3. 从每个最佳 checkpoint 生成 1:1 合成集

只有对应 HPO 目录已经产生 `best_run.json` 后才运行：

```bash
python rerun_best_hpo.py \
  --hpo-dir outputs/hpo/etth1/H96/etth1_T432_p96 \
  --gpu 0 --num-samples 8209 --size-every 128 \
  --output outputs/hpo/etth1/H96/etth1_T432_p96/best_synth.npz

python rerun_best_hpo.py \
  --hpo-dir outputs/hpo/etth1/H192/etth1_T528_p192 \
  --gpu 0 --num-samples 8113 --size-every 128 \
  --output outputs/hpo/etth1/H192/etth1_T528_p192/best_synth.npz

python rerun_best_hpo.py \
  --hpo-dir outputs/hpo/etth1/H336/etth1_T672_p336 \
  --gpu 0 --num-samples 7969 --size-every 128 \
  --output outputs/hpo/etth1/H336/etth1_T672_p336/best_synth.npz

python rerun_best_hpo.py \
  --hpo-dir outputs/hpo/etth1/H720/etth1_T1056_p720 \
  --gpu 0 --num-samples 7585 --size-every 128 \
  --output outputs/hpo/etth1/H720/etth1_T1056_p720/best_synth.npz
```

这里显式使用完整真实训练滑窗数 `N`，不是 generator-train 的条数。每个
`best_synth.npz` 的形状应为 `(N, 336+H, 7)`。

## 4. 按 H 接入 Sonnet

每次 Sonnet 实验只指向同一 H 的 `best_synth.npz`。例如 H=96：

```bash
cd /data/jinyuli/Projects/Sonnet

python scripts/run_experiment.py \
  model=sonnet dataset=exp_data_config/etth1 exp=etth \
  seed=2021 exp.seq_length=336 exp.pred_length=96 \
  model.model_params.revin=1 \
  exp.use_synthetic_train_data=True \
  exp.synthetic_train_csv_path=/data/jinyuli/Projects/Diffusion-TS/outputs/hpo/etth1/H96/etth1_T432_p96/best_synth.npz
```

`exp=etth` 已在 Sonnet 中启用 target-last validation/test 规则。

其余三档只成对替换 `exp.pred_length` 和路径：

| H | `exp.pred_length` | 合成 NPZ |
|---:|---:|---|
| 192 | 192 | `.../H192/etth1_T528_p192/best_synth.npz` |
| 336 | 336 | `.../H336/etth1_T672_p336/best_synth.npz` |
| 720 | 720 | `.../H720/etth1_T1056_p720/best_synth.npz` |

`T=336+H`，因此 Sonnet 会从每条合成序列构造恰好一个预测样本。其数据模块
会强制关闭 validation/test 的合成增强，只有真实 train 会追加这些样本。
