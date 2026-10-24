# MIMIC-CXR / MedCLIP 实验工程

六个正式入口严格命名为 `prepare_data.py`、`extract_features.py`、`train_metric.py`、`mine_negatives.py`、`evaluate_retrieval.py`、`analyze_geometry.py`。共用 `configs/default.json`：固定数据划分 seed=42，训练 seeds=[42,43,44]，方法 R0/A/B/C/D。所有新产物统一写入 `outputs/Step0/`；原始图片保留在 `subset/images/`，不会重复复制。

## 目录导航

| 位置 | 内容 |
|---|---|
| 根目录六个正式入口 | 数据准备、特征提取、训练、负例挖掘、检索评估、几何分析 |
| mimic_pipeline/ | 六入口共用的实际算法实现 |
| configs/default.json | 数据路径、种子、方法和训练参数 |
| scripts/ | 下载和检查工具，原 Step0/ 中的三个脚本已迁入此处 |
| tests/ | 关键算法与边界测试 |
| docs/handoff.md | 唯一交接说明：架构、数据接口、已有结果及后续待办 |
| outputs/Step0/ | 集中生成产物，详见其中 README.md |
| raw/、subset/ | 原始报告/元数据与固定图片、源索引 |
| MedCLIP/、.venv-medclip/ | 固定官方模型与本机运行环境 |
| hypotheses.md | 纳入版本管理的实验假设源文件 |
| requirements-medclip.txt | 环境依赖 |
| subset.rar、指导书 PDF | 按要求保留的交接备份与任务依据 |

先阅读 [交接说明](docs/handoff.md)，再按下面命令运行。`scripts/` 是工具源码，`outputs/Step0/checks/` 是检查结果，两者用途不同。

## CMD 运行

在项目根目录执行，例如先 `cd /d E:\MIMIC`。使用已安装的 uv 模型环境：

```cmd
".venv-medclip\Scripts\python.exe" prepare_data.py
".venv-medclip\Scripts\python.exe" extract_features.py
".venv-medclip\Scripts\python.exe" train_metric.py --smoke-test
".venv-medclip\Scripts\python.exe" train_metric.py
".venv-medclip\Scripts\python.exe" mine_negatives.py
".venv-medclip\Scripts\python.exe" evaluate_retrieval.py
".venv-medclip\Scripts\python.exe" analyze_geometry.py
```

`extract_features` 默认使用本地 Hugging Face 缓存；缺少配置或 tokenizer 时添加 `--online`。已生成且指纹匹配的特征可复用，数据/参数不匹配会停止，显式 `--force` 才重新提取。训练同样只复用配置和训练缓存匹配的完整运行。

所有入口支持 `--config`。模型入口支持 `--device cuda|cpu`。训练默认执行 A/B/C/D × 42/43/44 共 12 次，最多 20 epoch，val 双向 R@5 均值连续 3 轮不升即停止；可用 `--method B --seed 43` 指定单次。`--smoke-test` 固定使用 128 对训练数据跑 2 轮，写入单独的 `smoke_training/`，不会混入正式模型。

检索默认评估 val、R0/A/B/C/D、各训练种子及 1/5 版本提示；`--split test` 只在最终评测时使用。`mine_negatives` 默认对 B 的三个种子进行训练集批内最高 25% 挖掘；`--method D` 切换方法，`--mode budget` 额外输出 Top50 内直接 Top20 和冲突候选优先 Top20。普通 B/D 训练在每个批次动态重新挖掘，不读取静态候选清单。

几何分析固定使用 val，C/D 同时分析共享与私有分支，默认包括联合 PCA+t-SNE；快速数值排查可加 `--skip-visualization`。二维图仅辅助解释，类别着色使用第一个已知阳性标签，多病与未知的统计保留在原始 labels/mask。

## Step0 产物目录

```text
outputs/Step0/
  config_snapshot.json
  hypotheses.md
  README.md
  indices/
    mimic_5000.csv, mimic_5000_full.csv
    train.csv, val.csv, test.csv
    patient_selection.csv, selection_provenance.json
    label_names.json, train_class_statistics.csv, frequency_groups.json
    report_sections.csv
  features/
    train.npz, val.npz, test.npz
    train.json, val.json, test.json, metadata.json
    disease_prompts.npz
  text_statistics.json
  checks/
    data_audit/, medclip_check/
  logs/
    prepare_data.json, extract_features.json
  training/<method>/seed_<seed>/
    model.pt, val_features.npz, losses.csv, losses.png
    per_dimension_std.json, runtime.json
    hard_examples_20.csv  (B/D)
  negatives/<method>/seed_<seed>/<batch|budget>/
  evaluation/<val|test>/
  geometry/
```

`training/*/runtime.json` 为真实每次训练的种子、方法、耗时、峰值 CUDA 显存、参数量、早停轮次、梯度和冻结检查；不使用小样本耗时冒充正式训练记录。CPU 调试的 CUDA 显存记 null。根目录 `hypotheses.md` 是版本化的预先约定模板；`outputs/Step0/hypotheses.md` 是本次交付快照，H1-H3 均列比较组、主指标和预期方向。正式实验结论必须依据后续数值和区间填写。

### 缓存接口

每个 split 的 NPZ 至少保存 `patient_id`、`study_id`、`image_feature`、`text_feature`、`labels`、`mask`、`branch`、`tokens_before_truncation`。图文基础特征 float32、N×512、L2 单位向量；labels/mask 为 N×5；ID 为整数，顺序与 split 索引一致。

`text_prompt_features` 为 P×N×512，train 的 P=1，val/test 的 P=5。五版本为原文与指导书四个前缀，正文不改；评估时逐版本先经过对应方法文本头并归一化，再求均值并再次归一化。读取缓存时重新核对患者/检查 ID、标签、范数、文件 SHA256 和模型/索引来源。

`image_path` 沿用 `images/...`，始终相对于配置中的 `data.image_root`，不相对于集中后的索引目录。不要移动图片来匹配索引文件夹。

## 数据与训练约定

固定 5000 名患者，每人先按全部检查取最小 study_id，再要求该检查有 AP/PA，最后取最小 dicom_id。报告优先 Findings、缺失取 Impression，两者均空排除。固定名单的 SHA256、主索引 SHA256、MedCLIP 官方 commit 与权重 SHA256 在共享配置中锁定。prepare_data 从本地原始文件重建并核对指纹，不修改正式图片和原来的固定名单。

数据划分固定 train/val/test=4000/500/500。三个训练种子不重新划分。五病种顺序 Atelectasis、Cardiomegaly、Consolidation、Edema、Pleural Effusion；原始 0/1 有效，-1/空白设 mask=0。未知标签不能当作阴性，五类全未知仍可配对学习。已知阳性集合相同且非空的配对从负例中排除；同患者也不作负例。无有效负例的查询跳过并记录次数。

A/B 为两端各 Linear(512,256)，C/D 增加各自私有头与 Linear(512,512) 重建头。共享向量用于检索。双向 InfoNCE 温度 0.07；B/D 每方向在有效批内负例中选最高 25%（向上取整）权重 2，其余与正例权重 1。AdamW lr=1e-3、weight_decay=1e-4、batch=128。C/D 额外损失为 0.1×重建 MSE + 0.01×中心化共享/私有交叉协方差均方值，两模态取平均。

检索以同检查为唯一相关项，AP=正确项名次的倒数。候选库和 ID 顺序固定，相同分数按候选顺序打破平局。统计按自然类别频率及已知阳性进行分组；未知不算阴性。报告正文重复会产生语义相同但严格配对 ID 不同的候选，需要在实验分析中说明。

## 安装环境（uv、CMD）

项目统一使用 `.venv-medclip` 和 `requirements-medclip.txt`，数据准备与模型代码均已验证兼容；旧 `.venv` 已移除。已有 `.venv-medclip` 时无需重建。新机器先获取固定官方源码和合法取得的数据、权重：

```cmd
git clone --no-checkout https://github.com/RyanWangZf/MedCLIP.git MedCLIP
git -C MedCLIP checkout --detach 9c3396f20d5d54e4fae241b8cb06ca45848e98c9
uv venv --python 3.11 .venv-medclip
uv pip install --python ".venv-medclip\Scripts\python.exe" torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install --python ".venv-medclip\Scripts\python.exe" -r requirements-medclip.txt
uv pip install --python ".venv-medclip\Scripts\python.exe" --no-deps -e MedCLIP
```

将官方 ViT 权重 `pytorch_model.bin` 放入 `MedCLIP/`。官方名为 ViT 的视觉骨干实际为 Swin。代码使用官方 processor，固定 max_length padding，文本最大长度 77；全量原文截断比例为 34.42%，详见实际提取统计。首次需要 Hugging Face 配置和 tokenizer，缓存就绪后可离线。

需要本机代理时先执行：

```cmd
set HTTPS_PROXY=http://127.0.0.1:10808
set HTTP_PROXY=http://127.0.0.1:10808
```

## 核查、下载与协作

```cmd
".venv-medclip\Scripts\python.exe" -m unittest discover -s tests -v
".venv-medclip\Scripts\python.exe" scripts\audit_dataset.py
".venv-medclip\Scripts\python.exe" scripts\check_medclip.py --offline
".venv-medclip\Scripts\python.exe" scripts\check_repository.py
```

数据准备仅使用 `prepare_data.py`。`scripts/download_images.py` 为唯一下载脚本，默认读取原来的完整索引；可显式指定 `--manifest outputs/Step0/indices/mimic_5000_full.csv --output-dir subset`。认证通过 GNU Wget 终端询问密码，不写入配置，下载检查记录写入集中产物的 checks/download/。

GitHub 仅提交源码、配置和不含患者信息的说明。整个 raw、subset、outputs、本地 MedCLIP、环境、权重和压缩包均被忽略。`outputs/Step0` 中包含真实报告、患者级结果和特征，集中存放不代表可以公开上传。PhysioNet 每位使用者需各自取得权限，数据使用遵循对应协议。代码 CI 仅检查提交边界和语法，不会下载数据或运行 GPU 实验。

源码检查会同时检查工作区和 Git 暂存区。目录迁移后，暂存区仍有旧路径时会提示缺失；先用下面的 git add -A 记录迁移和删除，再检查后提交。当前工程不自动向远程仓库提交或推送。

### 提交本次代码更新（CMD）

```cmd
git add -A
".venv-medclip\Scripts\python.exe" -B scripts\check_repository.py
git status --short
git commit -m "Implement six configured entries and consolidated Step0 artifacts"
git push
```

确认检查为 PASS 后再提交；运行结果目录继续保留在本地。

清理说明：保留用于交接的 `subset.rar`；`subset/` 中的源索引是下载和独立审计的输入，`outputs/Step0/indices/` 是交付快照。根目录 hypotheses.md 是版本化源文件，集中目录内为交付副本。已清除旧小样本训练产物、重复汇总日志和缓存；正式训练模型、曲线、耗时与显存记录、验证结果及几何结果保留。未来执行 --smoke-test 会重新生成 smoke_training/，这是正常的可再生调试输出。
