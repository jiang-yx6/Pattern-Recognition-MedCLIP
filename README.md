# MIMIC-CXR 数据与 MedCLIP 流程

当前固定子集包含 5000 名患者，每人一张 AP/PA 正位图像和一份报告，患者级划分 train/val/test=4000/500/500，种子 42。5000 张图像已通过完整解码与数据审计；冻结 MedCLIP 已在 CUDA 上通过 16 对样本检查，两种特征均为 512 维。全量特征提取与正式检索实验尚未实现。

## 代码入口

| 文件 | 用途 |
| --- | --- |
| `prepare_mimic_subset.py` | 从原始 metadata、CheXpert 和报告按固定名单重建索引 |
| `download_images.py` | 唯一下载入口，使用 GNU Wget，完整解码并跳过已有完好图片 |
| `Step0/audit_dataset.py` | 全量核查图像、报告、标签、划分及严格抽样规则 |
| `Step0/check_medclip.py` | 权重检查、16 对特征验证及 5 对图文 HTML 核对页 |
| `Step0/check_repository.py` | 仅用标准库检查 Git 提交范围、代码语法、JSON、常见凭据及误提交数据 |

旧补下载与患者替换代码已移出代码入口；重复查看脚本、模型压缩包、历史归档及其中的旧日志已删除，过时说明已合并。安装、操作和后续规划统一见本 README。按用户要求，保留 `mimic-cxr-reports.zip`、`raw/cxr-record-list.csv.gz` 和 `raw/cxr-study-list.csv.gz`。

## 保留的文件

| 路径 | 用途 |
| --- | --- |
| `raw/mimic-cxr-2.0.0-metadata.csv.gz` | 原始患者、检查、图像、体位与尺寸；最小 study_id 规则的核查依据 |
| `raw/mimic-cxr-2.0.0-chexpert.csv.gz` | 原始弱标签与 mask 核查依据 |
| `raw/reports/files/` | 解压后的原始报告，供重建和来源核查 |
| `subset/images/` | 正式 5000 张图片 |
| `subset/mimic_5000.csv` | 主索引：patient_id、study_id、image_path、report、split、labels、mask |
| `subset/mimic_5000_full.csv` | 带原始标签、dicom_id、体位和下载 URL 的完整索引 |
| `subset/patient_selection.csv`、`selection_provenance.json` | 固定名单、顺序和 SHA256，复现当前子集所需 |
| `subset/label_names.json`、`train_class_statistics.csv`、`frequency_groups.json` | 标签顺序、训练频数及高低频分组 |
| `MedCLIP/` | 官方源码、本地 `pytorch_model.bin` 及可编辑安装信息 |
| `configs/default.json` | 数据路径、模型、seed、文本长度和 batch size 等配置 |
| `outputs/data_audit/` | 最新审计总结果、逐图解码/尺寸/哈希、原始报告来源记录 |
| `outputs/model_info.json` | 最近一次权重/模型检查状态、版本及哈希 |
| `outputs/medclip_check/` | 16 对特征、相似度、样本索引、截断记录、运行元数据及 5 对核对页和预览图 |
| `outputs/text_statistics.json` | 全量 5000 份报告在 77 token 限制下的截断统计 |
| `hypotheses.md` | 任务书要求的 H1—H3 实验假设，结论仍待实验 |

审计只在发现异常时生成错误清单。`outputs/medclip_check/metadata.json` 对应该次 16 对运行；`outputs/model_info.json` 是最近状态入口，后续运行可能更新。

## 数据规则

- 先确定每名患者全部检查中最小的 study_id，再要求该检查有 AP/PA 图像，最后按 dicom_id 取一张；不回退到其他检查。
- 报告优先 Findings，缺失时取 Impression，两者均空则排除。图片相对路径以 `subset/` 为根目录。
- 五病种顺序：Atelectasis、Cardiomegaly、Consolidation、Edema、Pleural Effusion。原标签 0/1 有效，-1/空白设 mask=0；不能将 mask=0 当作已知阴性。
- 高频为 Atelectasis、Cardiomegaly，低频为 Edema、Consolidation，中间为 Pleural Effusion；频数只使用训练集。
- 保留原报告中的脱敏占位符。重复报告及弱标签限制需写入实验说明；配对按检查 ID 确定。

## 环境：uv + CMD

Python 3.11。保留 `.venv` 数据准备环境与 `.venv-medclip` 模型环境，避免旧 MedCLIP 与新版数据依赖冲突。当前两个环境已经存在，无需重建。GNU Wget 需加入 PATH。

以下命令用于新机器重建，不需要激活环境：

所有命令均在项目根目录执行，例如先执行 `cd /d E:\MIMIC`；队友使用自己的克隆路径。新机器先获取固定版本官方源码，已有本地 MedCLIP 时跳过 clone，并用 `git -C MedCLIP rev-parse HEAD` 核对版本：

```cmd
git clone --no-checkout https://github.com/RyanWangZf/MedCLIP.git MedCLIP
git -C MedCLIP checkout --detach 9c3396f20d5d54e4fae241b8cb06ca45848e98c9
```

从项目使用的官方 ViT 权重包获取 `pytorch_model.bin` 并放在 `MedCLIP/`。配置记录权重 SHA256、官方代码 commit、正式主索引和固定名单 SHA256；模型检查会拒绝不匹配的权重、代码或主索引。仅克隆代码不会自动获得权重或受限数据。

```cmd
uv venv --python 3.11 ".venv"
uv pip install --python ".venv\Scripts\python.exe" -r "requirements-data.txt"

uv venv --python 3.11 ".venv-medclip"
uv pip install --python ".venv-medclip\Scripts\python.exe" torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install --python ".venv-medclip\Scripts\python.exe" -r "requirements-medclip.txt"
uv pip install --python ".venv-medclip\Scripts\python.exe" --no-deps -e "MedCLIP"
```

需要本机代理时，先在 CMD 会话设置：

```cmd
set HTTPS_PROXY=http://127.0.0.1:10808
set HTTP_PROXY=http://127.0.0.1:10808
```

已验证模型环境为 torch 2.5.1+cu121、torchvision 0.20.1、transformers 4.24.0，显卡 RTX 3060 Laptop。其余依赖见对应 requirements 文件。

## 下载、索引重建与审计

当前图片已下载完整。需要重新下载时使用：

```cmd
".venv\Scripts\python.exe" "download_images.py" --username YOUR_PHYSIONET_USERNAME --proxy http://127.0.0.1:10808
```

直接联网时省略 `--proxy ...`。Wget 在需要下载时询问密码，不将密码写入命令行或配置。完好图片会跳过；无法解码的旧图片先移入 `outputs/download/invalid_images/`。下载状态保存于 `outputs/download/`，不更改正式索引。

检查下载状态、按固定名单重建索引、执行全量审计：

```cmd
".venv\Scripts\python.exe" "download_images.py" --check-only
".venv\Scripts\python.exe" "prepare_mimic_subset.py"
".venv\Scripts\python.exe" "Step0\audit_dataset.py"
```

重建索引时保留 `subset/patient_selection.csv`，才能复现当前 5000 人；没有该名单时会重新抽样。审计不修改正式数据。

## MedCLIP 小样本检查

```cmd
".venv-medclip\Scripts\python.exe" "Step0\check_medclip.py" --device cuda --batch-size 4
start "" "outputs\medclip_check\pairs_5.html"
```

脚本从 train 固定抽取 16 对，使用官方 MedCLIPProcessor，严格加载本地完整权重，冻结骨干并设置 eval。以架构配置构建官方模型后加载 checkpoint，避免重复下载基础模型权重。首次仍需获取 Hugging Face 配置和 tokenizer；缓存就绪后可加 `--offline`。

文本使用最大长度 77、固定长度 padding。16 对有 8 份报告截断；全量检查为 1721/5000，即 34.42%，详见本地 `outputs/text_statistics.json`。HTML 展示完整报告，模型输入按 77 token 截断。程序检查维度、有限值、单位范数与配对索引，不要求全部 Top1 正确，也不会自动确认临床语义人工复核完成。

## GitHub 与多人开发

GitHub 只管理源码、配置、依赖、假设、汇总检查说明和工作流。`raw/`、整个 `subset/`、整个 `outputs/`、本地模型仓库/权重、环境、日志和课堂 PDF 均已忽略。索引含报告正文，HTML 含真实胸片和报告，特征及逐患者结果也不进入代码仓库。不要用网页拖拽或 ZIP 上传整个项目目录，因为这些方式不使用本地 Git 的忽略规则。

PhysioNet 数据使用协议要求不得共享受限数据访问权；每位参与数据处理的成员应取得相应资质和数据权限。数据和固定名单的交接应遵循对应协议与机构流程，不能因 GitHub 仓库设为 private 就直接上传。协议：https://physionet.org/content/mimic-cxr-jpg/view-dua/2.1.0/ 。

所有成员必须使用同一份固定名单与索引；不要在缺少 `patient_selection.csv` 时各自重新抽样，然后混用缓存或指标。恢复当前子集至少需要合规获取的原始 metadata、CheXpert、报告及固定名单。新克隆的代码支持源码检查，但数据审计和 GPU 检查还需要本地数据、权重及 Hugging Face 资源。

首次上传可在 CMD 执行下列命令。将远程地址替换为你自己的仓库；当前根目录尚未初始化 Git：

```cmd
git init -b main
git add .gitignore .gitattributes README.md hypotheses.md configs Step0 download_images.py prepare_mimic_subset.py requirements-data.txt requirements-medclip.txt .github docs
git status --short
".venv\Scripts\python.exe" -B Step0\check_repository.py
git commit -m "Initialize data preparation and MedCLIP checks"
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
git push -u origin main
```

源码检查只依赖 Python 标准库和 Git，新克隆后也可执行 `python -B Step0/check_repository.py`。它检查当前已跟踪与未忽略文件，并扫描工作区和暂存区内容，不审计既往 Git 历史，也不是完整的秘密扫描器。提交前检查必须为 PASS；不要使用 `git add -f` 绕过忽略规则。已跟踪的受限文件不会因新增 `.gitignore` 自动移除。

`.github/workflows/source-check.yml` 在 push/PR 后运行同一源码检查，不下载医疗数据、不使用 GPU。多人开发使用各自功能分支和 PR；修改数据规则、固定名单、配置或特征接口时先统一约定，不在各分支悄悄更换实验数据。详细检查结论见 `docs/readiness_check.md`。

建议全量基础缓存接口为每个 split 一个 NPZ，键固定为 `patient_id`、`study_id`、`image_feature`、`text_feature`、`labels`、`mask`；特征 float32，形状 N×512，L2 归一化；标签和 mask 为 N×5，ID 为整数。另存元数据，包含 split、ID 顺序、模型与索引哈希、processor 参数、文本长度和截断率。此接口为后续实现约定，当前尚无全量缓存文件。

## 下一阶段与最终交付

1. 实现全量特征提取入口，按 train/val/test 保存 4000/500/500 条图文特征，两种均为 512 维；保存 patient_id、study_id、labels、mask，核查顺序、数值、归一化与患者不重叠。
2. 使用与小样本检查一致的 processor、冻结骨干和文本参数，记录全量 token 长度/截断率、dtype、模型/代码版本、权重/索引/配置哈希、耗时与显存。
3. 先在固定 val 候选库跑通 R0 双向 R@1/5/10、mAP，再接 A—D 度量头及仅训练集的难负样本挖掘。模型选择、开发和早停使用 val，test 用于最终评测。
4. 完成共享/私有对照、五版本提示、固定 val 的几何分析、训练种子 42/43/44，以及逐查询结果、患者配对 bootstrap、案例和结论。提示按任务书处理，不改报告正文。
5. 按任务书交付可运行源码、配置、索引、特征缓存、训练模型、必要实验日志及结果，并完成综述/实验报告和 15 分钟 PPT。正式实验日志需要保留。

`.gitignore` 排除本地环境、原始数据、图片、权重和下载状态，便于管理源码；它不代表可以省略最终交付要求，也不代表可以公开分发受限数据。
