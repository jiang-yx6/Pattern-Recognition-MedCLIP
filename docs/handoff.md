# 数据处理与后续实验交接说明

交接核对日期：2026-10-24。

## 1. 当前完成范围

数据处理部分已完成，可以交付给训练、负样本研究、检索评估和几何分析负责人。

- 固定 5000 名患者，每人一项检查、一张 AP/PA 图像；先取患者全部检查中最小 study_id，再要求该检查存在 AP/PA，最后取最小 dicom_id。
- 报告按 subject_id/study_id 匹配，优先 Findings，缺失时使用 Impression，两者为空时排除。
- 患者级固定划分：train/val/test=4000/500/500，划分种子 42。
- 标签和 mask 已生成；类别频数及高低频组只依据训练集确定。
- 2026-10-21 全量数据审计 30/30 通过，记录包含全部图片解码和来源匹配检查。
- 官方 MedCLIP 骨干冻结，5000 对图文的 512 维特征已缓存；16 对模型检查及 5 对图文预览已生成。
- 六个正式入口已实现，共用 configs/default.json。
- 已运行 A/B/C/D × 42/43/44 共 12 次正式训练，以及验证集检索和几何分析；对应结果可复用和检查。
- 目录整理后，交付清单包含 481 个产物文件，并记录 19 个实现文件的当前哈希。已确认 479 个保留产物与整理前完全一致，另外两个文件是更新后的目录说明和完成记录；三个 split 的缓存 ID 顺序、标签、mask、特征维度、范数及来源检查通过。本次没有重新执行完整图片解码审计。
- 整理后九项算法测试、六个正式入口及三个辅助工具的 --help 检查均通过；新源码导出包含 28 个文件，提交范围检查无问题。当前 Git 暂存区仍是旧布局，提交前须按 README 使用 git add -A 记录迁移和删除。

以上是已有数据与工程交付状态，不代表已完成最终实验结论。

## 2. 项目目标和方法

项目研究胸片与英文报告之间的双向检索：以图片查找配对报告，以及以报告查找配对图片。使用冻结 MedCLIP 提供统一基础表征，比较难负样本、共享/私有分支、文本提示集成对检索和特征空间的影响。

| 方法 | 负样本 | 表征/度量头 |
|---|---|---|
| R0 | 无新增训练 | 冻结 MedCLIP 原始 512 维特征 |
| A | 普通有效批内负样本 | 图像/文本各自的普通 256 维度量头 |
| B | 有效批内负样本中最高 25% 相似度加权 | 普通 256 维度量头 |
| C | 普通有效批内负样本 | 共享/私有头及重建头，使用共享分支检索 |
| D | 有效批内负样本中最高 25% 相似度加权 | 共享/私有头及重建头，使用共享分支检索 |

比较组、主指标和预期方向以根目录 hypotheses.md 为准；不要将预期方向当成实验证实的结论。

## 3. 工程架构与数据流

```text
raw/ 元数据、疾病标签和原始报告 + subset/images/ 图片
                    |
              prepare_data.py
                    |
       outputs/Step0/indices/ 固定索引与分组
                    |
             extract_features.py
                    |
       outputs/Step0/features/ 冻结图文特征
                    |
              train_metric.py
                    |
       outputs/Step0/training/ 度量头模型与日志
          |                  |                 |
 mine_negatives.py  evaluate_retrieval.py  analyze_geometry.py
          |                  |                 |
     negatives/        evaluation/val/      geometry/
```

mine_negatives.py 是候选导出和额外筛选研究入口。普通 B/D 训练在每个批次动态选择难负样本，不要求提前运行该脚本，也不读取导出的静态候选文件。

当前共享输出根目录为 outputs/Step0/，其中也保存了提前运行的训练和验证结果。目录名不意味着里面只有数据准备文件。

## 4. 后续人员复用哪些代码

根目录六个文件为命令行入口，实际实现位于 mimic_pipeline/。修改对应模块即可保留统一的运行方式。

| 负责人/工作 | 入口 | 主要实现和公共接口 |
|---|---|---|
| 数据复现、筛选核查 | prepare_data.py | data.py；common.py 的路径与配置 |
| 特征提取、增加编码版本 | extract_features.py | features.py 的 Encoder；checkpoint.py 的权重与版本检查 |
| 模型与损失、训练实验 | train_metric.py | metric.py 的 MetricHead、contrastive_loss、auxiliary_losses、train_one |
| 难负样本、提示冲突筛选 | mine_negatives.py | mining.py 的 conflict_flags、mine_one；metric.py 的 valid_negatives、negative_weights |
| 检索、分组与统计比较 | evaluate_retrieval.py | retrieval.py 的 evaluate_one、group_masks、bootstrap_differences |
| 特征空间解释 | analyze_geometry.py | geometry.py 的 space_statistics、analyze_one |
| 所有开发者 | 六入口共同使用 | common.py 的 load_config、load_cache、seed_all、Runtime；metric.py 的 load_head、transform_cache |

辅助脚本统一放在 scripts/：audit_dataset.py 为全量数据审计；check_medclip.py 为小规模编码验证；check_repository.py 为提交范围检查；download_images.py 是唯一图片下载入口。tests/test_pipeline.py 为关键算法测试。现有数据完整时无需重新下载。

三个检查脚本都有保留价值：数据更新后运行 audit_dataset，模型或预处理调整后运行 check_medclip，提交源码前运行 check_repository。它们无需每次正式训练前全部重跑。

## 5. 数据接口约定

主索引位于 outputs/Step0/indices/mimic_5000.csv，包含 patient_id、study_id、image_path、report、split、labels、mask。CSV 中 labels/mask 是 JSON 数组字符串，不应当作普通标量读取。

image_path 中的 images/... 相对于配置 data.image_root（当前为 subset），不是相对于 indices 目录。

疾病顺序固定为：Atelectasis、Cardiomegaly、Consolidation、Edema、Pleural Effusion。mask=1 表示标签有效；mask=0 表示未知或不确定，对应 labels 的数值不能被解释为已知阴性。训练频数组固定为：高频 Atelectasis/Cardiomegaly，低频 Edema/Consolidation，中间组 Pleural Effusion。

特征位于 outputs/Step0/features/<split>.npz：

| 字段 | 形状/含义 |
|---|---|
| patient_id、study_id | N；样本身份，顺序与该 split 索引一致 |
| image_feature、text_feature | N×512，float32，L2 单位向量 |
| labels、mask | N×5，固定疾病顺序 |
| text_prompt_features | P×N×512；训练 P=1，验证/测试 P=5 |
| tokens_before_truncation | N；原文截断前 token 数 |
| branch | 缓存分支标识 |

N 在 train/val/test 中分别为 4000/500/500。推荐使用公共读取函数，它会验证 ID、标签、范数、来源及文件哈希：

```python
from mimic_pipeline.common import ROOT, load_config, load_cache

config = load_config(ROOT / "configs/default.json")
train = load_cache(config, "train")
image_features = train["image_feature"]
text_features = train["text_feature"]
labels = train["labels"]
mask = train["mask"]
```

在项目根目录运行调用上述代码的脚本。缓存、索引、metadata.json 和分组文件应作为完整目录一起交接，不能只拷贝一个 NPZ。

## 6. 必须共同保持的实验约定

- 划分种子固定为 42；训练种子为 42、43、44，三个种子不重新划分患者。
- 冻结 MedCLIP，主训练只学习缓存之上的度量头。若改为端到端微调，需要另外设计数据加载和训练流程。
- 同患者不作负例；已知且非空的阳性集合完全相同的样本对排除为负例。未知标签不当作阴性。
- 验证集用于早停、开发及固定集合几何分析；测试集留作最终评估，不用于选参数或挖掘负样本。
- 检索以同 study_id 为唯一正确配对，候选库和候选顺序固定。数据中存在重复报告文本，严格配对指标可能将相同文本但不同检查 ID 的候选视作不匹配，分析结果时需说明。
- 五版本提示逐版本经过对应文本头并归一化，再求均值并再次归一化；不要先平均基础向量再过头。
- 文本最大长度为 77，已有原文截断比例为 34.42%。更改文本长度、编码模型或预处理后，应重新生成相匹配的特征并单独记录实验。
- 当前实现会核对配置及缓存来源指纹，不能通过忽略校验混用不同实验；新变体应安排单独配置和输出，保留已有基线。

## 7. 接手顺序和 CMD 示例

1. 先读根目录 README.md、configs/default.json、hypotheses.md 和本说明。
2. 按 README 使用 uv 在自己的机器重建环境，不直接复制本机 .venv-medclip。
3. 准备与配置一致的数据、权重及完整产物目录；仅拉取 GitHub 源码不会取得这些被忽略的文件。保持现有目录布局即可避免改动相对路径。
4. 检查缓存并运行算法测试，然后阅读已有 training、evaluation/val 和 geometry 结果。
5. 只对新增变体继续开发和训练。最终方案确定后再执行测试集评估。

以下为代码用法示例，不代表交接时必须重跑全部已有实验；运行评估会重新写对应结果及汇总，先保留需要比较的版本。

```cmd
cd /d E:\MIMIC
".venv-medclip\Scripts\python.exe" -m unittest discover -s tests -v
".venv-medclip\Scripts\python.exe" train_metric.py --method A --seed 42
".venv-medclip\Scripts\python.exe" mine_negatives.py --method B --seed 42 --mode budget
".venv-medclip\Scripts\python.exe" evaluate_retrieval.py --split val
".venv-medclip\Scripts\python.exe" analyze_geometry.py --skip-visualization
```

后续负责人应优先关注：

- 将固定预算 Top20 与冲突优先 Top20 候选接入额外训练，完成相同预算下的对照。候选导出已实现，但训练器目前不消费这些静态候选。
- 整理现有三个种子结果，按 hypotheses.md 完成需要的对照、统计和解释。
- 最终执行 test 检索，形成正式结论、报告和 PPT。

## 8. 交付内容与状态证据

- 代码与说明：六入口、mimic_pipeline/、configs/、tests/、scripts/、README.md、requirements-medclip.txt、hypotheses.md、docs/ 和 Git 配置。
- 本地实验材料：outputs/Step0/ 的 indices、features、checks、training、negatives、evaluation、geometry、logs 及快照文件。
- 原始资料：raw/、subset/、官方 MedCLIP 源码及权重；subset.rar 按已有要求保留，但不能仅凭压缩包名称认定包含完整最新特征和实验结果。
- 既有状态证据：outputs/Step0/completion_summary.json、artifact_manifest.json、checks/data_audit/audit_summary.json、features/metadata.json、logs/training.csv。

数据及结果目录按现有 .gitignore 不进入源码仓库；交接时分别确认下一位负责人已获得所需目录。

目录整理后：原 Step0/ 工具移至 scripts/；重复的 readiness_check.md 合并到本说明，机器可读状态仍以 completion_summary.json 为准。清理了可再生的小样本训练模型与日志、重复的评估/挖掘/几何汇总日志和缓存；保留正式训练 runtime.json、losses.csv/png、logs/training.csv，以及数据准备和特征提取日志。这里不重新划分数据或重新训练模型。

## 9. 可直接用于口头或消息交接的介绍

我负责的数据处理已经完成：固定了 5000 名患者的胸片—报告配对，完成报告清洗、五类标签和未知标签掩码，并按患者以种子 42 划分为 4000/500/500。全量数据审计已通过，冻结 MedCLIP 的 512 维图文特征也已提取完。后续可以直接读取 outputs/Step0/features，无需重新下载或编码。

项目统一通过六个入口运行，共用 configs/default.json，具体实现放在 mimic_pipeline。A/B/C/D 用于比较普通与难负样本、普通与共享/私有度量头；训练使用种子 42、43、44。已有三种子训练、验证集检索及几何结果可以继续检查和复用。接下来主要补齐固定预算候选筛选的训练对照，再做最终测试评估和假设结论。请共同保持数据划分、标签顺序、mask 和检索规则不变。
