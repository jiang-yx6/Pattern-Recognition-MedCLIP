# Step0 集中交付目录

- indices/：完整与分 split 索引、固定名单、标签顺序、类别频数、高低频分组、原始报告段落来源。
- features/：train/val/test 基础图文特征、提示特征、疾病提示、模型与文件指纹。图文基础维数均为 512。
- hypotheses.md：H1-H3 预先约定的比较组、主指标与预期方向；不是已证实的结论。
- config_snapshot.json：本次集中产物使用的公共配置。
- checks/：全量数据审计、16 对模型检查和 5 对图文核对页。
- logs/：准备、提取和训练的运行记录与训练汇总 CSV。
- training/：A-D 各种子的正式模型、损失曲线、梯度/冻结检查、耗时与峰值 CUDA 显存。
- negatives/：训练集负例、冲突清单及 20 对核对样本。
- evaluation/：固定候选库的检索指标、逐查询排名、均值/标准差、配对 bootstrap。
- geometry/：固定 val 的共享/私有几何指标、SVD 和联合 t-SNE。
- artifact_manifest.json：交付文件清单与 SHA256（在最终检查时生成，不包含其自身哈希）。

图片仍位于配置的 data.image_root/images；索引中的 image_path 相对于 data.image_root，不相对于本文件所在目录。
此目录含真实报告和逐样本数据，保留在本地，不因集中整理而改变原数据的访问和共享要求。

整理后保留正式训练记录，小样本 smoke_training/ 及重复的聚合日志已删除；以后使用 --smoke-test 可重新生成小样本输出。logs/ 中的 training.csv 是正式训练汇总，各次训练的 runtime.json 保存完整耗时和显存记录。
检查工具源码位于项目根目录的 scripts/，不在这个产物目录中。后续开发交接说明位于 docs/handoff.md。
