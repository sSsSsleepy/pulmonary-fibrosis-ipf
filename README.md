# IPF vs 非 IPF：患者级 CT 基础模型方案

本项目为研究用途的第一阶段二分类基线。原始 Excel 和 `ILDclean` 只读使用，不会改写，也不会把患者数据上传到外部服务。

## 为什么先用 MedSigLIP

目标是 0/1 预测，不需要生成长文本。MedSigLIP 是医学图文基础模型，影像编码器和文本编码器各约 4 亿参数，训练数据明确包含 CT 切片。第一轮同时评估：

1. 文本提示（prompt）零样本分数；
2. 冻结 MedSigLIP 后，对患者 CT 切片嵌入训练逻辑回归线性探针。

默认从每次 CT 中选 16 张轴位肺窗切片，进行切片级编码，再做 mean+max 池化。主分析每位患者只取时间最早的可用肺部 CT；训练、验证、测试严格按患者划分。

## 已核实的数据结构

- 2,895 次有标签 CT，全部能找到影像；
- 740 位唯一患者：IPF 398，非 IPF 342；影像质控后 738 位具有可用肺部 CT，其中 IPF 396、非 IPF 342；
- 同一患者标签无冲突；
- 原始影像约 863 GiB、14,461 个 NIfTI 序列；
- 标签队列中有 13,028 个序列，其中 4,929 个层厚不超过 1.5 mm；
- 临床表当前仅能匹配 240/740 位标签患者，所以影像基线先使用完整 740 人，多模态实验另设匹配子队列。

## 运行

```powershell
.\setup_local.ps1

.\.venv\Scripts\python.exe -m ipf_binary.build_manifest

# 先测试 2 例；MedSigLIP 首次使用前需要在 Hugging Face 页面接受 HAI-DEF 条款并登录。
.\.venv\Scripts\python.exe -m ipf_binary.extract_medsiglip_embeddings --limit 2

# 构建胸部序列优先、排除非胸部影像后的肺部基线清单。
.\.venv\Scripts\python.exe -m ipf_binary.build_lung_index

# 提取肺部质控队列；当前正式产物为 738 位患者。
.\.venv\Scripts\python.exe -m ipf_binary.extract_medsiglip_embeddings --manifest artifacts/manifests_chest_priority/lung_index_ct_manifest.csv --output-dir artifacts/embeddings/medsiglip_lung_index

# 正式训练（冻结嵌入，仅用验证集选择 C 和分类阈值）。
.\.venv\Scripts\python.exe -m ipf_binary.train_probe --embedding-index artifacts/embeddings/medsiglip_lung_index/embedding_index.csv --output-dir artifacts/results/medsiglip_lung_linear_probe_formal --seed 20260827

# 重载模型并审计预测一致性、患者级划分、指标和文件指纹。
.\.venv\Scripts\python.exe -m ipf_binary.audit_probe --result-dir artifacts/results/medsiglip_lung_linear_probe_formal
```

## 修正标签的正式时间外验证

旧的随机训练/验证/测试划分已用于方案迭代，只保留为探索性结果。修正标签实验仅使用 651 名多次记录标签一致的患者，并将 2024–2026 年 83 人锁定为时间外测试集。

```powershell
# 生成新队列：651 人，IPF 326，非 IPF 325。
$CorrectedLabels = 'D:\private_medical_data\标准化出院诊断.xlsx'
.\.venv\Scripts\python.exe -m ipf_binary.build_corrected_cohort `
  --corrected-labels $CorrectedLabels

# 检查跨患者 DICOM UID 重复，并对入模 CT 生成 SHA-256 指纹。
.\.venv\Scripts\python.exe -m ipf_binary.audit_corrected_cohort

# 密封输出目录不会复用缺少 CT 指纹、模型 commit 或文件哈希的旧特征。
.\.venv\Scripts\python.exe -m ipf_binary.extract_medsiglip_embeddings `
  --manifest artifacts/manifests_corrected/corrected_lung_index_ct_manifest.csv `
  --output-dir artifacts/embeddings/medsiglip_corrected_sealed

# 只在 2011–2023 年开发队列内选模型和阈值；输出目录写入时间外评估锁，禁止静默覆盖。
.\.venv\Scripts\python.exe -m ipf_binary.train_corrected_probe `
  --embedding-index artifacts/embeddings/medsiglip_corrected_sealed/embedding_index.csv `
  --output-dir artifacts/results/medsiglip_corrected_temporal_sealed
.\.venv\Scripts\python.exe -m ipf_binary.audit_corrected_probe `
  --result-dir artifacts/results/medsiglip_corrected_temporal_sealed
```

正式报告同时给出采集年份、扫描设备、层厚和重建核构成的元数据对照模型，用于判断影像模型是否只学到了采集域差异。自由文本的序列/检查描述不入模，避免诊断关键词或患者信息造成目标泄漏。

## 肺野/肺叶分割和可视化

正式批处理默认使用 `lungmask` 的 `R231` 病理肺模型生成左右肺野标签；它在严重间质性改变病例上更稳健，且能够在当前队列规模上完成。若研究问题明确需要肺叶标签，可显式传入 `--modelname LTRCLobes_R231 --fillmodel R231`，但该组合的逐连通域融合后处理明显更慢，应先另做小样本计时和质控。

```powershell
# 安装本地分割依赖。
.\.venv\Scripts\python.exe -m pip install -e '.[segmentation]'

# 先从开发队列选定少量病例，生成 QC 预览并人工复核。
.\.venv\Scripts\python.exe -m ipf_binary.segment_lungs --limit 8 --write-previews

# 冒烟通过后对 651 人运行可恢复批处理。
.\.venv\Scripts\python.exe -m ipf_binary.segment_lungs --write-previews

# 在肺区域内重新提取特征，用固定时间外方案与原基线比较。
.\.venv\Scripts\python.exe -m ipf_binary.extract_medsiglip_embeddings `
  --manifest artifacts/manifests_corrected/corrected_lung_index_ct_manifest.csv `
  --mask-index artifacts/segmentation/lungmask_corrected/segmentation_index.csv `
  --input-mode lung-masked `
  --output-dir artifacts/embeddings/medsiglip_corrected_lung_masked_sealed

# 用同一训练方案拟合肺掩膜输入，并逐患者配对比较时间外预测。
.\.venv\Scripts\python.exe -m ipf_binary.train_corrected_probe `
  --embedding-index artifacts/embeddings/medsiglip_corrected_lung_masked_sealed/embedding_index.csv `
  --output-dir artifacts/results/medsiglip_corrected_lung_masked_temporal_sealed
.\.venv\Scripts\python.exe -m ipf_binary.audit_corrected_probe `
  --result-dir artifacts/results/medsiglip_corrected_lung_masked_temporal_sealed
.\.venv\Scripts\python.exe -m ipf_binary.compare_corrected_probes

# 只解释显式指定的开发队列 CT；不会自动挑选“漂亮”病例。
.\.venv\Scripts\python.exe -m ipf_binary.explain_corrected_probe --ct-id CT00000000
```

`segment_lungs` 生成的是肺部解剖掩膜。`explain_corrected_probe` 只接受与密封训练清单、分类器哈希、MedSigLIP commit、输入模式和切片参数完全一致的特征，生成的是局部遮挡后分类概率变化热图，并固定标注为 **model attention, not fibrosis segmentation**；在没有医生像素级标注和独立分割评估前，不得把该热图称为纤维化病灶分割。

如果 MedSigLIP 尚未完成 Hugging Face 授权，可先用公开的 MIT 许可 BiomedCLIP 做端到端技术冒烟：

```powershell
.\.venv\Scripts\python.exe -m ipf_binary.make_smoke_manifest
.\.venv\Scripts\python.exe -m ipf_binary.extract_biomedclip_embeddings --manifest artifacts/manifests/smoke_index_ct_manifest.csv --output-dir artifacts/embeddings/biomedclip_smoke --slices 4
.\.venv\Scripts\python.exe -m ipf_binary.train_probe --embedding-index artifacts/embeddings/biomedclip_smoke/embedding_index.csv --output-dir artifacts/results/biomedclip_smoke
```

BiomedCLIP 的训练语料来自生物医学文献图文对，并非专门的 3D 胸部 CT 模型；其结果只能证明工程链路可跑，不能替代 MedSigLIP 主实验。

若只想验证数据代码：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## 评价与防泄漏

- 模型和正则化强度只由训练/验证集确定；
- 决策阈值只在验证集确定；
- 测试集在最终评估前锁定；
- 报告 ROC-AUC、PR-AUC、敏感度、特异度、PPV、NPV、F1、Brier 分数与混淆矩阵；
- 需进一步按扫描设备、性别、年龄、层厚和时间做亚组评估及外部验证。

本项目输出不是临床诊断工具，不得直接用于患者处置。

## 数据隐私

Git 仓库只发布代码、测试和队列级汇总结果。原始 CT、Excel、患者级清单与预测、嵌入、缓存及训练权重均由 `.gitignore` 排除，不得通过 `git add -f` 强制加入。
