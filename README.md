# Dynamic-SLSQP

Acute toxicity prediction with a linear-tabular multimodal ensemble

本仓库整理最终去除GLU1的主模型，覆盖原始数据清洗、结构去冗余、固定划分、
特征提取、外层交叉拟合、OOF权重学习及保存模型后的预测。
这是本研究的组合实现，不是ChemBERTa或其他已发表模型的官方仓库。

## Directory Structure / 目录结构

| 文件 | 用途 |
|---|---|
| `prepare_data.py` | 合并原ry.py和split_data.py的数据处理逻辑 |
| `main_model.py` | 最终模型原始计算流程 |
| `runtime_artifacts.py` | 保存预处理器、网络、回归器、预测与优化记录 |
| `run.py` | 完整训练入口，提供路径参数和数据/缓存检查 |
| `predict.py` | 使用保存的五折完整流程预测新分子，不重新训练 |
| `verify_release.py` | CPU结构与流程检查，不下载预训练模型、不进行完整训练 |
| `requirements.txt` | 主模型安装依赖的兼容范围 |
| `environment/` | 原始服务器环境快照及核心版本清单 |
| `reference_results/` | 原归档运行的小型结果表和数据哈希，不是本次重新训练结果 |
| `CHANGELOG.md` | 与归档代码相比的工程整理说明 |

## 1. 环境

建议Python 3.12及GPU训练环境。先创建独立环境，再安装：

```bash
python -m pip install -r requirements.txt
python verify_release.py
```

PyTorch需要与设备/CUDA驱动匹配。可先按PyTorch官方安装说明安装所需版本，
再安装其余依赖。不要直接升级或覆盖正在使用的实验环境。

历史服务器曾记录RDKit 2026.03.2和scikit-learn 1.8.0；新环境训练时会在
`run_metadata.json`记录实际安装版本。仅有随机种子不能保证跨版本、硬件和GPU
实现逐位复现。要复算历史数值，应使用历史划分、对应特征缓存及原环境。


## Python Environment / 实验环境记录

以下版本摘自作者提供的服务器requirements.txt，表示该次环境导出的记录，
不能单凭它证明所有历史实验都使用同一环境，也不表示已在本机重新安装验证。

| Library | Recorded version | Notes |
|---|---|---|
| Python | CPython 3.12 | 从torch wheel的cp312标记识别；完整补丁版本未记录 |
| PyTorch | 2.8.0+cu128 | 原文件为Linux x86_64专用wheel链接 |
| NumPy | 2.3.2 | 原文件精确记录 |
| Pandas | 3.0.3 | 原文件精确记录 |
| SciPy | 1.17.1 | 原文件精确记录 |
| scikit-learn | 1.8.0 | 原文件精确记录 |
| RDKit | 2026.3.2 | 原文件精确记录 |
| Transformers | 5.9.0 | 原文件精确记录 |
| Joblib | 1.5.3 | 原文件精确记录 |
| tqdm | 未提供数字版本 | 原文件仅有服务器file://安装路径 |

`cu128`是PyTorch构建的CUDA标记，不是对本机CUDA Toolkit或显卡驱动版本的检测结果。
服务器原始快照保存在`environment/requirements_server_snapshot.txt`，用于环境溯源，
不要直接在其他机器运行pip install -r该快照：其中的/croot和/tmp/build路径通常不可访问，
而Linux CPython 3.12 wheel也不能直接安装到Windows或其他Python ABI。

常规安装使用仓库根目录的`requirements.txt`。需要参考服务器核心版本时，
可在Python 3.12独立环境中先安装适合目标设备的PyTorch，再参考
`environment/requirements_server_core.txt`；后者是从快照提取的版本清单，
没有经过本次全量安装验证。

```bash
python -m pip install -r requirements.txt
python -m pip check
python verify_release.py
```

模型训练完成后，`results/main/run_metadata.json`记录实际运行的Python和核心包版本。
发表论文时优先引用对应实验运行的元数据，而不是把其他时间导出的环境快照当作全部历史环境。

## 2. 数据来源及标签

来源：https://toxric.bioinforai.tech/download 。原研究下载日期：2025-11-21。
研究终点：mouse intravenous LD50，输入标签为数据库已经提供的pLD50数值。
程序不进行额外的负对数、剂量单位或分子量转换。
不要直接把未经转换的mg/kg原始剂量当成本程序的pLD50标签输入。

默认列名（含空格）为：

```text
Canonical SMILES
mouse_intravenous_LD50
```

数据不随代码包发布。先核实数据来源的使用和再分发要求，再决定是否公开数据。
训练数据包含小分子化合物，不应将全部样本描述为已上市药物。

## 3. 从原始CSV开始

在代码目录打开终端，将原始CSV放在`data/raw_mouse.csv`，运行：

```bash
python run.py --raw-csv data/raw_mouse.csv --data-dir data/splits --cache-dir cache/main --output-dir results/main --save-ml-models
```

或者分两步：

```bash
python prepare_data.py --raw-csv data/raw_mouse.csv --output-dir data/splits
python run.py --data-dir data/splits --cache-dir cache/main --output-dir results/main --save-ml-models
```

两种方式选择一种，不要在已生成划分后再次给run.py传`--raw-csv`。
如列名不同，可以同时指定`--smiles-col`和`--label-col`。
提供的标签需要与目标pLD50尺度一致，仅改列名不会转换标签含义。

数据处理顺序与原程序一致：删除SMILES缺失；完全相同字符串保留首次记录；
RDKit结构解析；半径2、2048位Morgan指纹按原始顺序执行贪心相似度过滤，
仅当与已保留分子的最大Tanimoto小于0.80时保留；然后执行原划分脚本的标签
缺失/非数字处理，使用两次seed=42的train_test_split实现80/10/10划分。
不执行盐剥离、规范SMILES重写、标签聚合或骨架划分。
`preparation_audit.json`记录各阶段数量及输入/输出文件SHA256。
同样的顺序、版本和源数据下，历史数量为14318及11454/1432/1432。
如果数量不同，先核对输入与版本，不要人为删补到这些数目。

## 4. 已有固定划分或历史缓存

把原来的train.csv、val.csv、test.csv放到同一目录，再运行：

```bash
python run.py --data-dir data/fixed --cache-dir cache/main --output-dir results/main --save-ml-models
```

如必须使用原先的v40_etkdg_3dmlp_train/val/test.pt，缓存目录可以指定其原位置。
历史缓存缺少新的逐行SMILES溯源文件，因此默认拒绝直接使用；只有在你已核对
分子顺序、分子数、预训练模型及提取协议后，才显式添加`--trust-legacy-cache`。
形状相同不代表分子顺序相同。首次信任导入的历史缓存无法通过本程序追溯其原始
生成过程；新的provenance记录只记录本次确认时的输入对应关系。

新生成缓存会保存有序SMILES哈希及形状检查。更换提取算法、RDKit版本或预训练
权重时应使用新的缓存目录，这个轻量检查不能替代完整环境溯源。
PT/joblib使用反序列化，只加载自己生成或可信来源的文件。

首次提取需要获取`DeepChem/ChemBERTa-77M-MTR`。可以使用
`--chemberta-model /path/to/local/model`指定已经下载好的目录；
离线运行添加`--offline`。本程序不会清空用户代理设置。

## 5. 实际模型与训练协议

- 冻结ChemBERTa，序列最大长度128，最后隐藏层按attention mask平均池化为384维。
- 理化描述符与结构指纹：10+167 MACCS+1024 Morgan=1201维。
  十项为MolWt、MolLogP、TPSA、NumHDonors、NumHAcceptors、NumRotatableBonds、
  NumAromaticRings、FractionCSP3、BertzCT和NumHeteroatoms。
- 三维输入：10项几何描述符+12项USR+1项有效性标志=23维。
  ETKDGv3请求10个候选，MMFF94s或UFF优化后比较候选能量；不保证10个成功构象、
  所有候选完全收敛或全局最低能量构象。
- 编码：384→256；1201→256；23→128→64。线性编码层后为BN和GELU。
  没有TabularGLU。拼接576维后使用Linear+Sigmoid Fusion Gate逐通道相乘，
  经576→256→128主干和128→1线性头输出连续目标。
- 主训练集外层5折KFold shuffle=True seed=42；每折内部停止集比例0.10，
  内部划分种子1000+折号。5个网络种子为42/1024/2026/8888/9999。
- 内部阶段最多搜索50轮，按内部停止集的变换尺度LogCosh最低轮数选择；
  不是patience提前终止。随后并回停止集，重新拟合外层训练预处理器、重新初始化网络。
- AdamW，学习率3e-4，weight decay=1e-2，batch size=64。
- 理化10维和有效构象22维连续描述符标准化仅在当前训练子集拟合；
  无效构象的连续输入置零，最后有效性标志保持0/1。
- 每折5个网络及7种回归器共用该折Yeo–Johnson standardize=True标签。
  预测分别逆变换至pLD50之后才做平均和集成。
- 每折5×128维表示拼接640维，以当前外层训练样本拟合各ML流水线的StandardScaler。
  七种回归器为RidgeCV、BayesianRidge、ElasticNetCV、HistGradientBoostingRegressor、
  RandomForestRegressor、ExtraTreesRegressor、KNeighborsRegressor。
- 5个深度预测折内平均为DL_mean，与七种ML构成8路候选。
  OOF按当前外层留出样本索引回填，不跨折平均，不把不同折的隐藏坐标混合后重拟合。
- 不执行基于OOF误差的候选筛选，八路候选全部参与SLSQP。
  OOF MSE仅用于报告各候选误差。SLSQP目标为MSE+0.015*Var(y)*sum((w-p)^2)，
  权重非负且和为1，参考分配为DL_mean=0.5，其余七路均分0.5。
  优化失败回退参考分配，随后归一化。权重为全局固定，不是每分子动态权重。
- 验证/测试：同类候选跨5折平均后加权。与使用相同权重先折内加权、后跨折平均等价。
- 不在5折后再用全部主训练数据训练一个新的单一基础模型。

## 6. 预测新分子

训练时必须使用`--save-ml-models`，才能保存25个正式网络、35个传统回归器及
预处理组件。树模型较大，完整训练输出可能需要数十GB，需事先留足空间。
若只需要训练评价，可省略此选项，但该输出不能直接用于完整集成推理。

新CSV只需要相同SMILES列，不需要毒性标签：

```bash
python predict.py --run-dir results/main --input-csv data/new_molecules.csv --cache-dir cache/new --output-csv results/new_predictions.csv
```

输出各个跨折平均候选和最终`pred_pLD50`。输入重复记录会保留，以维持输入行对应。
模型输出是连续pLD50值，不是概率。逆变换不能直接恢复成mg/kg剂量。
部署应保留完整训练目录；仅有神经网络pth不足以运行完整预测器。

## 7. 结果与解释范围

主要输出：`table_model_results.csv`、`dynamic_weights.csv`、逐样本预测、
`cross_fitted_arrays.npz`、`diagnostics.json`和`saved_run/`。
DL-E2E是深度直接预测的种子/跨折平均；Static-Ensemble是仅七种ML等权平均；
Dynamic-SLSQP是全部八路候选的固定权重组合。

OOF用于拟合集成权重，拟合后的OOF组合误差不是最终集成器无偏测试性能。
固定留出测试集不等同于独立外部数据；历史开发中该测试集已被多次查看。
新环境重跑不会自动构成外部验证。预训练数据重叠及结构覆盖仍需针对研究设计评估。

## 8. 上传GitHub

代码包不含数据、模型大文件、密钥或个人绝对路径，不会自动创建远程仓库或上传。
在GitHub创建空仓库后，在本目录执行：

```bash
git init
git add .
git status
git commit -m "Add final linear-tabular acute toxicity model"
git branch -M main
git remote add origin https://github.com/YOUR_ACCOUNT/YOUR_REPOSITORY.git
git push -u origin main
```

先查看git status，确认未加入原始数据、模型文件或凭据。data、cache、results目录
默认忽略；仅保留data/README.md作为数据获取说明。
发布前由作者决定代码许可证，本包未替作者声明MIT等授权。

## Sources

- TOXRIC: https://toxric.bioinforai.tech/download
- Frozen backbone: https://huggingface.co/DeepChem/ChemBERTa-77M-MTR
- RDKit: https://www.rdkit.org/
- PyTorch installation: https://pytorch.org/get-started/locally/

所有参考结果保留原运行来源；请勿将流程自检的合成数据结果报告为研究性能。

## 2026-11-17 无筛选版本

按作者要求删除OOF MSE相对阈值筛选，八路候选全部进入权重优化。
优化后某一路权重仍可能为0，这是约束优化结果，不是误差阈值筛除。
归档参考运行的八路原本均通过筛选，因此对完全相同的候选矩阵，这项修改不改变
参考分配、优化目标或候选顺序。没有为此次修改重新训练完整数据。
