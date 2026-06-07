# 用户需求记录（Claude 维护）

## 阶段 A：方向探索

### 研究目标
- 将现有数学建模竞赛论文（FCM-EMD-tSNE-RF 光伏日前功率预测）包装/改进后投稿 **Expert Systems with Applications (ESWA)**。

### 用户约束
- **方向约束**：
  - 核心方法保持 FCM-EMD-RF 框架（聚类 + 分解 + 随机森林）。
  - **不改数据集**：继续用 PVOD（河北 10 站，含 NWP + LMD）。
  - **不改"基于气象数据"这个立足点**：方法必须围绕气象/NWP 数据展开。
  - **完全舍弃 NWP 空间降尺度**（问题四，ERA5 二维插值）。
  - 倾向"不大改即可达发表要求"的方案，但接受必要的方法修补。
- **Motivation 切入点（用户指定）**：
  - 从"光伏预测必须结合气象数据"出发；
  - 引出"现有气象/NWP 数据在处理上存在的困难"作为痛点。

### 已确认（对话中）
- **改进幅度**：用户认可现有方案"只是合格工程、不够创新"，**接受适当大改**。
- **核心预测器**：换成**深度模型主干**（不再以 RF 为主干，原 FCM-EMD-RF 标题中的 RF 基本退役）。
- **算力**：单张 GPU（如 3090/4090），可上 LSTM/GRU/Transformer/Informer 等深度 baseline。

### 已确认（对话中，续）
- **核心新机制**：FCM 软隶属度作为**端到端可学的"天气区制门控"**，软混合多个轻量专家（MoE 思想）。检索确认此组合为相对空白点。
- **分解角色**：从"分解原始输入"改为"分解 NWP−LMD 误差残差"，并**因果化**（修数据泄漏，VMDNet/SciReports2024 为依据），本身即可引用的贡献点。
- **深度主干**：**轻量 GRU/TCN 多专家**（每天气区制一个专家），单 GPU 可跑。RF 退为消融基线之一。
- **t-SNE**：删除（消融负收益 + 致命泄漏/无 transform）。
- **可解释性卖点**：软区制 + 门控权重可视化"当前天气类型 + 各专家贡献"，契合 ESWA(Expert Systems) 决策支持定位。
- **文献补全方式**：用户**手动补关键闭源 PDF** 后，Claude 再正式汇编 Part 1。

### 候选研究方向（一句话，待最终成文确认）
天气区制感知、NWP 误差订正的软门控多专家深度光伏日前功率预测（复用 FCM+分解+树哲学，舍弃降尺度）。

### Part 1 状态：已汇编完成（docs/idea_report.md），参考文献已核实补全。
### Part 2 状态：Method 已落笔并按方案 Z 修订完成。
### 当前阶段：阶段 C 实验设计中。

## 阶段 C：实验约束（用户确认）
- **算力**：单 GPU（3090/4090）。
- **电站范围**：全 10 站。
- **数据划分**：标准时间顺序划分（按时间 train/val/test，贴近深度学习文献惯例）。
  - ⚠️ 连带影响：与前身竞赛 2/5/8/11月最后一周划分**不同**，故竞赛 results/ 旧数值**不能直接复用**；所有 baseline（含 RF）需在新划分下**重跑**以保证公平可比。
- **预测形式**：日前多步，一次预测未来整天 96 点（15min 分辨率），对齐"日前预测"定义与 CRAformer 设置。
- B-4 Introduction：✅ 已按论文风格精修完成（领域→三重困难逐条引用→动机→方法概述→4条贡献）；贡献第4条数值占位待实验。

### 阶段 C 对比/消融/可解释（用户确认，已写入 Part 3）
- **对比阵容（去掉范式复现③）**：①强时序SOTA(DLinear/Informer/PatchTST/iTransformer/TimesNet,统一Time Series Library) + ②订正基线(原始NWP/线性/QM/RF订正) + ③经典(RF)与本文；PV竞品(CRAformer/[4]/[6])仅作†参考引用不复现。依据 AutoPV[13] 等一区论文 baseline 惯例。
- **消融 7 变体**：A完整/B w/o订正(RQ1)/C硬门控(RQ2)/D单模型(RQ2)/E全序列VMD泄漏(RQ3,需"虚高精度"特别叙事)/F w/o VMD特征(RQ3)/G K敏感性。
- **可解释 4 实验**：区制可视化/专家门控权重/订正前后对比/区制贡献个案（含个案）。需模型预留中间量输出接口。
### 阶段 C 状态：实验设计定稿。

## 阶段 D：实现约束（用户确认）
- **策略**：从头构建（路径B）。旧 code/ 代码弃用，仅关键函数逻辑可借鉴（评估指标、FCM调用、VMD）。
- **代码位置**：直接覆盖 code/（保留 code/dataset/ 原始数据）。
- **数据**：从原始 PVOD（code/dataset/station*.csv + metadata.csv）重新预处理；原始数据齐全，行数与processed一致（前身仅做异常值处理未删行）。
- **专家基座**：默认 GRU，config 可切 TCN（Claude判断：GRU稳快、序列不长够用；TCN留作加分消融）。
- **框架**：PyTorch；单 GPU。
- **当前阶段**：D-B2 生成 implementation.md。

### 方案 Z（VMD 角色，用户确认）
- 订正器（模块B）**只吃 [NWP, u(t)]**，干净自洽、推理可部署，训练用 LMD 监督。
- 因果 VMD **分解历史功率/历史NWP**（推理可得），产出多尺度模态作为预测专家的输入特征 → RQ3 落到预测增强，无时间错位、无泄漏。
- 误差残差 e=NWP−LMD 仅用于：训练监督 + 误差结构实证分析（进 motivation/实验图）。
- 待定：第二阶段是否微调订正器（标低置信度，阶段C定）；K=3 暂定。

## 阶段 D-末2：编码前确认清单（用户确认）
- **运行环境**：新建 conda 环境 `zw@PV-Mode-Clust`（Python 3.10）。✅ 已创建。
- **设备**：4× RTX 3090 (24GB)，CUDA 可用；PyTorch 按官网 CUDA 版安装。
- **数据集**：PVOD（station*.csv + metadata.csv）正在上传（~11min），到位后放 `code/dataset/`，由 Claude 跑预处理。
- **运行策略**：混合——快脚本（预处理、最小 smoke test、单站单种子小步验证）Claude 跑；完整训练（10站×5种子两阶段）/全部消融/baseline 大跑由用户跑。
- **README 位置**：`code/`（implementation.md 已定）。

### 已确认技术框架（锁定）
- 四模块：A=FCM软区制 → B=因果**VMD**分解(NWP−LMD)误差残差做订正 → C=K个轻量GRU/TCN专家 → D=软隶属度门控混合
- **耦合方式：两阶段（先订正再预测）**（Claude权衡决定，理由：干净消融回答RQ1、组件可独立消融、单GPU友好）
- t-SNE删除；RF降为baseline；舍弃降尺度
- **订正目标：对齐 LMD 实测**（用户认可）
- **训练用LMD当老师、推理不用LMD**（用户认可，方法可部署前提）

### 数据验证结论（已用真实数据核实，可进论文）
- 字段配对：辐照(nwp_globalirrad/lmd_totalirrad, r=0.72~0.93)、温度(r=0.97)、气压(r=0.99) **可相减做误差**；风速(r=0.55)、风向(r=0.33) 仅作特征不做误差。
- NWP偏差站间差异巨大：station00 偏差均值+1.5、station04 −159.7、station09 −86.4（绝对偏差94~219）→ 证明订正必要性。
- **NWP误差随天气反转**：station00 阴天(低辐照)偏差+60、晴天(高辐照)−113 → 铁证支撑"分天气区制订正"(RQ2)。
- station00 均值偏差小(1.5)但std大(135)：误差是双向波动而非常数偏移 → 强化"分天气订正+VMD分解残差"设计。

### 待用户补全的关键闭源 PDF（放入 docs/papers/，文件名=论文完整标题）
1. A photovoltaic power output dataset (PVOD) — Solar Energy 2021
2. Operational day-ahead PV forecasting based on transformer variant (CRAformer) — Applied Energy 2024
3. Day-ahead PV forecasting based on corrected NWP and domain generalization — Energy & Buildings 2024
4. Day-ahead NWP solar irradiance correction using weather-condition clustering — Applied Energy 2024
5. Research on information leakage in EMD-based forecasting — Scientific Reports 2024
（其余如 Eng.Appl.AI 2025、Renewable Energy 2025、Earth Sci.Info 2025 为加分项，有则更好）

### 已下载（docs/papers/，arXiv）
VMDNet、TimeExpert、MoWE、IDS-Net、Fuzzy Cognitive Maps Survey、003131A(自有竞赛论文)

### 关键事实（来自代码与结果文件，已核实）
- 现有消融（acc，total 测试集，10 站均值口径）：
  - problem_02（RF + 仅历史功率）：≈ 89.7%
  - problem_03_1（RF + 原始 NWP 辐照度，无 FCM/EMD/tSNE）：≈ **95.87%（最高）**
  - problem_03_2（EMD+tSNE，无 FCM）：≈ 90.96%（**低于 03_1**）
  - problem_03_4（FCM + 原始 NWP）：≈ 95.62%
  - problem_03_3（完整 FCM-EMD-tSNE-RF）：≈ 95.74%（**仍低于 03_1**）
- 结论：**当前 EMD-tSNE 特征通道不仅没有提升，反而略微损害精度**；FCM 分簇收益也很微弱。
- 方法学隐患：t-SNE 在全量数据（含测试集）上 fit_transform，存在数据泄漏，且 t-SNE 无 transform，无法对真正的新样本一致映射 —— 这在期刊评审中是硬伤。
