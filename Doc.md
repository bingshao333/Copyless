1. 引言

1.1 项目背景与目标
随着学术研究的快速发展和在线资源的极大丰富，学术诚信问题日益受到关注。传统的基于关键词匹配或N-gram的论文查重系统在检测复杂的改写、释义（paraphrasing）等抄袭行为时存在局限性。为了应对这一挑战，我们计划开发一款名为 “Copyless” 的新一代论文查重产品。

Copyless 将借鉴 Copyleaks 等先进产品的理念，利用前沿的自然语言处理（NLP）技术和向量数据库技术，实现对论文深层次的语义相似性检测。项目的主要目标是：
- 构建一个全面的学术论文向量库：以 arXiv 预印本论文库为基础，建立一个句子级别的向量索引，作为查重的比对数据源。
- 实现高精度的相似性检测：结合语义相似度（捕捉意译和改写）和文本编辑距离（捕捉字面微小改动），提供更精准的查重结果。
- 提供多层次的重复度分析：将句子级别的相似性划分为“完全相同 (identical)”、“微小改动 (minor changes)”、“释义改写 (paraphrased)”和“原创 (original)”四个等级，为用户提供更精细的分析报告。
- 生成全面的查重报告：综合整篇文章的检测结果，计算总体重复率，并显示最相似的N篇源文献及其各自的重复贡献度。
  
1.2 文档目的
本技术文档旨在详细阐述 Copyless 系统的整体架构、核心技术模块、实现流程、关键参数配置以及未来的评估与优化方向，为项目的研发、部署和迭代提供全面的技术指导。


---

2. 系统架构设计

Copyless 系统主要由两个核心流程组成：数据处理与索引构建 和 实时查重检测。

2.1 数据处理与索引构建流程
此流程为离线批处理过程，旨在将海量的 arXiv 论文数据转化为可供快速检索的向量索引。

1. 数据源获取：从 arXiv 获取全部论文的源文件（如 LaTeX 或 PDF），保存于 fnlp@jw-10-176-51-26:/mnt/data/corpus/DocAI/arXiv_2023-12$。
2. 文本提取与预处理：
  - 解析源文件，提取纯文本内容。
  - 进行文本清洗，去除无关字符、格式标记等。
  - 句子切分：使用专门的句子分割工具（如 NLTK, spaCy）将论文内容切分成独立的句子。这是构建句子级向量库的基础。
3. 句子向量化 (Sentence Embedding)：
  - 加载预训练的句子嵌入模型。
  - 将每个句子输入模型，生成固定维度的向量表示（Embedding）。
4. 数据持久化与索引：
  - 将每个句子的向量、原始文本、以及元数据（如所属论文ID、在论文中的位置等）批量写入 Qdrant 向量数据库。
  - Qdrant 内部会为这些向量构建高效的 HNSW（Hierarchical Navigable Small World）索引，以支持后续的快速相似性搜索。
    
2.2 实时查重检测流程
此流程为在线服务，处理用户提交的待查重论文。

1. API 接收请求：用户通过 API 提交待查重的论文文本。
2. 输入文本预处理：对待查重论文执行与索引构建时相同的文本清洗和句子切分操作。
3. 逐句处理与相似性分析：
  - 向量化：将待查论文的每个句子转化为向量。
  - 向量检索：以该句子向量为查询，在 Qdrant 数据库中搜索最相似的 Top-K 个向量。Qdrant 的搜索基于余弦相似度。
  - 精确匹配与分析：对检索到的 Top-K 个候选句子，进行第二阶段的精确计算：
    - 计算待查句子与每个候选句子的文本编辑距离（Levenshtein Distance）对语言的文本编辑距离要考虑一下，要引入多次翻译。
    - 结合第一步得到的语义相似度（余弦相似度）和编辑距离，根据预设的阈值规则，将该句子的查重结果判定为 identical, minor changes, paraphrased, cited,original 四个级别之一。
4. 结果聚合与报告生成：
  - 句子级结果汇总：聚合所有句子的判定结果。
  - 文档级重复率计算：根据不同重复级别的句子数量及其权重，计算整篇论文的综合重复率。
  - 相似文献筛选：统计每个源文献的句子匹配次数和相似度得分，筛选出与待查论文最相似的 Top-N 篇源文献，并计算每篇源文献的重复贡献率。
5. API 返回结果：将生成的详细查重报告（包括总体重复率、相似文献列表、句子级别的详细比对等）通过 API 返回给用户。
  

---

3. 核心技术模块详解

3.1 向量数据库 (Qdrant) 架构与配置

Qdrant 是一个用 Rust 编写的高性能向量数据库，支持分布式部署和水平扩展，非常适合存储和查询亿级的句子向量 。

- 集群规模与硬件建议：
整个 arXiv 语料库包含数百万篇论文，句子总量将达到数亿甚至十亿级别。假设初期目标是索引 5000 万个句子向量。
  - 集群拓扑：建议采用至少3-5个节点的集群以保证高可用和读写分离。随着数据量增长，可以平滑地增加节点数量。对于大规模部署，可以考虑8节点集群 。
  - 节点规格：
    - CPU：推荐使用高主频的多核 CPU（如 16 核或以上），因为 HNSW 索引的构建和搜索都是计算密集型任务 。
    - RAM：内存是关键。Qdrant 可以将部分或全部向量和索引加载到内存中以实现极低延迟。对于768维向量，每个向量约需 768 * 4 bytes = 3 KB 的存储空间。考虑索引开销（HNSW索引会占用比原始向量更多的空间），5000万向量至少需要 50M * 3KB * 1.5 (开销系数) ≈ 225 GB 的内存。因此，建议单节点配置 128GB 或 256GB RAM 。
    - 存储：必须使用高性能的 NVMe SSD。磁盘 I/O 是向量搜索性能的主要瓶颈之一，尤其是在内存无法完全容纳所有数据时 。
    - 网络：节点间需要高带宽、低延迟的网络（如 10Gbps 或更高），以保证分片和副本之间的数据同步与查询路由效率 。

- Qdrant 集合 (Collection) 与索引配置：
针对768维的句子向量，HNSW 索引参数的调优至关重要，它直接影响了搜索的召回率、延迟和资源消耗。
  - M (Max Connections)**: 控制图中每个节点的最大连接数。更高的 M 值可以提升索引质量和召回率，但会显著增加内存消耗和索引构建时间。推荐值：16 (通用) 或 32 (追求更高召回率) 。
  - ef_construction (Construction Search Size)**: 索引构建期间的搜索范围大小。该值越大，索引质量越高，但构建速度越慢。对于追求高质量离线索引的场景，可以设置得较高。推荐值：256 或 512 。
  - ef (Search Size): 查询时的搜索范围大小。该值是延迟和召回率之间最直接的权衡。初始可设置为 128，并根据实际测试结果进行调整 。
  - 量化 (Quantization): 为了降低内存占用和加速查询，可以启用标量量化 (Scalar Quantization)，将 float32 的向量压缩为 int8。这会带来微小的精度损失，但能将内存占用减少约75%，并可能提升查询速度 。对于学术查重这种对精度要求极高的场景，需要在评估后谨慎使用或不使用。

3.2 相似度计算与重复级别判定

这是 Copyless 的核心算法模块，通过双重指标融合实现对不同层次重复行为的精准识别。

- 计算流程：
  1. 一级筛选 (语义召回) ：对于待查句子 S_q，在 Qdrant 中执行 Top-K（例如 K=5）的相似性搜索，得到 K 个候选句子 {S_c1, S_c2, ..., S_ck}。这一步返回的结果本身就带有语义相似度分数（余弦相似度 Sim_cos）。
  2. 二级精排 (词法分析) ：对每个候选句子 S_ci，计算其与 S_q 的归一化 Levenshtein 编辑距离相似度 Sim_lev。
    - 归一化公式：Sim_lev = 1 - (LevenshteinDistance(S_q, S_ci) / max(length(S_q), length(S_ci))) 。该公式将编辑距离转换为一个 [0, 1] 区间的相似度分数，1表示完全相同。
      
- 重复级别判定规则：
我们设计一个基于规则的决策树来判定句子的重复级别，该方法比单一的加权融合得分更具解释性和鲁棒性。阈值 T_cos_high, T_cos_low, T_lev_high, T_lev_med 需要通过在标准抄袭检测数据集（如 PAN 系列）上进行测试来校准 。

重复级别 (Level)
判定规则 (伪代码)
解释
Identical

IF Sim_lev >= T_lev_high (e.g., 0.99)
文本几乎完全一样，可能只有标点、空格或个别字母的差异。
Minor Changes

ELSE IF Sim_lev >= T_lev_med (e.g., 0.90) AND Sim_cos >= T_cos_high (e.g., 0.95)
文本有少量词汇替换、增删，但句子结构和主体内容高度一致。
Paraphrased

ELSE IF Sim_cos >= T_cos_high (e.g., 0.88)
文本在词汇和句式上有较大改动（因此Sim_lev不高），但语义上高度相似，属于典型的释义改写。
Original
ELSE
与数据库中的任何句子在语义和词法上均无显著相似性。
Cited

判断为Minor Changes或Paraphrased，但加了引用

- 加权融合方案 (备选)：
作为备选或辅助方案，可以计算一个融合得分。权重的设定应体现我们的业务逻辑：语义相似是基础，词法相似决定了抄袭的“明目张胆”程度 。
    *   **公式**: Score_final = w_cos * Sim_cos + w_lev * Sim_lev
    *   **权重依据**: w_cos 应占主导，例如 w_cos = 0.7, w_lev = 0.3。因为即使词法完全不同，只要语义高度一致，也应被视为潜在的重复。
    *   **基于融合得分的阈值划分**:
        *   Score_final >= 0.98 -> Identical
        *   0.92 <= Score_final < 0.98 -> Minor Changes
        *   0.85 <= Score_final < 0.92 -> Paraphrased
        *   Score_final < 0.85 -> Original
注意：此方案的阈值设定较为困难，不如基于规则的决策树直观。
3.3 引用识别与抄袭排除
正确处理引用是区分学术不端与规范写作的关键。
1. 挑战：一个句子可能与源文献内容高度相似，但如果其上下文中有对该源文献的正确引用（如 ... as shown by Author (2025) <span data-key="38" class="reference-num" data-pages="undefined">1</span>.），则不应被标记为抄袭。挑战在于，引用标记  可能出现在句子的前面、后面，甚至在相邻的句子中 。
2. 实现步骤：
  - 步骤一：参考文献解析：在论文预处理阶段，识别并解析文末的参考文献列表（Bibliography/References）。建立一个从引用标记（如 , [Author et al., 2025]）到具体文献（如 paper_id）的映射表。
  - 步骤二：引用标记定位：在论文正文中，使用正则表达式或模式匹配，识别所有出现的引用标记。以上两个步骤我们有实现代码
  - 步骤三：上下文窗口扫描：当一个输入句子被初步判定为 Minor Changes 或 Paraphrased，并且我们已经通过Qdrant知道了其最可能的来源文献 source_paper_id 时，启动上下文扫描。
  - 步骤四：关联与判定：
    - 定义一个“上下文窗口”，例如，包含当前句子、其前2句和后2句，或者整个段落。
    - 检查这个窗口内是否存在指向 source_paper_id 的引用标记。
    - 如果存在这样的引用标记，则将该句子的分类从 Paraphrased 或 Minor Changes 修正为 Cited。

---

4. 整体重复率计算与报告生成

4.1 文档级重复率计算
简单的将被标记为非原创的句子数量除以总句数是不够的，因为它无法区分不同重复级别的严重性。我们采用加权计算方式：
- 令 N_i, N_m, N_p 分别为被判定为 identical, minor changes, paraphrased 的句子数量。
- 令 N_total 为待查论文的总句数。
- 设定权重 W_i = 1.0, W_m = 0.8, W_p = 0.6。
- 整体重复率 (Overall Similarity Score):
Score = (N_i * W_i + N_m * W_m + N_p * W_p) / N_total

4.2 相似文献筛选
在逐句检测的过程中，我们需要记录每个匹配上的源句子的所属论文ID。检测完成后，通过聚合这些信息来确定最主要的抄袭来源。
1. 创建一个计分板，key 为源论文ID，value 为一个累加得分。
2. 遍历所有非原创句子，根据其重复级别和相似度得分为其对应的源论文ID累加分数。
3. 排序计分板，得到 Top-N (例如 N=3) 的最相似源文献列表。
4. 对每一篇 Top-N 的源文献，单独计算其与待查论文的重复率（只考虑来自该源文献的匹配句子）。
  
4.3 查重报告
最终生成的报告应包含以下内容：
- 概览部分：
  - 总体重复率。
  - Identical, Minor Changes, Paraphrased , cited三个级别的句子数量和占比。
  - Top-N 相似文献列表及其各自的重复贡献率。
- 详情部分：
  - 展示待查论文的全文。
  - 使用不同颜色高亮标记不同重复级别的句子。
  - 当用户点击某个高亮句子时，可以并排显示最相似的源句子及其出处（论文ID、链接等）。
参考竞品：https://app.copyleaks.com/dashboard/v1/account/scans/f25tskhrr1yrv7oe/report?viewMode=one-to-many&contentMode=html&sourcePage=1&suspectPage=1&showAIPhrases=false

---

5. 系统 API 设计

系统将通过 RESTful API 提供服务，遵循无状态、资源导向的设计原则 。

5.1 异步查重接口

考虑到论文处理可能耗时较长，推荐采用异步处理模式。

- Endpoint: POST /v1/papers/check
- 功能: 提交一篇论文进行查重，并立即返回一个任务ID。
- Request Body (JSON):
{
  "content": "The full text of the paper...",
  "callback_url": "https://your-service.com/webhook/notify" // 可选，用于任务完成后回调通知
}
  
- Success Response (202 Accepted):
{
  "task_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
  "status": "pending",
  "message": "Your paper has been queued for checking. Use the task_id to poll for results."
}
5.2 结果查询接口

- Endpoint: GET /v1/reports/{task_id}
- 功能: 根据任务ID查询查重结果。
- Success Response (200 OK):
  - 如果任务仍在处理中:
{
  "task_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
  "status": "processing"
}
    
  - 如果任务已完成:
{
  "task_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
  "status": "completed",
  "report": {
    "overall_similarity_score": 0.235,
    "summary": {
      "total_sentences": 500,
      "identical_count": 20,
      "minor_changes_count": 45,
      "paraphrased_count": 58
    },
    "top_sources": [
      {
        "paper_id": "arXiv:2401.12345",
        "similarity_score": 0.152,
        "url": "https://arxiv.org/abs/2401.12345"
      },
      {
        "paper_id": "arXiv:2308.06789",
        "similarity_score": 0.083,
        "url": "https://arxiv.org/abs/2308.06789"
      }
    ],
    "sentence_details": [
      {
        "original_sentence": "This method achieves state-of-the-art results.",
        "status": "minor_changes",
        "similarity_score": 0.96,
        "matched_source": {
          "sentence": "Our approach obtains state-of-the-art performance.",
          "paper_id": "arXiv:2401.12345"
        }
      },
      {
        "original_sentence": "The experiment was conducted in a controlled environment.",
        "status": "original",
        "similarity_score": 0.55,
        "matched_source": null
      }
      // ... more sentences
    ]
  }
}

---
6. 系统性能评估与基准测试

为了科学地评估 Copyless 系统的性能并持续优化，需要建立一套标准的评测流程。

6.1 评估指标
- 准确性指标:
  - 查准率 (Precision): 系统检测出的重复句子中，真正是重复的比例。
  - 召回率 (Recall): 所有真实存在的重复句子中，被系统成功检测出的比例。
  - F1-Score: 查准率和召回率的调和平均值，是综合评价模型准确性的核心指标 。
这些指标需要在句子级别和文档级别上分别进行评估。
- 性能指标:
  - 延迟 (Latency): 从提交论文到返回结果所需的平均时间及 P95/P99 分位数时间 。
  - 吞吐量 (Throughput): 系统单位时间内（如每秒）能够处理的查重请求数量或句子数量 。
    
6.2 基准数据集
使用公开的、经过人工标注的剽窃检测数据集是评估和调优算法（特别是相似度判定阈值）的唯一科学方法。
- PAN 竞赛数据集: 这是学术界最权威和广泛使用的抄袭检测基准之一，例如 PAN-PC-11, PAN 2013, PAN 2014 等 。这些数据集包含了各种类型的抄袭，从简单的复制粘贴到复杂的释义改写，并提供了详细的句子级和文档级标注，非常适合用于评估我们系统的 Precision, Recall 和 F1-Score。
- 自定义数据集: 我们可以基于 arXiv 论文库，人工构建一个小的、高质量的标注数据集，用于快速验证和迭代。
  
通过在这些基准数据集上运行我们的系统，可以系统性地调整相似度计算模块的阈值，以在查准率和召回率之间找到最佳平衡点。
7. 下一步工作
公式 和方法的相似度判定问题  Benchmark怎么构建 以及  学习方法
