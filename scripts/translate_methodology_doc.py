# -*- coding: utf-8 -*-
"""Translate methodology_selection_report.docx from Vietnamese to English.

Preserves 100% of formatting, styles, tables, layout, headers, footers, and diagrams.
"""

from __future__ import annotations

import io
import json
import re
import shutil
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
import docx

ROOT = Path(__file__).resolve().parents[1]
SRC_DOCX = ROOT / "reports" / "summary" / "week2_methodology" / "methodology_selection_report.docx"
OUT_DIR = ROOT / "reports" / "summary" / "week2_methodology"
OUT_DOCX = OUT_DIR / "methodology_selection_en.docx"

FIG_EN_DIR = ROOT / "reports" / "figures" / "en"
FIG5_EN = FIG_EN_DIR / "fig5_recall_by_rule.png"
FIG3_EN = FIG_EN_DIR / "fig3_user_depth.png"

# Paragraph translations
# For mixed paragraphs: (bold_prefix, regular_text)
# For uniform paragraphs: text
PARAGRAPH_TRANSLATIONS = {
    0: "Methodology Selection for Trip-Context Hotel Cluster Recommendation",
    1: "Analysis of LightGBM, AdaGIN, and SMLP4Rec on the Expedia Hotel Recommendations Dataset",
    5: "1. Problem and data properties",
    6: "1.1 Problem statement",
    7: (
        "A traveller searches with a destination, stay dates, party composition, channel, and device. "
        "The model ranks 100 anonymous hotel clusters so that the booked cluster appears as high as possible. "
        "Clusters carry no name, description, or attributes, acting as a proxy for room category or product package. "
        "Only booking events are used for training; clicks (92.03% of rows) are not used as positive labels."
    ),
    8: "1.2 Properties that determine method choice",
    10: (
        "Figure 1. Recall@5 of conditional frequency tables on the temporal test split. "
        "Conditioning on destination is the reference a learned model must exceed."
    ),
    12: "Figure 2. Bookings per user. The median user has two bookings.",
    13: "2. LightGBM — Light Gradient Boosting Machine",
    14: "2.1 Source",
    15: "2.2 Algorithmic principle",
    16: (
        "LightGBM predicts by accumulating multiple small decision trees; each tree corrects the residual errors of the preceding trees. "
        "Pipeline for a single search event:"
    ),
    17: (
        "Step 1 — Input. ",
        "A search event is a single row: destination, market, party composition, check-in date, package, channel, alongside handcrafted historical features."
    ),
    18: (
        "Step 2 — Initialisation. ",
        "The model starts from a simple prediction, identical for all search events across the 100 clusters."
    ),
    19: (
        "Step 3 — Add error-correcting trees. ",
        "At each iteration, the model determines where the current prediction errs, and then builds one tree per cluster to correct those errors. "
        "The tree partitions search events into groups via threshold queries (lead time > 30 days? destination belongs to which group?); each leaf carries a score adjustment value."
    ),
    20: (
        "Step 4 — Accumulate and rank. ",
        "The score of each cluster is the cumulative sum of adjustments across all constructed trees. "
        "The process repeats until accuracy on the validation set stops improving; the 100 clusters are then ranked by final score."
    ),
    21: (
        "Acceleration techniques. ",
        "Histograms (binning numerical values), leaf-wise growth (splitting the leaf that yields the largest loss reduction), GOSS (prioritising samples with larger gradients), and EFB (bundling mutually exclusive sparse columns) do not alter the workflow above; they make Step 3 substantially faster and lighter. "
        "Categorical features are split directly, requiring no one-hot encoding."
    ),
    22: "2.3 Mathematical foundations",
    23: (
        "Training is formulated as loss minimisation. Softmax converts the 100 scores into probabilities; cross-entropy is large when the probability of the booked cluster is low. "
        "The objective is to adjust scores so as to minimise this loss."
    ),
    24: (
        "The gradient indicates the direction of adjustment: for each search event, which cluster's score should increase, which should decrease, and by how much. "
        "The tree in each boosting round is constructed to fit this direction, so each tree represents a step down the loss slope — hence the name gradient boosting."
    ),
    25: (
        "The Hessian indicates the curvature of the loss, i.e., how rapidly the slope changes. Where curvature is sharp, the step size must be small to avoid overshooting the optimum; where the loss surface is flat, steps can be larger. "
        "LightGBM uses both: split points are selected to group search events with similar adjustment directions, and the leaf values are weighted by the curvature."
    ),
    26: (
        "A learning rate shrinks each step; caps on the number of leaves, minimum sample counts per leaf, and L2 regularisation on leaf values prevent the model from memorising the training data."
    ),
    27: "2.4 Strengths and suitable scenarios",
    28: (
        "Strong on tabular data mixing numerical and categorical fields; frequently represents a difficult baseline to surpass for deep learning models on this data modality."
    ),
    29: (
        "Direct handling of numerical fields and missing values (each split point learns a default branch for missing values); no bucketing required."
    ),
    30: "Fast training on CPU; few hyper-sensitive hyperparameters.",
    31: "Interpretability via feature importance and SHAP values.",
    32: "No user ID required; cold users are naturally ranked from trip context.",
    33: "2.5 Weaknesses and practical limitations",
    34: (
        "Implicit interactions. ",
        "Interactions are learned only via sequential split paths; destination × month × party composition requires deep trees, and deep branches fragment the data similarly to frequency tables (Figure 1)."
    ),
    35: (
        "High-cardinality fields. ",
        "Raw categorical features with 59,455 destinations easily overfit the long tail; alternative target encoding carries leakage risk unless strictly computed using events strictly prior to each observation."
    ),
    36: (
        "No shared representations. ",
        "Rare destinations cannot borrow statistical strength from similar destinations unless higher-level hierarchy fields (market, country) are manually engineered."
    ),
    37: (
        "Multi-class cost. ",
        "Each boosting round builds 100 trees; training time and model footprint scale with the number of rounds × 100."
    ),
    38: (
        "Feature engineering dependency. ",
        "Historical features (most recent cluster, historical frequency by cluster) must be manually engineered."
    ),
    39: (
        "Practical applicability: ",
        "High. GBDTs are ubiquitous for industrial tabular ranking and forecasting; the model is a tree ensemble with fast CPU inference and straightforward deployment. Operational overhead lies primarily in the feature pipeline: encodings and historical aggregates must be recomputed as data drifts."
    ),
    40: "2.6 Implementation requirements",
    41: "2.7 Suitability for Expedia",
    42: (
        "Overall: ",
        "medium–high. Strong and computationally inexpensive; limitations coincide precisely with the two decisive data properties — feature interactions and high-cardinality fields."
    ),
    43: "3. AdaGIN — Adaptive Graph Interaction Network",
    44: "3.1 Source",
    45: "3.2 Algorithmic principle",
    46: (
        "AdaGIN scores each (search event, candidate cluster) pair individually, and then sorts the 100 scores into a ranking. "
        "Pipeline for a single pair:"
    ),
    47: (
        "Step 1 — Embedding. ",
        "Each field (destination, market, party, month, candidate cluster, etc.) is mapped to a vector. Field values associated with similar booking behaviours are mapped close together in the embedding space."
    ),
    48: (
        "Step 2 — Construct a sample-specific graph (GFIM). ",
        "Each field acts as a node. The model measures the pairwise affinity between fields for the specific sample under consideration and retains only the strongest edges. "
        "Consequently, each trip archetype receives its own interaction topology: for summer family holidays, the destination–month–party edges may dominate; for short-lead travellers, the channel–lead time edge may become prominent."
    ),
    49: (
        "Step 3 — Message passing on the graph. ",
        "Each field node aggregates information from its connected neighbours. Iterating across multiple layers, each field progressively absorbs compositional information across multiple fields; high-order interactions emerge without explicit combinatorial enumeration."
    ),
    50: (
        "Step 4 — Read out interactions via three spaces (MFIM). ",
        "The graph output is read out through three branches: element-wise products of field pairs, inner products of field pairs, and the concatenated vector. "
        "Each branch produces a score; learned gates modulate each branch's contribution, and the sum forms the pair score."
    ),
    51: (
        "Step 5 — Pruning (NFS). ",
        "During training, branches whose gate values remain non-positive across a designated number of batches are permanently disabled, reducing model complexity."
    ),
    52: (
        "Step 6 — Ranking. ",
        "Iterate across all 100 candidate clusters and rank them by their predicted scores."
    ),
    53: "3.3 Mathematical foundations",
    54: (
        "The foundation stems from the principle of factorization machines: the interaction between two field values is measured by the inner product of their latent vectors, rather than a dedicated scalar parameter for each combination. "
        "Because each vector is updated across all samples containing that value, rare combinations (e.g., low-volume destination × month) still obtain reliable scores — in contrast to frequency tables, which demand explicit support for every combination."
    ),
    55: (
        "Graph representation extends this principle from pairs to higher-order groups of fields: multi-layer message passing corresponds to multi-field interactions, while retaining only strong edges filters out spurious interactions."
    ),
    56: (
        "During training, each pair's score passes through a sigmoid function into a booking probability; binary cross-entropy penalises deviations from ground truth (1 for the booked cluster, 0 for sampled negative clusters). "
        "The loss gradient directs error reduction and backpropagates through the entire network, jointly optimising embeddings, graph construction, and gating parameters."
    ),
    57: "3.4 Strengths and suitable scenarios",
    58: (
        "Explicit, high-order field interactions; aligns directly with the empirical finding that seasonality and party composition act primarily via destination."
    ),
    59: (
        "Sample-adaptive structure: different trip archetypes (solo short-lead, family summer, package) leverage distinct interaction topologies."
    ),
    60: (
        "Native embedding for high-cardinality categorical fields; hotel market acts as a shared neighbourhood node for rare destinations."
    ),
    61: "No user history required; cold users are ranked purely from context.",
    62: "Field-importance gates and learned adjacency provide partial interpretability into which context variables drive the ranking.",
    63: "Runs inside FuxiCTR, where DeepFM, DCNv2, and related family models are readily available under a unified data pipeline, configuration, and evaluation framework.",
    64: "3.5 Weaknesses and practical limitations",
    65: (
        "Formulation mismatch. ",
        "The model scores one (context, item) pair at a time. Ranking 100 clusters requires expanding each event into 100 rows: 240 million training rows before negative sampling, and 60 million scoring rows for the test set."
    ),
    66: (
        "Pointwise loss. ",
        "Binary cross-entropy does not optimise the relative ranking across the 100 candidates directly."
    ),
    67: (
        "Numerical fields. ",
        "Lead time, stay length, and distance must be bucketed or scalar-embedded, resulting in partial loss of threshold information."
    ),
    68: (
        "Evidence base. ",
        "Reported gains stem from CTR benchmarks, not travel data; no independent replication exists; performance margins between tuned CTR models are often narrow."
    ),
    69: (
        "Compute and tuning cost. ",
        "GPU recommended for training; more hyperparameters than tree models (number of graph layers, gating temperatures, NFS thresholds)."
    ),
    70: (
        "Codebase maturity. ",
        "Small repository with concise documentation; execution relies heavily on FuxiCTR conventions."
    ),
    71: (
        "Practical applicability: ",
        "medium–high. The feature-interaction model family (DeepFM, DCN) is ubiquitous at the ranking layer of industrial recommender systems; AdaGIN shares their input format and serving mechanics. No public industrial deployments of AdaGIN specifically are documented. With 100 candidates per search event, serving scoring cost remains modest; computational overhead concentrates heavily in training."
    ),
    72: "3.6 Implementation requirements",
    73: "3.7 Suitability for Expedia",
    74: (
        "Overall: ",
        "high. Simultaneously addresses field interactions, high-cardinality features, and cold users without architectural modification; limitations reside in the 100-row expansion cost and pointwise loss formulation."
    ),
    75: "3.8 Concrete application design",
    76: "4. SMLP4Rec — All-MLP Architecture for Sequential Recommendation",
    77: "4.1 Source",
    78: "4.2 Algorithmic principle",
    79: "SMLP4Rec predicts the next cluster from the user's prior sequence of bookings. Workflow:",
    80: (
        "Step 1 — Construct input tensor. ",
        "Each booking in the sequence is represented by the cluster embedding and auxiliary feature embeddings, structured into a 3D tensor: sequence position × feature dimension × hidden channel. Sequences are truncated or padded to a fixed length."
    ),
    81: (
        "Step 2 — Three-axis token mixing. ",
        "Three MLPs execute in parallel: along the sequence axis, allowing each booking to aggregate information from other bookings (replacing self-attention); along the feature axis, combining the cluster with its side attributes; and along the hidden channel axis, mixing representation dimensions. Outputs from the three branches are summed and iterated across multiple layers."
    ),
    82: (
        "Step 3 — User summarisation. ",
        "The representation vector at the final non-padded booking position serves as the user's current latent preference."
    ),
    83: (
        "Step 4 — Ranking. ",
        "This summary vector is scored against embeddings of all 100 clusters; clusters are sorted by alignment score."
    ),
    84: "4.3 Mathematical foundations",
    85: (
        "An MLP is a sequence of linear transformations interleaved with non-linear activation functions. Operating along an axis means each output element is a learned weighted combination of all elements along that axis; hence the final booking absorbs context from prior bookings. "
        "Unlike self-attention, mixing weights are fixed per position rather than computed dynamically from content — reducing computational overhead, but mandating a fixed sequence length."
    ),
    86: (
        "During training, alignment scores between the user vector and the 100 clusters pass through softmax into probabilities; cross-entropy penalises low probability on the booked cluster. "
        "Over a closed set of 100 clusters, loss is computed over the full item vocabulary without requiring negative sampling. Backpropagating gradients jointly update embeddings and all three mixing MLPs to minimise loss."
    ),
    87: "4.4 Strengths and suitable scenarios",
    88: "Combines sequential booking history with auxiliary features; offers public author implementation.",
    89: "Full softmax over 100 clusters matches the ranking formulation directly; obviates negative sampling.",
    90: "Lightweight architecture; CPU training is viable for short sequences.",
    91: "Runs within RecBole, where GRU4Rec and SASRec are available under the exact same pipeline.",
    92: "4.5 Weaknesses and practical limitations",
    93: (
        "No current context. ",
        "The vanilla architecture predicts solely from historical sequences; it does not observe the current search destination, which represents the single strongest signal."
    ),
    94: (
        "Item-level side features only. ",
        "Auxiliary features are looked up strictly by item ID; per-booking trip context (destination, month, party) cannot be ingested without code modifications."
    ),
    95: (
        "Short sequences. ",
        "With a median history of two bookings, the majority of the fixed-length sequence window consists of padding; the sequence mixer finds limited sequential order to learn [6]."
    ),
    96: (
        "Cold users. ",
        "Users without history have no sequence, requiring a fallback mechanism."
    ),
    97: (
        "Code quality. ",
        "Hard-coded sequence length of 50; a single mixing block reused across layers without residual connections; lack of padding masks; superfluous classes; dependency on RecBole 1.0."
    ),
    98: (
        "Licensing. ",
        "No licence file found in the repository; requires verification before porting code into the project repository."
    ),
    99: (
        "Practical applicability: ",
        "low in this problem setting. The architecture is engineered for inference efficiency in large-scale sequential recommendation; utility degrades in low-frequency domains such as travel bookings. Codebase remains at research prototype quality, requiring refactoring prior to deployment."
    ),
    100: "4.6 Implementation requirements",
    101: "4.7 Suitability for Expedia and mandatory adaptations",
    102: (
        "As published, SMLP4Rec exhibits low suitability: it ignores current trip context and cannot handle cold users. "
        "Two adaptations are necessary to operationalise it on Expedia: (1) positional context features assigned to each historical booking; (2) a query token representing the current search event containing its context, from which candidate ranking is computed. "
        "With the query token, cold users are scored purely from search context."
    ),
    103: "5. Comparative evaluation",
    104: "5.1 Mechanisms and requirements",
    105: "5.2 Mapping against core requirements",
    106: "6. Selection and rationale",
    107: "6.1 Decision",
    108: (
        "AdaGIN is selected as the method for the task. Among the three evaluated candidates, AdaGIN is the sole method that simultaneously satisfies high-order context field interactions, high-cardinality categorical features, and cold-user ranking without structural architectural modifications (Section 5.2)."
    ),
    109: "6.2 Evidence-based rationale",
    110: "6.3 Risks and mitigation strategies",
    111: "6.4 Validation benchmarks (Recall@5, test split)",
    112: "7. Limitations",
    113: "Hotel clusters are anonymous proxies for room categories or packages; business conclusions inherit this proxy limitation.",
    114: "Empirical evidence for AdaGIN derives from CTR benchmarks; performance on travel booking datasets remains unproven.",
    115: "The Expedia competition leaderboard was compromised by a leakage path via user location and orig–destination distance; published scores on this dataset are not used as targets, and that lookup match is never constructed.",
    116: "Dataset terms permit research use only; raw data is not redistributed.",
    117: "Hardware requirements and training time for AdaGIN (Section 3.6) are qualitative estimates; exact empirical benchmarks are recorded in Phase 2.",
    118: "References",
    119: "[1] L. Sang, H. Li, Y. Zhang, Y. Zhang, Y. Yang. AdaGIN: Adaptive Graph Interaction Network for Click-Through Rate Prediction. ACM TOIS, 2024. doi:10.1145/3681785. Code: github.com/salmon1802/AdaGIN",
    120: "[2] J. Gao, X. Zhao, M. Li, M. Zhao, R. Wu, R. Guo, Y. Liu, D. Yin. SMLP4Rec: An Efficient All-MLP Architecture for Sequential Recommendations. ACM TOIS, 2024. doi:10.1145/3637871. Code: github.com/Applied-Machine-Learning-Lab/SMLP4Rec",
    121: "[3] J. Zhu et al. BARS: Towards Open Benchmarking for Recommender Systems. SIGIR, 2022. FuxiCTR: github.com/reczoo/FuxiCTR",
    122: "[4] W. X. Zhao et al. RecBole: A Unified, Comprehensive and Efficient Framework for Recommendation. github.com/RUCAIBox/RecBole",
    123: "[5] G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, T.-Y. Liu. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS, 2017",
    124: "[6] Does It Look Sequential? An Analysis of Datasets for Evaluation of Sequential Recommendations. RecSys, 2024. doi:10.1145/3640457.3688195",
    125: "[7] Y. Fan, Y. Ji, J. Zhang, A. Sun. Our Model Achieves Excellent Performance on MovieLens: What Does It Mean? ACM TOIS, 2024. doi:10.1145/3675163",
}

# Table translations: list of rows, each row is a list of cell texts
TABLE_TRANSLATIONS = {
    0: [
        ["Item", "Content"],
        ["Task", "Rank 100 hotel clusters for a search event, given trip context and the user's prior bookings where available"],
        ["Dataset", "Expedia Hotel Recommendations: 3,000,693 bookings, 813,985 booking users, 2013-01-07 to 2014-12-31"],
        ["Protocol", "Global temporal split [7] (2,400,554 train / 600,139 test bookings); full ranking over 100 clusters; Recall@K, NDCG@K, MRR, K = 5, 10, 20"],
        ["Selection", "AdaGIN — Adaptive Graph Interaction Network [1]"],
        ["Date", "23 September 2026"],
    ],
    1: [
        ["Property", "Measured value", "Requirement on the method"],
        ["Context dominates", "Mean Jensen–Shannon divergence of the cluster distribution: destination 0.4076; market 0.3953; package 0.0386; party 0.0139; check-in month 0.0019 bits", "Consume trip context directly"],
        ["Signal lies in interactions", "Recall@5: destination 53.07%; destination × check-in month 46.55%; destination × party 51.59%", "Learn feature interactions without count fragmentation"],
        ["High-cardinality destination", "59,455 destination values; 36,933 with a booking; 10,789 with exactly one booking; market fallback (2,118 values) covers 99.998% of test", "Embedding or thresholded encoding with fallback"],
        ["Short user history", "Median 2 bookings per user; 38.78% have one booking; 40.57% have three or more; 21.33% of bookings repeat a prior cluster", "Sequence mechanism has limited signal"],
        ["Cold users", "30.14% of test events belong to users unseen in training", "Rank from context without user history"],
        ["Non-stationarity", "Package share 21.78% (Jan 2014) to 9.42% (Dec 2014); volume +92.9% from 2013 to 2014", "Validate on the quarter before test; temporal features"],
    ],
    2: [
        ["Item", "Detail"],
        ["Paper", "G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, T.-Y. Liu. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS, 2017 [5]"],
        ["Code", "github.com/microsoft/LightGBM, MIT; mature library, Python API"],
        ["Formulation", "Multi-class classification over 100 clusters, one row per event, softmax cross-entropy loss"],
    ],
    3: [
        ["Suitable", "Unsuitable"],
        ["Tabular data, medium to large sample sizes", "Extremely high-cardinality categorical fields with long tails"],
        ["Numerical features with clear thresholds (lead time, stay length)", "Signal primarily driven by high-order interactions among high-cardinality fields"],
        ["Need for fast, easily deployable, interpretable baseline", "Need for shared representations across rare entities; sequential data"],
    ],
    4: [
        ["Item", "Requirement"],
        ["Input data", "One row per booking event; numerical and categorical context columns; label is cluster ID (0–99)"],
        ["Preprocessing", "Thresholded target encoding for destination, computed strictly chronologically on training split; history features derived strictly from bookings prior to current event"],
        ["Scale", "2,400,554 train rows, 600,139 test rows"],
        ["Hardware", "Multi-core CPU, no GPU required; RAM and training time benchmarked in Phase 2"],
        ["Software", "Python lightgbm library"],
    ],
    5: [
        ["Expedia property", "LightGBM response", "Fit"],
        ["Context dominates; seasonality and party act via interaction", "Learned through split paths; high-order interactions require deep trees", "Medium"],
        ["Destination: 59,455 values, long singleton tail", "Thresholded target encoding with fallback to market; leakage risk", "Medium"],
        ["30.14% test users are cold", "No user ID used; ranks from context", "High"],
        ["Closed candidate set of 100 clusters", "Multi-class with 100 classes, one row per event; 100 trees per round", "High"],
        ["Numerical context, 33.83% missing distance", "Handled natively, learns default path for missing values", "High"],
        ["Short history (median 2)", "Handcrafted historical summary features", "Medium"],
    ],
    6: [
        ["Item", "Detail"],
        ["Paper", "L. Sang, H. Li, Y. Zhang, Y. Zhang, Y. Yang. AdaGIN: Adaptive Graph Interaction Network for Click-Through Rate Prediction. ACM Transactions on Information Systems, 2024 [1]"],
        ["Code", "github.com/salmon1802/AdaGIN (second author), Apache-2.0, built on FuxiCTR (PyTorch) [3]"],
        ["Benchmarks in repo", "Criteo, Avazu, Frappe (context-aware app usage), MovieLens-1M"],
        ["Formulation", "Pointwise binary classification per (context, item) row, trained with binary cross-entropy"],
    ],
    7: [
        ["Suitable", "Unsuitable"],
        ["Many categorical fields, high cardinality", "Small datasets; embeddings and graph layers require abundant samples"],
        ["Signal concentrated in interactions among fields", "Signal primarily driven by continuous numerical features"],
        ["Large-scale data, GPU available", "Signal primarily driven by sequential order of behavioural history"],
    ],
    8: [
        ["Item", "Requirement"],
        ["Input data", "One row per (event, candidate cluster); all fields in categorical or bucketed form; FuxiCTR format"],
        ["Preprocessing", "OOV token based on minimum occurrence threshold; bucketing for numerical fields with missing indicator; in-event negative sampling"],
        ["Scale", "240 million training rows before negative sampling; 60 million scoring rows on test"],
        ["Hardware", "GPU recommended for training; batched test scoring; VRAM, RAM, and runtime benchmarked in Phase 2"],
        ["Software", "PyTorch, FuxiCTR, author's codebase (Apache-2.0)"],
    ],
    9: [
        ["Expedia property", "AdaGIN response", "Fit"],
        ["Context dominates; seasonality and party act via interaction", "GFIM and MFIM explicitly model field interactions, including cluster × destination × month", "High"],
        ["Destination: 59,455 values, long singleton tail", "Embedding with frequency-thresholded OOV token; market shares signal via graph topology", "High"],
        ["30.14% test users are cold", "Omits user ID; ranks from context and optional history field", "High"],
        ["Closed candidate set of 100 clusters", "Cluster is treated as a field; scoring 100 rows per event remains feasible", "Medium (expansion cost)"],
        ["Numerical context, 33.83% missing distance", "Bucketing accompanied by missing indicator field", "Medium"],
        ["Short history (median 2)", "Previously booked clusters ingested as an unordered history field", "Medium"],
    ],
    10: [
        ["Component", "Specification"],
        ["Row unit", "One row per (search event, candidate cluster); label 1 for booked cluster"],
        ["Training rows", "All positives plus sampled negative clusters per event (sample size finalised on validation)"],
        ["Evaluation", "Score all 100 candidate clusters per test event; full ranking without sampled metrics"],
        ["Fields", "Party composition (adults, children, room count, party type), lead time, stay duration, check-in month and day of week, package flag, channel, mobile flag, site ID, distance bucket and missing indicator, destination ID, destination type, market, country, continent, user location, candidate cluster ID"],
        ["Cluster-side features", "Cluster share within destination and within market, computed strictly on training split"],
        ["History field", "Clusters booked prior to current event (unordered multi-hot); empty for cold users"],
        ["Validation", "Quarter immediately preceding the test window; minimum of three random seeds"],
        ["Ablation", "Sequentially remove context field groups; run with and without NFS pruning"],
    ],
    11: [
        ["Item", "Detail"],
        ["Paper", "J. Gao, X. Zhao, M. Li, M. Zhao, R. Wu, R. Guo, Y. Liu, D. Yin. SMLP4Rec: An Efficient All-MLP Architecture for Sequential Recommendations. ACM Transactions on Information Systems, 2024 [2]"],
        ["Code", "github.com/Applied-Machine-Learning-Lab/SMLP4Rec, single model file on RecBole 1.0 [4]; no licence file observed"],
        ["Formulation", "Next-item prediction from an ordered sequence of user items; supports full softmax loss"],
    ],
    12: [
        ["Suitable", "Unsuitable"],
        ["Long histories where order carries strong signal (e-commerce, streaming)", "Short histories, low-frequency purchases"],
        ["Items possess rich side attributes", "Anonymous items without attributes"],
        ["Frequent repeat visitors", "High cold-user ratio"],
    ],
    13: [
        ["Item", "Requirement"],
        ["Input data", "Chronologically ordered booking sequence per user (RecBole atomic files); auxiliary features looked up by item ID; post-adaptation: positional context features"],
        ["Preprocessing", "Truncate or pad sequence to fixed length; user requires at least one prior booking, unless using query token"],
        ["Scale", "One sequence prefix per training booking; 100-class output layer"],
        ["Hardware", "CPU feasible for short sequences; runtime benchmarked in Phase 2"],
        ["Software", "PyTorch, RecBole 1.0; code licence unconfirmed"],
    ],
    14: [
        ["Expedia property", "SMLP4Rec response", "Fit"],
        ["Context dominates", "Vanilla model ignores current-event context; requires query token", "Low (vanilla)"],
        ["Destination: 59,455 values", "Ingests item-level features only; requires modification for per-booking context", "Low (vanilla)"],
        ["30.14% test users are cold", "No sequence available; handled only via query token", "Low (vanilla)"],
        ["Closed candidate set of 100 clusters", "Full softmax over 100 clusters", "High"],
        ["Short history (median 2)", "Majority of fixed-length window is padding", "Low"],
    ],
    15: [
        ["Criterion", "LightGBM", "AdaGIN", "SMLP4Rec"],
        ["Codebase", "Mature library", "Author code, FuxiCTR, Apache-2.0", "Author code, RecBole 1.0, unconfirmed licence"],
        ["Utilised signal", "Current context and handcrafted history features", "Current trip context", "Ordered booking history (+ context post-adaptation)"],
        ["Row unit", "One row per event (multi-class)", "100 rows per event", "One sequence prefix per booking"],
        ["Interactions", "Implicit via tree branches", "Explicit, sample-adaptive", "Mixed along feature axis"],
        ["Destination (59,455 values)", "Thresholded target encoding", "Embedding", "Embedding (post-adaptation)"],
        ["Numerical features", "Handled directly", "Bucketed", "Bucketed"],
        ["Cold users", "Handled natively", "Handled natively", "Only with query token"],
        ["Loss vs. target metric", "Full softmax", "Pointwise BCE", "Full softmax"],
        ["Interpretability", "SHAP, ablation", "Gates, adjacency matrix, ablation", "Ablation only"],
        ["Compute", "CPU", "GPU recommended", "CPU feasible"],
        ["Primary risk", "Leakage via target encoding", "Expansion cost; narrow margin over DCNv2", "Short sequences; code modifications required"],
        ["Suitable scenario", "Tabular data, numerical features", "High-cardinality categoricals, signal in interactions", "Long histories, attributed items"],
        ["Practical applicability", "High", "Medium–high", "Low"],
        ["Fit for Expedia", "Medium–high", "High", "Low (vanilla); Medium (adapted)"],
    ],
    16: [
        ["Requirement", "LightGBM", "AdaGIN", "SMLP4Rec"],
        ["Public reference implementation", "Satisfied", "Satisfied", "Satisfied; licence unclear"],
        ["Interaction among context fields without manual enumeration", "Partial (implicit via split paths)", "Satisfied (explicit, sample-adaptive)", "Partial (only post-adaptation)"],
        ["High-cardinality fields (59,455 destinations)", "Partial (target encoding)", "Satisfied (embedding, OOV)", "Unsatisfied (vanilla)"],
        ["Cold-user ranking", "Satisfied", "Satisfied", "Only with query token"],
        ["Feasible within 2–3 weeks on one workstation", "Satisfied", "Satisfied; requires GPU, negative sampling", "Satisfied after code refactoring"],
    ],
    17: [
        ["Empirical evidence", "Resulting requirement", "Corresponding AdaGIN mechanism"],
        ["Destination JSD 0.4076; market 0.3953", "Rank from trip context", "All inputs are context fields"],
        ["Recall@5: destination × month 46.55% < destination 53.07%", "Learn interactions without count fragmentation", "Interactions via shared embeddings, not count combinations"],
        ["59,455 destinations; 10,789 with single booking", "Share statistical strength for rare entities", "Embedding with OOV; market acts as shared neighbour node"],
        ["30.14% test events from cold users", "No dependency on history", "Omits user ID"],
        ["Median 2 bookings per user", "Sequential order carries limited signal", "History is an optional, unordered field"],
    ],
    18: [
        ["Risk", "Mitigation strategy"],
        ["Expansion cost of 100 rows per event", "Negative sampling during training, sample size finalised on validation; batched test scoring; pilot on user subsample before full run"],
        ["Pointwise BCE does not directly optimise rank order", "Model checkpoint selection by validation NDCG; optional softmax-over-100-rows variant, reported explicitly as an adaptation"],
        ["Narrow margin over DeepFM, DCNv2", "Evaluated in same FuxiCTR framework, identical tuning budget, minimum three seeds, reporting mean ± std"],
        ["Original evidence limited to CTR benchmarks", "Benchmarked against 53.07% Recall@5 destination frequency reference; revision conditions defined in Section 6.5"],
        ["Small codebase, concise documentation", "Pin FuxiCTR and author repository commit versions; log full reproducible configuration per run"],
    ],
    19: [
        ["Benchmark reference", "Recall@5", "Role"],
        ["Global frequency", "14.06%", "Below this: implementation error"],
        ["Repeat most recent cluster", "16.82%", "Ceiling of repeat-booking heuristic alone"],
        ["Destination × check-in month frequency", "46.55%", "Fragmentation loss to be recovered"],
        ["Destination frequency", "53.07%", "Threshold a learned machine learning model must exceed"],
    ],
}


TOC_TRANSLATIONS = {
    "Nội dung": "Contents",
    "Nội dung": "Contents",
    "1. Bài toán và đặc tính dữ liệu": "1. Problem and data properties",
    "1.1 Bài toán": "1.1 Problem statement",
    "1.2 Đặc tính quyết định lựa chọn phương pháp": "1.2 Properties that determine method choice",
    "2. LightGBM — Light Gradient Boosting Machine": "2. LightGBM — Light Gradient Boosting Machine",
    "2.1 Nguồn gốc": "2.1 Source",
    "2.2 Ý tưởng thuật toán": "2.2 Algorithmic principle",
    "2.3 Nền tảng toán học": "2.3 Mathematical foundations",
    "2.4 Điểm mạnh và kịch bản phù hợp": "2.4 Strengths and suitable scenarios",
    "2.5 Điểm yếu và giới hạn thực tiễn": "2.5 Weaknesses and practical limitations",
    "2.6 Yêu cầu triển khai": "2.6 Implementation requirements",
    "2.7 Mức phù hợp với Expedia": "2.7 Suitability for Expedia",
    "3. AdaGIN — Adaptive Graph Interaction Network": "3. AdaGIN — Adaptive Graph Interaction Network",
    "3.1 Nguồn gốc": "3.1 Source",
    "3.2 Ý tưởng thuật toán": "3.2 Algorithmic principle",
    "3.3 Nền tảng toán học": "3.3 Mathematical foundations",
    "3.4 Điểm mạnh và kịch bản phù hợp": "3.4 Strengths and suitable scenarios",
    "3.5 Điểm yếu và giới hạn thực tiễn": "3.5 Weaknesses and practical limitations",
    "3.6 Yêu cầu triển khai": "3.6 Implementation requirements",
    "3.7 Mức phù hợp với Expedia": "3.7 Suitability for Expedia",
    "3.8 Thiết kế áp dụng": "3.8 Concrete application design",
    "4. SMLP4Rec — All-MLP Architecture for Sequential Recommendation": "4. SMLP4Rec — All-MLP Architecture for Sequential Recommendation",
    "4.1 Nguồn gốc": "4.1 Source",
    "4.2 Ý tưởng thuật toán": "4.2 Algorithmic principle",
    "4.3 Nền tảng toán học": "4.3 Mathematical foundations",
    "4.4 Điểm mạnh và kịch bản phù hợp": "4.4 Strengths and suitable scenarios",
    "4.5 Điểm yếu và giới hạn thực tiễn": "4.5 Weaknesses and practical limitations",
    "4.6 Yêu cầu triển khai": "4.6 Implementation requirements",
    "4.7 Mức phù hợp với Expedia và điều chỉnh bắt buộc": "4.7 Suitability for Expedia and mandatory adaptations",
    "5. So sánh": "5. Comparative evaluation",
    "5.1 Cơ chế và yêu cầu": "5.1 Mechanisms and requirements",
    "5.2 Đối chiếu yêu cầu cốt lõi": "5.2 Mapping against core requirements",
    "6. Lựa chọn và lập luận": "6. Selection and rationale",
    "6.1 Quyết định": "6.1 Decision",
    "6.2 Lập luận từ dữ liệu": "6.2 Evidence-based rationale",
    "6.3 Rủi ro và biện pháp giảm thiểu": "6.3 Risks and mitigation strategies",
    "6.4 Mốc kiểm định (Recall@5, tập test)": "6.4 Validation benchmarks (Recall@5, test split)",
    "7. Giới hạn": "7. Limitations",
    "Tài liệu tham khảo": "References",
}


def build_en_docx():
    print("Opening source docx:", SRC_DOCX)
    doc = docx.Document(str(SRC_DOCX))

    # Translate paragraphs
    for p_idx, p in enumerate(doc.paragraphs):
        if p_idx not in PARAGRAPH_TRANSLATIONS:
            continue
        item = PARAGRAPH_TRANSLATIONS[p_idx]
        if isinstance(item, tuple):
            bold_prefix, regular_text = item
            # Mixed paragraph
            # run 0: bold prefix
            if len(p.runs) > 0:
                p.runs[0].text = bold_prefix
                p.runs[0].bold = True
            if len(p.runs) > 1:
                p.runs[1].text = regular_text
                p.runs[1].bold = False
                for r in p.runs[2:]:
                    r.text = ""
            else:
                r1 = p.add_run(regular_text)
                r1.bold = False
        else:
            # Uniform paragraph
            text = item
            if len(p.runs) > 0:
                p.runs[0].text = text
                for r in p.runs[1:]:
                    r.text = ""
            else:
                p.text = text

    # Translate tables
    for t_idx, table in enumerate(doc.tables):
        if t_idx not in TABLE_TRANSLATIONS:
            print(f"Warning: Table {t_idx} not in translations!")
            continue
        expected_rows = TABLE_TRANSLATIONS[t_idx]
        for r_idx, row in enumerate(table.rows):
            if r_idx >= len(expected_rows):
                print(f"Warning: Row {r_idx} exceeds expected rows in Table {t_idx}")
                continue
            expected_cells = expected_rows[r_idx]
            for c_idx, cell in enumerate(row.cells):
                if c_idx >= len(expected_cells):
                    continue
                cell_text = expected_cells[c_idx]
                if cell.paragraphs and len(cell.paragraphs[0].runs) > 0:
                    cell.paragraphs[0].runs[0].text = cell_text
                    for r in cell.paragraphs[0].runs[1:]:
                        r.text = ""
                    # If multiple paragraphs in cell, clear additional paragraphs
                    for extra_p in cell.paragraphs[1:]:
                        for r in extra_p.runs:
                            r.text = ""
                else:
                    cell.text = cell_text

    # Save to intermediate path
    temp_docx = OUT_DIR / "temp_translated.docx"
    doc.save(str(temp_docx))
    print("Saved intermediate docx to:", temp_docx)

    # Post-process zip package:
    # 1. Replace image1.png with fig5_recall_by_rule.png (EN)
    # 2. Replace image2.png with fig3_user_depth.png (EN)
    # 3. Update footer1.xml: replace ' C1 — Lựa chọn phương pháp  ·  ' with ' C1 — Methodology Selection  ·  '
    # 4. Update image extent cy in document.xml for aspect ratio accuracy
    # 5. Update TOC strings inside <w:sdt> in document.xml
    with open(FIG5_EN, "rb") as f:
        fig5_bytes = f.read()
    with open(FIG3_EN, "rb") as f:
        fig3_bytes = f.read()

    buf = io.BytesIO()
    with zipfile.ZipFile(temp_docx, "r") as zin:
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                content = zin.read(item.filename)
                if item.filename == "word/media/image1.png":
                    zout.writestr(item, fig5_bytes)
                elif item.filename == "word/media/image2.png":
                    zout.writestr(item, fig3_bytes)
                elif item.filename == "word/footer1.xml":
                    text_xml = content.decode("utf-8")
                    text_xml = text_xml.replace(
                        "C1 — Lựa chọn phương pháp",
                        "C1 — Methodology Selection"
                    )
                    zout.writestr(item, text_xml.encode("utf-8"))
                elif item.filename == "word/document.xml":
                    text_xml = content.decode("utf-8")
                    # Adjust extent cy for image1 (P9) and image2 (P11)
                    text_xml = text_xml.replace(
                        'cx="5715000" cy="2790825"',
                        'cx="5715000" cy="2423377"'
                    )
                    text_xml = text_xml.replace(
                        'cx="5715000" cy="3124200"',
                        'cx="5715000" cy="3091488"'
                    )
                    # Translate TOC entries
                    for vn_str, en_str in TOC_TRANSLATIONS.items():
                        text_xml = text_xml.replace(f">{vn_str}<", f">{en_str}<")
                    zout.writestr(item, text_xml.encode("utf-8"))
                else:
                    zout.writestr(item, content)

    # Write final docx to OUT_DOCX
    final_bytes = buf.getvalue()
    with open(OUT_DOCX, "wb") as f:
        f.write(final_bytes)
    print("Wrote final docx to:", OUT_DOCX)

    # Clean up temp
    if temp_docx.exists():
        temp_docx.unlink()


if __name__ == "__main__":
    build_en_docx()
