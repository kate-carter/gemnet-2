# Claude-Visualizations

Visualization suite for comparing LLM-derived concept feature networks to
human concept networks.  The goal is to identify the correlation method,
embedding dimensionality, and thresholding condition under which Gemini word
embedding networks most closely resemble human-generated networks, and to find
which network statistics best predict semantic diversity (SemD).

## Running

```bash
# Single script
python vis_method_comparison.py

# All figures at once
python run_all.py
```

Output is written to `vis_results/`
(located at `gemnet-2/gemini-2/vis_results/`).

---

## Scripts and output figures

### 1. `vis_correlation_landscape.py`

**Output:** `correlation_landscape_{Method}.png`, `concept_erank_heatmap_{Method}.png`
(one file per correlation method: Mahalanobis, Spearman, Pearson, Cosine)

Characterizes the geometry of the feature-correlation matrices before any
thresholding.

| Panel | Quantity |
|-------|----------|
| A | **Eigenvalue spectra.** Top-50 eigenvalues of each \(d \times d\) correlation matrix, averaged over concepts with SE bands. Log scale. |
| B | **Correlation-value distributions.** Violin plots of off-diagonal correlation coefficients per dimensionality. Pairwise two-sample KS tests annotated. |
| C | **Effective rank.** Shannon-entropy effective rank per concept at each dimensionality. Kruskal-Wallis and pairwise Mann-Whitney U tests annotated. |
| D | **Mean absolute correlation.** \(\overline{|r|} = \frac{1}{N_\text{pairs}} \sum_{i<j} |r_{ij}|\) per concept per dim. |
| E | **Sparsity vs threshold.** Fraction of zero entries in thresholded matrices at five representative thresholds, averaged across concepts. |
| F | **Heatmap** of a representative correlation matrix (lowest available dim). |

**Effective rank:**

\[
\text{erank}(\mathbf{R}) = \exp\!\Bigl(-\sum_i p_i \ln p_i\Bigr),
\quad
p_i = \frac{\lambda_i}{\sum_j \lambda_j},
\quad
\lambda_i > 0
\]

**Normalized effective rank** (erank heatmap):

\[
\text{erank}_\text{norm} = \frac{\text{erank}(\mathbf{R})}{d}
\]

where \(d\) is the embedding dimensionality.

---

### 2. `vis_semd_network.py`

**Output:** `semd_network_integration_{Method}.png`
(one file per method)

Integrates semantic diversity (SemD) with network and embedding properties.

| Panel | Content |
|-------|---------|
| A | Ranked bar chart of SemD values for the 15 concepts. |
| B | Scatter: SemD vs modularity at a representative threshold (\(t \approx 0.15\)) with Spearman \(\rho\) and \(p\). |
| C | **SemD-metric correlation landscape.** Spearman \(\rho(\text{SemD}, M_c(t))\) across concepts at each threshold, one trajectory per BCT metric. |
| D | Scatter: SemD vs mean absolute off-diagonal correlation \(\overline{|r|}\) at 256-d (current method). |
| E | Scatter: SemD vs human network density. |
| F | Scatter: SemD vs log BNC frequency. |

All scatter panels annotate Spearman \(\rho\) and \(p\)-value.

---

### 3. `vis_human_comparison.py`

**Output:** `human_vs_llm_comparison.png`, `human_vs_llm_metric_profiles.png`

Compares LLM feature networks to human concept networks at 256-d (Spearman).

| Panel | Content |
|-------|---------|
| A | **Normalized distance heatmap.** Per-concept, per-threshold NRMSE between LLM and human metrics. Stars mark the minimum-distance threshold per concept. |
| B | **Best-match threshold.** Horizontal bar chart of which threshold minimizes NRMSE for each concept. |
| C | **Human weight distributions.** Histograms of positive edge weights from the upper triangle of human adjacency matrices. Pairwise KS tests annotated. |
| D | **Degree CDFs.** Empirical cumulative distribution of node degree in human networks. Pairwise KS tests annotated. |
| E | **Network size.** Number of nodes in each human network vs the fixed LLM dimensionality. |

**Metric profiles figure:** Per-metric panels showing LLM trajectories with
human baselines.  Green-shaded threshold regions indicate \(p > 0.05\) on the
Wilcoxon signed-rank test.

---

### 4. `vis_method_comparison.py`

**Output:** `method_comparison_human.png`, `method_comparison_semd.png`,
`method_comparison_primary.png`

Cross-method comparison across four correlation measures (Cosine, Mahalanobis,
Pearson, Spearman), five dimensionalities (256–3072), and 18 thresholds.

#### Figure 1 — Human distance landscape (`method_comparison_human.png`)

Identifies the method-dim-threshold condition that produces LLM networks most
similar to human networks.

**NRMSE** with double weighting on modularity and core-periphery fit:

\[
\text{NRMSE}_{c,t} = \sqrt{
  \frac{\sum_m w_m \left(\frac{L_{c,t,m} - H_{c,m}}{|H_{c,m}|}\right)^2}
       {\sum_m w_m}
},
\quad
w_m = \begin{cases}
2 & m \in \{\text{modularity}, \text{corefit}\} \\
1 & \text{otherwise}
\end{cases}
\]

| Panel | Content |
|-------|---------|
| A | Heatmap: rows = method × dim, columns = threshold, color = \(\overline{\text{NRMSE}}\). Star at global minimum. |
| B | Boxplot of per-concept NRMSE at each method's optimal condition. Kruskal-Wallis and pairwise Mann-Whitney U tests. |
| C | Scatter of LLM vs human metric values at the global-best cell. Spearman \(\rho\) and \(p\). |
| D | Grouped bar: human mean vs LLM mean per metric at the best cell. Wilcoxon signed-rank \(p\) per bar pair. |
| E | Top-10 leaderboard of closest-to-human conditions. |

#### Figure 2 — SemD prediction landscape (`method_comparison_semd.png`)

\[
\rho_m(t) = \text{Spearman}\bigl(\{S_c\}_{c \in C},\, \{M_{c,t,m}\}_{c \in C}\bigr)
\]

| Panel | Content |
|-------|---------|
| A | Heatmap of \(\max_m |\rho_m|\). Hatching = \(p < 0.05\). Star at global best. |
| B | Bar chart of best achievable \(|\rho|\) per method, with winning metric, dim, and significance. |
| C | Scatter: SemD vs winning metric at global-best cell with per-concept labels, \(\rho\) and \(p\). |
| D | \(\rho(\text{SemD}, m)\) trajectories across thresholds at global-best method/dim. |
| E | Top-10 leaderboard. |

#### Figure 3 — Primary metrics (`method_comparison_primary.png`)

| Panel | Content |
|-------|---------|
| A | Heatmap of \(|\rho(\text{SemD}, \text{modularity})|\). Hatching = \(p < 0.05\). |
| B | Heatmap of \(|\rho(\text{SemD}, \text{corefit})|\). |
| C | Scatter: SemD vs modularity at best modularity cell. |
| D | Scatter: SemD vs corefit at best corefit cell. |
| E | Modularity \(\rho\) trajectories, one line per method (at best dim). |
| F | Corefit \(\rho\) trajectories, same layout. |

---

### 5. `vis_human_correlation.py`

**Output:** `human_correlation_by_method.png`

2 × 5 grid (metric × dimensionality) showing Spearman \(\rho\) between LLM and
human BCT values across the 15 concepts, as a function of threshold.  One line
per correlation method; filled markers indicate \(p < 0.05\) (vs null \(\rho=0\)).
Legend reports mean ± SD of \(\rho\) across thresholds.

Priority metrics **Modularity** and **Core-fit** occupy the two rows.

---

### 6. `vis_human_alignment.py`

**Output:** `human_alignment_{Method}.png`
(one file per method)

Tests whether percolation phase-transition features align with the thresholds at
which LLM networks best match the human networks (published BCT baselines).
LLM values are averaged across all five dimensionalities.  Distance is
scale-agnostic: both LLM and human values are z-scored across concepts before
computing \(|z_\text{LLM} - z_\text{Human}|\).

| Panel | Content |
|-------|---------|
| A | **Cross-concept Spearman ρ** (LLM vs human) for core-fit, modularity, clustering, and strength, plotted against threshold. Secondary axis: mean percolation rate of change \(-d(\text{GiantFraction})/d\theta\), showing where phase transitions occur. |
| B | **Core-fit z-distance heatmap.** Rows = concepts, columns = thresholds, color = \(|z_\text{LLM} - z_\text{Human}|\). ★ marks the best-fit threshold per concept; \(\vert\) marks the percolation midpoint. |
| C | Same as B for **Modularity**. |
| D | **Percolation landscape.** Giant-component fraction (concepts × thresholds) with best-fit threshold symbols for all six BCT metrics overlaid. |
| E | **Phase transition vs best-fit threshold.** Scatter of percolation midpoint against the threshold of minimum z-distance, one point per concept per metric. Spearman ρ and regression per metric. |
| F | **Steepness vs best achievable fit.** Scatter of logistic steepness \(|k|\) against minimum z-distance. |
| F₂ | Same as F excluding modularity, with linear regression overlaid. |
| G | **Multi-metric z-distance bar chart.** At each concept's best core-fit threshold, shows the z-scored distance to human for all six BCT metrics (mean ± SD across concepts). |
| H | **Best-fit threshold dot plot.** Per concept, individual metric best-fit thresholds (light markers) and group means ± SE for the dense (modularity/strength/diversity) and mid-range (core-fit/clustering/degree) groups. Mann-Whitney U between groups. |
| I | Same as H but x-axis is **edge retention (%)** rather than threshold. |

**Z-scored distance:**

\[
d_m(c, t) = \left| \frac{L_{c,t,m} - \bar{L}_{t,m}}{\sigma_{L,t,m}} -
                   \frac{H_{c,m} - \bar{H}_m}{\sigma_{H,m}} \right|
\]

**Logistic percolation fit** (used to locate midpoint and steepness):

\[
G(t) = \frac{L}{1 + e^{-k(t - t_0)}} + b
\]

where \(G(t)\) is the giant-component fraction at threshold \(t\), \(t_0\) is
the midpoint, and \(|k|\) is the steepness.

---

### 7. `vis_structural_drivers.py`

**Output:** `structural_drivers_{Method}.png`
(one file per method)

Identifies which structural features drive variation across concept networks,
and which BCT metrics best predict SemD.  BCT profiles are averaged across all
available dimensionalities.

| Panel | Content |
|-------|---------|
| A | **PCA of structural profiles.** Each concept as a point in BCT-metric space (mean across all thresholds × dims). PC loadings annotated in text box. |
| B | **Metric co-variation.** Spearman ρ matrix of all BCT metrics across concepts — shows which structural properties share a common axis. |
| C | **Edge weight distribution properties.** Skewness, kurtosis, Shannon entropy, and positive-fraction of the 256-d unthresholded correlation matrices, normalized and plotted per concept. |
| D | **BCT metrics as predictors of SemD.** Horizontal bar chart of Spearman ρ between each metric's concept-level mean and SemD. Modularity and core-fit highlighted in red; \(* p < 0.05\). |
| E | **Best predictor scatter.** Scatter plot of the strongest BCT predictor of SemD with a regression line and annotated ρ and \(p\). |
| F | **Structural clustering dendrogram.** Ward hierarchical clustering of concepts by their full BCT profiles. |

**Shannon edge-weight entropy:**

\[
H = -\sum_{k} p_k \log_2 p_k,
\quad p_k = \frac{\text{count in bin } k}{\text{total edges}}
\]

---

### 8. `vis_semd_predictors.py`

**Output:** `semd_predictors.png`

Exhaustive sweep of all 4 methods × 5 dimensionalities × 18 thresholds × 6
BCT metrics to identify the conditions under which network statistics best
predict semantic diversity.  Modularity and corefit are always shown and
highlighted.

| Panel | Content |
|-------|---------|
| A | **Peak ρ heatmap.** Rows = correlation method, columns = BCT metric. Each cell shows the peak \(\rho(\text{SemD}, M)\) found across all (dim, threshold) conditions for that method–metric pair. \(* = p < .05\) against null \(\rho = 0\). |
| B | **Forest plot for modularity & corefit.** Each method's best-condition ρ with the specific (dim, threshold) that produced it. Filled marker = \(p < .05\); p-values annotated with \(*/\!\!**/\!\!***\) notation. |
| C | **Threshold landscape — Modularity.** \(\rho(\text{SemD}, \text{modularity})\) vs threshold, one line per method. Shaded band = ± SE across dimensionalities. Filled dots = at least one dim is significant at that threshold. |
| D | **Threshold landscape — Corefit.** Same as C for core-periphery fit. |
| E | **Best-condition scatter.** The single strongest overall predictor (any method, dim, threshold, metric) plotted concept-by-concept with regression line. Spearman ρ and p annotated. |
| F | **All metrics ranked.** Horizontal bar chart of all 6 BCT metrics at their respective best conditions, sorted by \(|\rho|\). Red bars = priority metrics (modularity/corefit). Significance and p-values annotated per bar. |

**Sweep logic:**

For each (method, dim, threshold, metric) quadruplet, the Spearman rank
correlation between concept-level BCT values and SemD values is computed:

\[
\rho_{m,d,t,k} = \text{Spearman}\bigl(\{M_{c,d,t,k}\}_{c},\, \{S_c\}_{c}\bigr)
\]

where \(M_{c,d,t,k}\) is the BCT metric \(k\) for concept \(c\) at
dimensionality \(d\) and threshold \(t\), and \(S_c\) is the SemD value.
P-values are two-sided tests against null \(\rho = 0\).

---

### 9. `vis_network_gallery.py`

**Output:** `network_gallery_llm.png`, `network_gallery_human.png`,
`network_gallery_sidebyside.png`

Force-directed layouts (Kamada-Kawai) of thresholded LLM and human networks.
Node size ∝ weighted degree; edge width ∝ edge weight.

---

## BCT metric definitions

All metrics are computed by `bctpy` on weighted, thresholded adjacency matrices.

| Metric | Symbol | Definition |
|--------|--------|------------|
| **Modularity** | \(Q\) | Newman's modularity: \(Q = \frac{1}{2m}\sum_{ij}\bigl[A_{ij} - \frac{k_i k_j}{2m}\bigr]\delta(c_i, c_j)\). Higher \(Q\) = stronger community structure. |
| **Core-periphery fit** | \(C_\text{cp}\) | Borgatti-Everett correlation between observed adjacency and idealized core-periphery structure. Higher = clearer core-periphery organization. |
| **Clustering** | \(C_i\) | Weighted clustering coefficient: \(C_i = \frac{1}{s_i(k_i-1)}\sum_{j,h}(w_{ij}w_{ih}w_{jh})^{1/3}\), node-averaged. |
| **Strength** | \(s_i\) | \(s_i = \sum_j w_{ij}\), node-averaged. |
| **Degree** | \(k_i\) | \(k_i = \sum_j \mathbb{1}[w_{ij}>0]\), node-averaged. |
| **Diversity** | \(D_i\) | Shannon entropy of connection weights: \(D_i = -\sum_m \frac{w_{im}}{s_i}\ln\frac{w_{im}}{s_i}\), node-averaged. |

**Modularity and core-periphery fit are the primary metrics** in this analysis,
based on prior evidence of statistically significant prediction of SemD.

---

## Statistical tests

| Test | Usage | Null hypothesis |
|------|-------|-----------------|
| **Wilcoxon signed-rank** | Paired LLM vs human across 15 concepts at a fixed threshold | Median paired difference = 0 |
| **Kruskal-Wallis** | Compare distributions across >2 groups | All groups from the same distribution |
| **Mann-Whitney U** | Pairwise comparison between two groups | Two samples from the same distribution |
| **Kolmogorov-Smirnov (2-sample)** | Compare correlation-value or weight distributions | Two samples from the same continuous distribution |
| **Spearman rank correlation** | SemD vs BCT metric; LLM vs human metric values | No monotonic association (\(\rho = 0\)) |

Significance thresholds: \(* p < 0.05\), \(** p < 0.01\), \(*** p < 0.001\).

---

## Dependencies

- Python 3.9+
- numpy, pandas, matplotlib, scipy, networkx, openpyxl
- scikit-learn (optional; numpy fallback used if absent)
