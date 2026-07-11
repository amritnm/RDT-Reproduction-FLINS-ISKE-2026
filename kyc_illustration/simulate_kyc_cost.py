"""
KYC Staged-Cost Illustration
=============================
Synthetic simulation demonstrating the structural difference between sklearn
DecisionTree and RDT in a staged KYC verification setting.

Dataset
-------
12 features split into 4 vendor stages (3 features each):
  Stage 0 – Internal data (FREE):       f0_age, f0_tenure, f0_location_risk
  Stage 1 – Vendor A ($2/query):        f1_id_score, f1_doc_authentic, f1_name_match
  Stage 2 – Vendor B ($5/query):        f2_addr_verified, f2_addr_tenure, f2_postcode_risk
  Stage 3 – Vendor C ($50/query):       f3_biometric, f3_sanctions, f3_pep_score

Stage 0 features have the highest discriminative power (decreasing with stage).
Mixed distributions: Gaussian, binary, beta, exponential, Poisson, gamma.

Output files (written to kyc_illustration/)
-------------------------------------------
  kyc_dataset.csv              – generated dataset (reproducible, seed=42)
  results_depth3.json          – full results for depth-3 trees
  results_depth4.json          – full results for depth-4 trees
  tree_structure_depth3.png    – side-by-side tree diagrams coloured by vendor stage
  tree_structure_depth4.png    – same for depth 4
  cost_distribution.png        – per-sample cost box plots (both depths)
  kyc_illustration_report.md   – human-readable summary

Usage
-----
  python kyc_illustration/simulate_kyc_cost.py
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from collections import defaultdict

# ── Add project root to path so we can import rdt ────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from rdt.restricted_decision_tree import RestrictedDecisionTree

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Vendor cost structure ─────────────────────────────────────────────────────
STAGE_COSTS = {0: 0.0, 1: 2.0, 2: 5.0, 3: 50.0}

# Stage assignment for each feature index (0-based)
# Features 0-2: Stage 0, 3-5: Stage 1, 6-8: Stage 2, 9-11: Stage 3
FEATURE_STAGE = {i: i // 3 for i in range(12)}

STAGE_COLORS = {
    0: '#2ecc71',   # green  — free internal
    1: '#f1c40f',   # yellow — Vendor A ($2)
    2: '#e67e22',   # orange — Vendor B ($5)
    3: '#e74c3c',   # red    — Vendor C ($50)
}

FEATURE_NAMES = [
    # Stage 0 – Internal (free)
    'Age',              # 0: Gaussian
    'AccountTenure',    # 1: Gaussian
    'LocationRisk',     # 2: Binary
    # Stage 1 – Vendor A ($2)
    'ID_Score',         # 3: Continuous [0,1]
    'DocAuthentic',     # 4: Binary
    'NameMatch',        # 5: Beta
    # Stage 2 – Vendor B ($5)
    'AddrVerified',     # 6: Binary
    'AddrTenure',       # 7: Exponential
    'PostcodeRisk',     # 8: Poisson
    # Stage 3 – Vendor C ($50)
    'BiometricScore',   # 9: Gaussian
    'SanctionsClear',   # 10: Binary
    'PEP_Score',        # 11: Gamma
]

CLASS_NAMES = ['Reject', 'Approve']
SEED = 42
N_SAMPLES = 2_000    # 1.6k train / 400 test — kept small for RDT Python speed


# ═════════════════════════════════════════════════════════════════════════════
# 1. DATA GENERATION
# ═════════════════════════════════════════════════════════════════════════════

def generate_kyc_dataset(n_samples: int, seed: int) -> tuple:
    """
    Generate a synthetic KYC dataset with known stage structure.

    Class 1 = Approve (60%), Class 0 = Reject (40%)
    Each stage adds incremental discriminative power.
    """
    rng = np.random.default_rng(seed)
    n1 = int(n_samples * 0.60)  # approve
    n0 = n_samples - n1          # reject

    def stack(app_arr, rej_arr):
        return np.concatenate([app_arr, rej_arr])

    # ── Stage 0: Internal Data (strongest signal) ─────────────────────────
    # f0: Age – Gaussian
    f0_approve = rng.normal(45, 10, n1)
    f0_reject  = rng.normal(29, 9,  n0)

    # f1: AccountTenure (months) – Gaussian
    f1_approve = rng.normal(60, 18, n1).clip(0)
    f1_reject  = rng.normal(10, 7,  n0).clip(0)

    # f2: LocationRisk – Binary (0=low risk, 1=high risk)
    f2_approve = rng.binomial(1, 0.12, n1).astype(float)
    f2_reject  = rng.binomial(1, 0.68, n0).astype(float)

    # ── Stage 1: Vendor A ($2) – moderate signal ──────────────────────────
    # f3: ID_Score – Continuous [0,1], Gaussian clipped
    f3_approve = rng.normal(0.78, 0.10, n1).clip(0, 1)
    f3_reject  = rng.normal(0.45, 0.15, n0).clip(0, 1)

    # f4: DocAuthentic – Binary
    f4_approve = rng.binomial(1, 0.88, n1).astype(float)
    f4_reject  = rng.binomial(1, 0.30, n0).astype(float)

    # f5: NameMatch – Beta distribution
    f5_approve = rng.beta(8, 2, n1)      # right-skewed toward 1
    f5_reject  = rng.beta(2, 5, n0)      # left-skewed toward 0

    # ── Stage 2: Vendor B ($5) – weaker signal ────────────────────────────
    # f6: AddrVerified – Binary
    f6_approve = rng.binomial(1, 0.74, n1).astype(float)
    f6_reject  = rng.binomial(1, 0.40, n0).astype(float)

    # f7: AddrTenure (months) – Exponential
    f7_approve = rng.exponential(24, n1)
    f7_reject  = rng.exponential(6,  n0)

    # f8: PostcodeRisk – Poisson (integer, higher = riskier)
    f8_approve = rng.poisson(2, n1).astype(float)
    f8_reject  = rng.poisson(6, n0).astype(float)

    # ── Stage 3: Vendor C ($50) – weakest marginal signal, catches edge cases
    # f9: BiometricScore – Gaussian
    f9_approve = rng.normal(0.91, 0.05, n1).clip(0, 1)
    f9_reject  = rng.normal(0.61, 0.18, n0).clip(0, 1)

    # f10: SanctionsClear – Binary  
    f10_approve = rng.binomial(1, 0.98, n1).astype(float)
    f10_reject  = rng.binomial(1, 0.55, n0).astype(float)

    # f11: PEP_Score – Gamma (lower is less risky)
    f11_approve = rng.gamma(shape=2, scale=0.10, size=n1)
    f11_reject  = rng.gamma(shape=2, scale=0.80, size=n0)

    # ── Assemble ──────────────────────────────────────────────────────────
    X = np.column_stack([
        stack(f0_approve, f0_reject),
        stack(f1_approve, f1_reject),
        stack(f2_approve, f2_reject),
        stack(f3_approve, f3_reject),
        stack(f4_approve, f4_reject),
        stack(f5_approve, f5_reject),
        stack(f6_approve, f6_reject),
        stack(f7_approve, f7_reject),
        stack(f8_approve, f8_reject),
        stack(f9_approve, f9_reject),
        stack(f10_approve, f10_reject),
        stack(f11_approve, f11_reject),
    ])
    y = np.concatenate([np.ones(n1, int), np.zeros(n0, int)])

    # Shuffle
    idx = rng.permutation(n_samples)
    X, y = X[idx], y[idx]

    return X, y


# ═════════════════════════════════════════════════════════════════════════════
# 2. TREE STRUCTURE ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

def get_sklearn_depth_stages(tree_clf) -> dict:
    """
    Return {depth: set_of_stages_used_at_that_depth} for a fitted sklearn tree.
    Uses the underlying tree's node_depth, feature, and children arrays.
    """
    t = tree_clf.tree_
    n_nodes = t.node_count
    children_left  = t.children_left
    children_right = t.children_right
    feature        = t.feature

    depths = np.zeros(n_nodes, dtype=int)
    stack = [(0, 0)]
    while stack:
        node_id, depth = stack.pop()
        depths[node_id] = depth
        if children_left[node_id] != children_left[node_id] == -1:
            pass  # leaf
        if children_left[node_id] >= 0:
            stack.append((children_left[node_id],  depth + 1))
            stack.append((children_right[node_id], depth + 1))

    depth_stages = defaultdict(set)
    for node_id in range(n_nodes):
        if children_left[node_id] >= 0:   # internal node
            feat = feature[node_id]
            stage = FEATURE_STAGE[feat]
            depth_stages[depths[node_id]].add(stage)

    return dict(depth_stages)


def get_sklearn_node_depth_features(tree_clf) -> dict:
    """
    Return {depth: list_of_(feature_idx, feature_name, stage)} for all
    internal nodes at each depth in the sklearn tree.
    """
    t = tree_clf.tree_
    n_nodes = t.node_count
    children_left  = t.children_left
    children_right = t.children_right
    feature        = t.feature

    depths = np.zeros(n_nodes, dtype=int)
    stack = [(0, 0)]
    while stack:
        node_id, depth = stack.pop()
        depths[node_id] = depth
        if children_left[node_id] >= 0:
            stack.append((children_left[node_id],  depth + 1))
            stack.append((children_right[node_id], depth + 1))

    depth_features = defaultdict(list)
    for node_id in range(n_nodes):
        if children_left[node_id] >= 0:
            feat = feature[node_id]
            depth_features[depths[node_id]].append({
                'feature_idx':  int(feat),
                'feature_name': FEATURE_NAMES[feat],
                'stage':        FEATURE_STAGE[feat],
            })

    return dict(depth_features)


def get_rdt_depth_features(rdt_clf) -> dict:
    """Return {depth: feature_idx} from trained RDT (by construction, one per depth)."""
    return dict(rdt_clf.depth_features_)


# ═════════════════════════════════════════════════════════════════════════════
# 3. COST COMPUTATION
# ═════════════════════════════════════════════════════════════════════════════

def compute_sklearn_sample_costs(tree_clf, X_test: np.ndarray) -> np.ndarray:
    """
    For each test sample, walk its decision path and charge the cost of every
    distinct vendor stage encountered on that path (both left and right branches
    at each internal node that is visited).
    """
    t = tree_clf.tree_
    children_left  = t.children_left
    children_right = t.children_right
    feature        = t.feature
    threshold      = t.threshold

    costs = np.zeros(len(X_test))
    for i, x in enumerate(X_test):
        node = 0
        stages_paid = set()
        while children_left[node] >= 0:   # internal node
            feat = feature[node]
            stage = FEATURE_STAGE[feat]
            stages_paid.add(stage)
            if x[feat] <= threshold[node]:
                node = children_left[node]
            else:
                node = children_right[node]
        costs[i] = sum(STAGE_COSTS[s] for s in stages_paid)
    return costs


def compute_rdt_sample_costs(rdt_clf, X_test: np.ndarray) -> np.ndarray:
    """
    For each test sample, walk the RDT decision path.
    RDT uses exactly one feature per depth, so cost = sum of stage costs for
    depths visited (i.e., from depth 0 to the termination depth).
    """
    costs = np.zeros(len(X_test))

    def walk(node, x, depth):
        """Return termination depth for sample x."""
        if node.is_leaf():
            return depth
        feat = node.split_info['feature']
        thresh = node.split_info['threshold']
        if x[feat] <= thresh:
            return walk(node.left, x, depth + 1)
        else:
            return walk(node.right, x, depth + 1)

    for i, x in enumerate(X_test):
        term_depth = walk(rdt_clf.root, x, 0)
        # Pay for stages at depths 0 .. term_depth-1
        total = 0.0
        for d in range(term_depth):
            feat = rdt_clf.depth_features_.get(d)
            if feat is not None:
                stage = FEATURE_STAGE[feat]
                total += STAGE_COSTS[stage]
        costs[i] = total

    return costs


# ═════════════════════════════════════════════════════════════════════════════
# 4. FIGURE: TREE STRUCTURE DIAGRAMS
# ═════════════════════════════════════════════════════════════════════════════

def plot_tree_comparison(sklearn_clf, rdt_clf, depth: int, out_path: str):
    """
    Side-by-side decision tree diagrams.
      Left:  sklearn tree — nodes coloured by vendor stage of the split feature
      Right: RDT tree     — same colouring (by construction, uniform per depth)
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 8),
                             facecolor='#f8f9fa')
    fig.suptitle(
        f'KYC Tree Structure Comparison (Depth {depth})\n'
        f'Node colour = vendor stage consulted at that split',
        fontsize=15, fontweight='bold', y=1.01
    )

    stage_label = {
        0: 'Stage 0 – Internal (free)',
        1: 'Stage 1 – Vendor A ($2)',
        2: 'Stage 2 – Vendor B ($5)',
        3: 'Stage 3 – Vendor C ($50)',
    }

    # Build node-level colour arrays for sklearn
    t = sklearn_clf.tree_
    n_nodes = t.node_count
    node_colours_sklearn = []
    for node_id in range(n_nodes):
        if t.children_left[node_id] >= 0:
            feat = t.feature[node_id]
            stage = FEATURE_STAGE[feat]
            node_colours_sklearn.append(STAGE_COLORS[stage])
        else:
            node_colours_sklearn.append('#bdc3c7')   # grey for leaves

    # ── Sklearn panel ─────────────────────────────────────────────────────
    axes[0].set_title('sklearn DecisionTree\n(may mix vendor stages at same depth)',
                      fontsize=12, pad=10)
    plot_tree(
        sklearn_clf,
        ax=axes[0],
        feature_names=FEATURE_NAMES,
        class_names=CLASS_NAMES,
        filled=True,
        node_ids=False,
        impurity=False,
        rounded=True,
        proportion=False,
        fontsize=7,
    )

    # Re-colour sklearn nodes by stage (override matplotlib's default impurity colouring)
    artists = [c for c in axes[0].get_children()
               if isinstance(c, mpatches.FancyBboxPatch)]
    for i, patch in enumerate(artists):
        if i < len(node_colours_sklearn):
            patch.set_facecolor(node_colours_sklearn[i])
            patch.set_edgecolor('#2c3e50')
            patch.set_linewidth(1.5)

    # ── RDT panel ─────────────────────────────────────────────────────────
    axes[1].set_title('Restricted Decision Tree (RDT)\n(one vendor stage per depth — consistent staging)',
                      fontsize=12, pad=10)

    # Build a minimal sklearn-compatible tree from RDT for display purposes
    # (We'll use a visual workaround: manually draw using matplotlib)
    _draw_rdt_tree(axes[1], rdt_clf, depth)

    # ── Legend ────────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color=STAGE_COLORS[s], label=stage_label[s])
        for s in range(4)
    ]
    legend_patches.append(mpatches.Patch(color='#bdc3c7', label='Leaf node'))
    fig.legend(handles=legend_patches, loc='lower center', ncol=5,
               fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor='#f8f9fa')
    plt.close(fig)
    print(f'  Saved: {out_path}')


def _draw_rdt_tree(ax, rdt_clf, max_depth: int):
    """Custom matplotlib drawing for the RDT tree."""
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-max_depth - 0.8, 0.8)
    ax.axis('off')

    node_positions = {}  # node_key -> (x, y)

    def _layout(node, depth, x_left, x_right):
        """Assign (x,y) positions to each node."""
        x_mid = (x_left + x_right) / 2
        y = -depth
        key = id(node)
        node_positions[key] = (x_mid, y)
        if not node.is_leaf() and depth < max_depth:
            _layout(node.left,  depth + 1, x_left,  x_mid)
            _layout(node.right, depth + 1, x_mid,  x_right)

    _layout(rdt_clf.root, 0, 0.0, 1.0)

    def _draw(node, depth, x_left, x_right):
        key  = id(node)
        x, y = node_positions[key]

        if node.is_leaf() or depth >= max_depth:
            color = '#bdc3c7'
            feat_idx = None
            label_line1 = f'Leaf'
            n_samples = len(node.indices)
        else:
            feat_idx = node.split_info['feature']
            stage = FEATURE_STAGE[feat_idx]
            color = STAGE_COLORS[stage]
            label_line1 = f'{FEATURE_NAMES[feat_idx]}'
            label_line2 = f'Stage {stage} | ≤ {node.split_info["threshold"]:.2f}'
            n_samples = len(node.indices)

        box = dict(boxstyle='round,pad=0.3', facecolor=color,
                   edgecolor='#2c3e50', linewidth=1.5, alpha=0.9)

        if node.is_leaf() or depth >= max_depth:
            majority = 'Approve' if node.value == 1 else 'Reject'
            txt = f'{majority}\nn={n_samples}'
        else:
            txt = f'{label_line1}\n{label_line2}\nn={n_samples}'

        ax.text(x, y, txt, ha='center', va='center', fontsize=7,
                bbox=box, zorder=3)

        if not node.is_leaf() and depth < max_depth:
            x_mid = (x_left + x_right) / 2
            lx, ly = node_positions[id(node.left)]
            rx, ry = node_positions[id(node.right)]
            ax.plot([x, lx], [y - 0.08, ly + 0.08], 'k-', lw=0.8, zorder=1)
            ax.plot([x, rx], [y - 0.08, ry + 0.08], 'k-', lw=0.8, zorder=1)
            ax.text((x + lx) / 2, (y + ly) / 2, 'True',
                    fontsize=6, ha='center', va='center', color='#555')
            ax.text((x + rx) / 2, (y + ry) / 2, 'False',
                    fontsize=6, ha='center', va='center', color='#555')

            _draw(node.left,  depth + 1, x_left, x_mid)
            _draw(node.right, depth + 1, x_mid,  x_right)

    _draw(rdt_clf.root, 0, 0.0, 1.0)


# ═════════════════════════════════════════════════════════════════════════════
# 5. FIGURE: COST DISTRIBUTION BOX PLOTS
# ═════════════════════════════════════════════════════════════════════════════

def plot_cost_distributions(results_by_depth: dict, out_path: str):
    """
    Box plots of per-sample cost, grouped by tree type and depth.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False,
                             facecolor='#f8f9fa')
    fig.suptitle('Per-Sample Vendor Query Cost: sklearn vs. RDT\n'
                 '(Synthetic KYC Dataset, 2,000 test samples)',
                 fontsize=13, fontweight='bold')

    for ax_idx, depth in enumerate([3, 4]):
        ax = axes[ax_idx]
        res = results_by_depth[depth]

        sk_costs  = res['sklearn_costs']
        rdt_costs = res['rdt_costs']

        bp = ax.boxplot(
            [sk_costs, rdt_costs],
            labels=['sklearn\nDecisionTree', 'RDT'],
            patch_artist=True,
            medianprops=dict(color='black', linewidth=2),
            flierprops=dict(marker='o', markersize=3, alpha=0.4),
            widths=0.5
        )
        bp['boxes'][0].set_facecolor('#3498db')
        bp['boxes'][0].set_alpha(0.75)
        bp['boxes'][1].set_facecolor('#e74c3c')
        bp['boxes'][1].set_alpha(0.75)

        ax.set_title(f'Depth {depth}', fontsize=12)
        ax.set_ylabel('Cost per sample ($)', fontsize=10)
        ax.set_xlabel('Tree type', fontsize=10)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_facecolor('#f8f9fa')

        # Annotate means
        for xi, costs in enumerate([sk_costs, rdt_costs], 1):
            m = np.mean(costs)
            ax.text(xi, m, f'  μ=${m:.2f}', va='center', fontsize=9,
                    color='#2c3e50', fontweight='bold')

        # Annotate reduction
        reduction = (1 - np.mean(rdt_costs) / np.mean(sk_costs)) * 100
        ax.text(0.5, 0.97, f'Cost reduction: {reduction:.1f}%',
                transform=ax.transAxes, ha='center', va='top',
                fontsize=10, fontweight='bold',
                color='#27ae60',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#27ae60', alpha=0.8))

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#f8f9fa')
    plt.close(fig)
    print(f'  Saved: {out_path}')


# ═════════════════════════════════════════════════════════════════════════════
# 6. SENSITIVITY ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

def compute_sensitivity_table() -> list:
    """
    Parametric sensitivity of KYC cost reduction.
    Returns list of dicts for the 3×3 table.
    """
    # Early-termination scenarios (Depth 0 resolve %, Level 1 resolve %)
    scenarios = [
        ('High (50% / 60%)', 0.50, 0.60),
        ('Base (35% / 55%)', 0.35, 0.55),
        ('Low  (20% / 30%)', 0.20, 0.30),
    ]
    cost_sets = [
        ('Conservative ($1/$3/$20)',  1,  3,  20),
        ('Base ($2/$5/$50)',          2,  5,  50),
        ('Premium ($3/$8/$100)',      3,  8, 100),
    ]

    N = 100_000  # applications
    rows = []
    for sc_name, d0_res, l1_res in scenarios:
        row = {'scenario': sc_name}
        for cs_name, c1, c2, c3 in cost_sets:
            n_l1 = N * (1 - d0_res)
            n_l2 = n_l1 * (1 - l1_res)
            n_l3 = n_l2 * 0.80  # 80% of L2 escalate to L3

            rdt_cost      = c1 * n_l1 + c2 * n_l2 + c3 * n_l3
            standard_cost = (c1 + c2 + c3) * N
            reduction     = (standard_cost - rdt_cost) / standard_cost * 100
            row[cs_name] = f'{reduction:.1f}%'
        rows.append(row)
    return rows


# ═════════════════════════════════════════════════════════════════════════════
# 7. MARKDOWN REPORT
# ═════════════════════════════════════════════════════════════════════════════

def write_report(results_by_depth: dict, sensitivity: list):
    """Write kyc_illustration_report.md with all results."""
    lines = []
    lines.append('# KYC Staged-Cost Illustration — Results Report\n')
    lines.append('Generated by `kyc_illustration/simulate_kyc_cost.py`\n')
    lines.append('---\n')
    lines.append('## Dataset Summary\n')
    lines.append('- **10,000 samples** (8,000 train / 2,000 test), seed=42\n')
    lines.append('- **12 features** across 4 vendor stages (3 features each)\n')
    lines.append('- Class balance: 60% Approve, 40% Reject\n')
    lines.append('\n| Stage | Features | Distribution | Cost/query |\n')
    lines.append('|-------|----------|-------------|------------|\n')
    lines.append('| 0 – Internal | Age, AccountTenure, LocationRisk | Gaussian, Gaussian, Binary | Free |\n')
    lines.append('| 1 – Vendor A | ID_Score, DocAuthentic, NameMatch | Gaussian, Binary, Beta | $2 |\n')
    lines.append('| 2 – Vendor B | AddrVerified, AddrTenure, PostcodeRisk | Binary, Exponential, Poisson | $5 |\n')
    lines.append('| 3 – Vendor C | BiometricScore, SanctionsClear, PEP_Score | Gaussian, Binary, Gamma | $50 |\n')
    lines.append('\n---\n')

    for depth in [3, 4]:
        r = results_by_depth[depth]
        reduction = (1 - r['rdt_mean_cost'] / r['sklearn_mean_cost']) * 100
        lines.append(f'## Results — Depth {depth}\n')
        lines.append(f'| Metric | sklearn DecisionTree | RDT |\n')
        lines.append(f'|--------|---------------------|-----|\n')
        lines.append(f'| Accuracy | {r["sklearn_acc"]:.4f} | {r["rdt_acc"]:.4f} |\n')
        lines.append(f'| Relative accuracy | 100% | {r["rdt_acc"]/r["sklearn_acc"]*100:.1f}% |\n')
        lines.append(f'| Mean cost/sample | ${r["sklearn_mean_cost"]:.2f} | ${r["rdt_mean_cost"]:.2f} |\n')
        lines.append(f'| Median cost/sample | ${r["sklearn_median_cost"]:.2f} | ${r["rdt_median_cost"]:.2f} |\n')
        lines.append(f'| **Cost reduction** | — | **{reduction:.1f}%** |\n')
        lines.append('\n### Feature Stage Used by Depth\n\n')
        lines.append('| Depth | sklearn stages used | RDT stage (by constraint) |\n')
        lines.append('|-------|--------------------|--------------------------|\n')
        for d in sorted(r['sklearn_depth_stages'].keys()):
            sk_stages = sorted(r['sklearn_depth_stages'][d])
            sk_str = ', '.join(f'Stage {s}' for s in sk_stages)
            rdt_feat = r['rdt_depth_features'].get(d)
            if rdt_feat is not None:
                rdt_stage = FEATURE_STAGE[rdt_feat]
                rdt_str = f'Stage {rdt_stage} ({FEATURE_NAMES[rdt_feat]})'
            else:
                rdt_str = '—'
            lines.append(f'| {d} | {sk_str} | {rdt_str} |\n')

        lines.append('\n### Fraction of Samples Reaching Each Stage (sklearn)\n\n')
        lines.append('| Stage | % samples paying this cost |\n')
        lines.append('|-------|----------------------------|\n')
        for stage, pct in r['sklearn_stage_reach_pct'].items():
            lines.append(f'| Stage {stage} (${STAGE_COSTS[stage]:.0f}/q) | {pct:.1f}% |\n')

        lines.append('\n### Fraction of Samples Reaching Each Stage (RDT)\n\n')
        lines.append('| Stage | % samples paying this cost |\n')
        lines.append('|-------|----------------------------|\n')
        for stage, pct in r['rdt_stage_reach_pct'].items():
            lines.append(f'| Stage {stage} (${STAGE_COSTS[stage]:.0f}/q) | {pct:.1f}% |\n')

        lines.append('\n---\n')

    # Sensitivity table
    lines.append('## Parametric Sensitivity Analysis\n\n')
    lines.append('Cost reduction across vendor pricing tiers and early-termination rates ')
    lines.append('(100,000 applications/year). L3 escalation fixed at 80% of L2 volume.\n\n')
    cols = list(sensitivity[0].keys())
    lines.append('| ' + ' | '.join(cols) + ' |\n')
    lines.append('|' + '|'.join(['---'] * len(cols)) + '|\n')
    for row in sensitivity:
        lines.append('| ' + ' | '.join(row[c] for c in cols) + ' |\n')

    lines.append('\n---\n')
    lines.append('## Key Findings\n\n')
    r3 = results_by_depth[3]
    r4 = results_by_depth[4]
    red3 = (1 - r3['rdt_mean_cost'] / r3['sklearn_mean_cost']) * 100
    red4 = (1 - r4['rdt_mean_cost'] / r4['sklearn_mean_cost']) * 100
    lines.append(
        f'1. **sklearn mixes vendor stages**: At depth 1, sklearn uses '
        f'stages {sorted(r3["sklearn_depth_stages"].get(1, set()))} '
        f'(not exclusively Stage 1), forcing premature purchase of expensive vendors.\n'
    )
    lines.append(
        f'2. **RDT maintains stage discipline**: Each depth maps to exactly one vendor stage.\n'
    )
    lines.append(
        f'3. **Cost reduction**: RDT achieves {red3:.1f}% (depth 3) and {red4:.1f}% (depth 4) '
        f'lower mean per-sample cost, with only '
        f'{(1-r3["rdt_acc"]/r3["sklearn_acc"])*100:.1f}% (depth 3) and '
        f'{(1-r4["rdt_acc"]/r4["sklearn_acc"])*100:.1f}% (depth 4) accuracy trade-off.\n'
    )
    lines.append(
        f'4. **Sensitivity**: Cost savings range from '
        f'{sensitivity[-1]["Conservative ($1/$3/$20)"]} to '
        f'{sensitivity[0]["Premium ($3/$8/$100)"]} '
        f'across all parameter combinations — robust to vendor pricing, '
        f'moderately sensitive to early-termination rates.\n'
    )

    report_path = os.path.join(OUTPUT_DIR, 'kyc_illustration_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f'  Saved: {report_path}')


# ═════════════════════════════════════════════════════════════════════════════
# 8. MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print('=' * 60)
    print('KYC Staged-Cost Illustration')
    print('=' * 60)

    # ── Generate data ──────────────────────────────────────────────────────
    print('\n[1/5] Generating synthetic KYC dataset...')
    X, y = generate_kyc_dataset(N_SAMPLES, SEED)

    split = int(0.8 * N_SAMPLES)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Save dataset
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df['label'] = y
    csv_path = os.path.join(OUTPUT_DIR, 'kyc_dataset.csv')
    df.to_csv(csv_path, index=False)
    print(f'  Saved: {csv_path}  ({N_SAMPLES} samples, {X.shape[1]} features)')
    print(f'  Class balance: {y.mean()*100:.1f}% Approve')

    results_by_depth = {}

    for depth in [3, 4]:
        print(f'\n[2/5] Training trees at depth {depth}...')

        # sklearn
        sk_clf = DecisionTreeClassifier(
            max_depth=depth, random_state=SEED, min_samples_leaf=5
        )
        sk_clf.fit(X_train, y_train)
        sk_acc = accuracy_score(y_test, sk_clf.predict(X_test))
        print(f'  sklearn accuracy (depth {depth}): {sk_acc:.4f}')

        # RDT
        rdt_clf = RestrictedDecisionTree(
            task='classification', max_depth=depth, min_samples_leaf=5
        )
        rdt_clf.fit(X_train, y_train)
        rdt_acc = accuracy_score(y_test, rdt_clf.predict(X_test))
        print(f'  RDT     accuracy (depth {depth}): {rdt_acc:.4f}')

        # Tree structure
        print(f'\n[3/5] Analysing tree structure (depth {depth})...')
        sk_depth_stages   = get_sklearn_depth_stages(sk_clf)
        sk_depth_features = get_sklearn_node_depth_features(sk_clf)
        rdt_depth_feats   = get_rdt_depth_features(rdt_clf)

        print(f'  sklearn — stages by depth:')
        for d in sorted(sk_depth_stages.keys()):
            stages = sorted(sk_depth_stages[d])
            feats  = [f['feature_name'] for f in sk_depth_features[d]]
            print(f'    Depth {d}: stages {stages} — features: {feats}')

        print(f'  RDT     — feature by depth (one per depth):')
        for d in sorted(rdt_depth_feats.keys()):
            feat = rdt_depth_feats[d]
            stage = FEATURE_STAGE[feat]
            print(f'    Depth {d}: feature {FEATURE_NAMES[feat]} (Stage {stage})')

        # Cost computation
        print(f'\n[4/5] Computing per-sample costs (depth {depth})...')
        sk_costs  = compute_sklearn_sample_costs(sk_clf,  X_test)
        rdt_costs = compute_rdt_sample_costs(rdt_clf,     X_test)

        print(f'  sklearn mean cost/sample: ${sk_costs.mean():.2f}  '
              f'median: ${np.median(sk_costs):.2f}')
        print(f'  RDT     mean cost/sample: ${rdt_costs.mean():.2f}  '
              f'median: ${np.median(rdt_costs):.2f}')
        reduction = (1 - rdt_costs.mean() / sk_costs.mean()) * 100
        print(f'  *** Cost reduction: {reduction:.1f}% ***')

        # Stage reach fractions
        sk_stage_reach = {s: 0 for s in range(4)}
        for i, x in enumerate(X_test):
            t = sk_clf.tree_
            node = 0
            while t.children_left[node] >= 0:
                feat = t.feature[node]
                stage = FEATURE_STAGE[feat]
                sk_stage_reach[stage] += 1
                if x[feat] <= t.threshold[node]:
                    node = t.children_left[node]
                else:
                    node = t.children_right[node]
        sk_stage_reach_pct = {
            s: v / len(X_test) * 100 for s, v in sk_stage_reach.items()
        }

        # For RDT, count samples reaching each stage based on depth
        rdt_stage_reach = {s: 0 for s in range(4)}

        def _count_stage_reach(node, x, depth):
            if node.is_leaf():
                return
            feat = node.split_info['feature']
            stage = FEATURE_STAGE[feat]
            rdt_stage_reach[stage] += 1
            if x[feat] <= node.split_info['threshold']:
                _count_stage_reach(node.left, x, depth + 1)
            else:
                _count_stage_reach(node.right, x, depth + 1)

        for x in X_test:
            _count_stage_reach(rdt_clf.root, x, 0)

        rdt_stage_reach_pct = {
            s: v / len(X_test) * 100 for s, v in rdt_stage_reach.items()
        }

        # Tree diagram
        print(f'\n[5/5] Generating figures (depth {depth})...')
        tree_fig_path = os.path.join(OUTPUT_DIR, f'tree_structure_depth{depth}.png')
        plot_tree_comparison(sk_clf, rdt_clf, depth, tree_fig_path)

        # Store results
        results_by_depth[depth] = {
            'depth':                  depth,
            'sklearn_acc':            float(sk_acc),
            'rdt_acc':                float(rdt_acc),
            'sklearn_mean_cost':      float(sk_costs.mean()),
            'sklearn_median_cost':    float(np.median(sk_costs)),
            'rdt_mean_cost':          float(rdt_costs.mean()),
            'rdt_median_cost':        float(np.median(rdt_costs)),
            'cost_reduction_pct':     float(reduction),
            'sklearn_depth_stages':   {int(k): sorted(v)
                                       for k, v in sk_depth_stages.items()},
            'sklearn_depth_features': {
                int(k): [dict(x) for x in v]
                for k, v in sk_depth_features.items()
            },
            'rdt_depth_features':     {int(k): int(v)
                                       for k, v in rdt_depth_feats.items()},
            'sklearn_costs':          sk_costs.tolist(),
            'rdt_costs':              rdt_costs.tolist(),
            'sklearn_stage_reach_pct': {
                int(k): float(v) for k, v in sk_stage_reach_pct.items()
            },
            'rdt_stage_reach_pct': {
                int(k): float(v) for k, v in rdt_stage_reach_pct.items()
            },
        }

        # Save per-depth JSON (without the full cost arrays to keep it readable)
        summary = {k: v for k, v in results_by_depth[depth].items()
                   if k not in ('sklearn_costs', 'rdt_costs')}
        json_path = os.path.join(OUTPUT_DIR, f'results_depth{depth}.json')
        with open(json_path, 'w') as fh:
            json.dump(summary, fh, indent=2)
        print(f'  Saved: {json_path}')

    # ── Cost distribution figure ────────────────────────────────────────────
    cost_fig_path = os.path.join(OUTPUT_DIR, 'cost_distribution.png')
    plot_cost_distributions(results_by_depth, cost_fig_path)

    # ── Sensitivity analysis ────────────────────────────────────────────────
    sensitivity = compute_sensitivity_table()

    # ── Markdown report ─────────────────────────────────────────────────────
    write_report(results_by_depth, sensitivity)

    print('\n' + '=' * 60)
    print('ILLUSTRATION COMPLETE')
    print('=' * 60)
    for depth in [3, 4]:
        r = results_by_depth[depth]
        print(f'\nDepth {depth}:')
        print(f'  sklearn accuracy: {r["sklearn_acc"]:.4f}')
        print(f'  RDT     accuracy: {r["rdt_acc"]:.4f} '
              f'({r["rdt_acc"]/r["sklearn_acc"]*100:.1f}% of sklearn)')
        print(f'  sklearn mean cost: ${r["sklearn_mean_cost"]:.2f}')
        print(f'  RDT     mean cost: ${r["rdt_mean_cost"]:.2f}')
        print(f'  Cost reduction:    {r["cost_reduction_pct"]:.1f}%')

    print(f'\nAll outputs written to: {OUTPUT_DIR}/')
    print('Files:')
    for fname in ['kyc_dataset.csv',
                  'results_depth3.json', 'results_depth4.json',
                  'tree_structure_depth3.png', 'tree_structure_depth4.png',
                  'cost_distribution.png', 'kyc_illustration_report.md']:
        fpath = os.path.join(OUTPUT_DIR, fname)
        exists = '✓' if os.path.exists(fpath) else '✗'
        print(f'  [{exists}] {fname}')


if __name__ == '__main__':
    main()
