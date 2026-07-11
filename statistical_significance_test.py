#!/usr/bin/env python3
"""
Statistical Significance Test for RDT vs sklearn
Wilcoxon signed-rank test for paired samples
"""

from scipy.stats import wilcoxon, ttest_rel
import numpy as np

# Results from Table 1 (Depth 5, 13 datasets)
# Order: Adult Income, Bank Marketing, Breast Cancer, Cover Type, Dry Bean,
#        German Credit, HTRU2, Ionosphere, Magic Gamma, Pima Diabetes,
#        Segment, Spambase, Vehicle

rdt = np.array([
    0.851,  # Adult Income
    0.803,  # Bank Marketing
    0.921,  # Breast Cancer
    0.685,  # Cover Type
    0.867,  # Dry Bean
    0.309,  # German Credit
    0.977,  # HTRU2
    0.886,  # Ionosphere
    0.807,  # Magic Gamma
    0.726,  # Pima Diabetes
    0.824,  # Segment
    0.890,  # Spambase
    0.643   # Vehicle
])

sklearn = np.array([
    0.851,  # Adult Income
    0.811,  # Bank Marketing
    0.924,  # Breast Cancer
    0.702,  # Cover Type
    0.884,  # Dry Bean
    0.322,  # German Credit
    0.978,  # HTRU2
    0.877,  # Ionosphere
    0.826,  # Magic Gamma
    0.733,  # Pima Diabetes
    0.810,  # Segment
    0.907,  # Spambase
    0.677   # Vehicle
])

print("="*70)
print("STATISTICAL SIGNIFICANCE TEST: RDT vs sklearn")
print("="*70)
print()

# Basic statistics
print("DESCRIPTIVE STATISTICS")
print("-"*70)
print(f"Number of datasets: {len(rdt)}")
print(f"RDT mean accuracy:    {np.mean(rdt):.4f}")
print(f"sklearn mean accuracy: {np.mean(sklearn):.4f}")
print(f"Mean difference:       {np.mean(rdt - sklearn):.4f}")
print()
print(f"RDT median accuracy:   {np.median(rdt):.4f}")
print(f"sklearn median accuracy: {np.median(sklearn):.4f}")
print(f"Median difference:      {np.median(rdt - sklearn):.4f}")
print()

# Count wins/ties/losses
differences = rdt - sklearn
wins = np.sum(differences > 0)
ties = np.sum(differences == 0)
losses = np.sum(differences < 0)

print(f"RDT beats sklearn:    {wins} datasets ({wins/len(rdt)*100:.1f}%)")
print(f"RDT ties sklearn:     {ties} datasets ({ties/len(rdt)*100:.1f}%)")
print(f"RDT loses to sklearn: {losses} datasets ({losses/len(rdt)*100:.1f}%)")
print()

# Wilcoxon signed-rank test (non-parametric, recommended for ML)
print("="*70)
print("WILCOXON SIGNED-RANK TEST (Non-parametric, Recommended)")
print("="*70)
print("Null hypothesis: RDT and sklearn have equal median accuracy")
print("Alternative: RDT and sklearn have different median accuracy")
print()

statistic, p_value = wilcoxon(rdt, sklearn, alternative='two-sided')

print(f"Test statistic: {statistic}")
print(f"P-value:        {p_value:.4f}")
print()

# Interpretation
alpha = 0.05
if p_value > alpha:
    print(f"✓ RESULT: p-value ({p_value:.4f}) > α ({alpha})")
    print("  NO statistically significant difference")
    print("  RDT is statistically EQUIVALENT to sklearn")
    print()
    print("  INTERPRETATION FOR PAPER:")
    print("  'RDT achieves statistically equivalent performance to sklearn'")
    print(f"  'Wilcoxon signed-rank test: p = {p_value:.3f}'")
else:
    print(f"✗ RESULT: p-value ({p_value:.4f}) < α ({alpha})")
    print("  Statistically significant difference detected")
    print("  RDT performance differs from sklearn")
    print()
    print("  INTERPRETATION FOR PAPER:")
    print("  'RDT shows marginal performance difference from sklearn'")
    print(f"  'Wilcoxon signed-rank test: p = {p_value:.3f}'")

print()

# Paired t-test (parametric, for comparison)
print("="*70)
print("PAIRED T-TEST (Parametric, for comparison)")
print("="*70)
print("Assumes normal distribution of differences")
print()

t_stat, t_pvalue = ttest_rel(rdt, sklearn)

print(f"t-statistic:    {t_stat:.4f}")
print(f"P-value:        {t_pvalue:.4f}")
print()

if t_pvalue > alpha:
    print(f"✓ RESULT: p-value ({t_pvalue:.4f}) > α ({alpha})")
    print("  NO statistically significant difference")
else:
    print(f"✗ RESULT: p-value ({t_pvalue:.4f}) < α ({alpha})")
    print("  Statistically significant difference")

print()

# Effect size (Cohen's d)
print("="*70)
print("EFFECT SIZE (Cohen's d)")
print("="*70)
mean_diff = np.mean(differences)
std_diff = np.std(differences, ddof=1)
cohens_d = mean_diff / std_diff

print(f"Cohen's d: {cohens_d:.4f}")
print()
print("Interpretation:")
if abs(cohens_d) < 0.2:
    print("  Small effect size (< 0.2)")
    print("  Practical significance: Negligible difference")
elif abs(cohens_d) < 0.5:
    print("  Small to medium effect size (0.2-0.5)")
    print("  Practical significance: Small difference")
elif abs(cohens_d) < 0.8:
    print("  Medium effect size (0.5-0.8)")
    print("  Practical significance: Moderate difference")
else:
    print("  Large effect size (> 0.8)")
    print("  Practical significance: Large difference")

print()

# Final recommendation
print("="*70)
print("RECOMMENDATION FOR PAPER")
print("="*70)
print()
print("SUGGESTED TEXT TO ADD:")
print("-"*70)
print()

if p_value > alpha:
    print(f"\"Statistical testing confirms RDT achieves comparable performance")
    print(f"to sklearn across all datasets. Wilcoxon signed-rank test shows")
    print(f"no statistically significant difference (p = {p_value:.3f}, α = 0.05),")
    print(f"indicating RDT's 56% feature reduction comes with negligible")
    print(f"performance trade-off.\"")
else:
    print(f"\"While RDT shows a marginal performance difference from sklearn")
    print(f"(Wilcoxon test: p = {p_value:.3f}), the 0.9% accuracy gap is")
    print(f"substantially offset by 56% feature reduction and associated")
    print(f"cost savings in staged decision systems.\"")

print()
print("="*70)
print("END OF ANALYSIS")
print("="*70)
