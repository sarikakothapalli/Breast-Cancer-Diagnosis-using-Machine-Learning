import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.patches import Patch

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
# If using sklearn's built-in dataset:
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['diagnosis'] = pd.Categorical.from_codes(data.target, ['M', 'B'])

# ── OR if loading your own CSV, comment above and use: ──
# df = pd.read_csv('your_breast_cancer_file.csv')

# ─────────────────────────────────────────────
# GLOBAL STYLE
# ─────────────────────────────────────────────
MALIGNANT_COLOR = '#E05C5C'   # red
BENIGN_COLOR    = '#5B8FD4'   # blue
PALETTE         = {'M': MALIGNANT_COLOR, 'B': BENIGN_COLOR}

plt.rcParams.update({
    'figure.facecolor': '#F9F9F9',
    'axes.facecolor':   '#F9F9F9',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'font.family':       'DejaVu Sans',
    'axes.titlesize':    13,
    'axes.labelsize':    11,
})

# ─────────────────────────────────────────────
# FIGURE 1 — Diagnosis Distribution
# ─────────────────────────────────────────────
counts = df['diagnosis'].value_counts()
labels = ['Benign (B)', 'Malignant (M)']
sizes  = [counts.get('B', 0), counts.get('M', 0)]
colors = [BENIGN_COLOR, MALIGNANT_COLOR]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Figure 1 — Diagnosis Distribution', fontsize=15, fontweight='bold', y=1.02)

# Donut
wedges, texts, autotexts = axes[0].pie(
    sizes, labels=labels, colors=colors,
    autopct='%1.1f%%', startangle=90,
    wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2),
    textprops={'fontsize': 11}
)
for at in autotexts:
    at.set_fontweight('bold')
axes[0].set_title('Proportion of Diagnoses')

# Bar
bars = axes[1].bar(labels, sizes, color=colors, width=0.4,
                   edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, sizes):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 str(val), ha='center', va='bottom', fontweight='bold', fontsize=12)
axes[1].set_title('Count of Diagnoses')
axes[1].set_ylabel('Number of Samples')
axes[1].set_ylim(0, max(sizes) * 1.2)

plt.tight_layout()
plt.savefig('fig1_diagnosis_distribution.png', dpi=180, bbox_inches='tight')
plt.show()
print("✅ Figure 1 saved.")


# ─────────────────────────────────────────────
# FIGURE 2 — Box Plots: Mean Features by Diagnosis
# ─────────────────────────────────────────────
mean_features = [
    'mean radius', 'mean texture', 'mean perimeter',
    'mean area', 'mean concavity', 'mean concave points'
]
# handle both naming conventions
available = []
for f in mean_features:
    if f in df.columns:
        available.append(f)
    elif f.replace('mean ', '') + '_mean' in df.columns:
        available.append(f.replace('mean ', '') + '_mean')

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('Figure 2 — Distribution of Mean Features by Diagnosis',
             fontsize=15, fontweight='bold')

for ax, feat in zip(axes.flatten(), available):
    sns.boxplot(x='diagnosis', y=feat, data=df,
                palette=PALETTE, width=0.5,
                flierprops=dict(marker='o', markerfacecolor='grey',
                                markersize=3, alpha=0.5),
                ax=ax)
    ax.set_title(feat.replace('_', ' ').title())
    ax.set_xlabel('')
    ax.set_ylabel('')

legend_elements = [Patch(facecolor=MALIGNANT_COLOR, label='Malignant (M)'),
                   Patch(facecolor=BENIGN_COLOR,    label='Benign (B)')]
fig.legend(handles=legend_elements, loc='upper right', fontsize=11)
plt.tight_layout()
plt.savefig('fig2_boxplots_mean_features.png', dpi=180, bbox_inches='tight')
plt.show()
print("✅ Figure 2 saved.")


# ─────────────────────────────────────────────
# FIGURE 3 — Correlation Heatmap
# ─────────────────────────────────────────────
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()

fig, ax = plt.subplots(figsize=(16, 13))
fig.suptitle('Figure 3 — Feature Correlation Heatmap',
             fontsize=15, fontweight='bold', y=1.01)

mask = np.triu(np.ones_like(corr, dtype=bool))   # show lower triangle only
sns.heatmap(corr, mask=mask, annot=False, cmap='coolwarm',
            center=0, linewidths=0.4, linecolor='white',
            square=True, ax=ax, cbar_kws={'shrink': 0.8})

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

plt.tight_layout()
plt.savefig('fig3_correlation_heatmap.png', dpi=180, bbox_inches='tight')
plt.show()
print("✅ Figure 3 saved.")


# ─────────────────────────────────────────────
# FIGURE 4 — Average Feature Values by Diagnosis (Grouped Bar)
# ─────────────────────────────────────────────
compare_features = ['mean radius', 'mean texture', 'mean perimeter',
                    'mean smoothness', 'mean compactness', 'mean symmetry']
available_compare = []
for f in compare_features:
    if f in df.columns:
        available_compare.append(f)
    else:
        alt = f.replace('mean ', '') + '_mean'
        if alt in df.columns:
            available_compare.append(alt)

group_means = df.groupby('diagnosis')[available_compare].mean()

x      = np.arange(len(available_compare))
width  = 0.35
fig, ax = plt.subplots(figsize=(14, 6))
fig.suptitle('Figure 4 — Average Feature Values by Diagnosis',
             fontsize=15, fontweight='bold')

bars_m = ax.bar(x - width/2, group_means.loc['M'], width,
                label='Malignant', color=MALIGNANT_COLOR, edgecolor='white')
bars_b = ax.bar(x + width/2, group_means.loc['B'], width,
                label='Benign',    color=BENIGN_COLOR,    edgecolor='white')

ax.set_xticks(x)
ax.set_xticklabels([f.replace('_', ' ').title() for f in available_compare],
                   rotation=20, ha='right')
ax.set_ylabel('Average Value')
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig('fig4_avg_feature_comparison.png', dpi=180, bbox_inches='tight')
plt.show()
print("✅ Figure 4 saved.")


# ─────────────────────────────────────────────
# FIGURE 5 — Scatter Plot: Radius vs Area
# ─────────────────────────────────────────────
radius_col = 'mean radius' if 'mean radius' in df.columns else 'radius_mean'
area_col   = 'mean area'   if 'mean area'   in df.columns else 'area_mean'

fig, ax = plt.subplots(figsize=(10, 7))
fig.suptitle('Figure 5 — Radius vs Area by Diagnosis',
             fontsize=15, fontweight='bold')

for diag, color, label in [('M', MALIGNANT_COLOR, 'Malignant'),
                             ('B', BENIGN_COLOR,   'Benign')]:
    subset = df[df['diagnosis'] == diag]
    ax.scatter(subset[radius_col], subset[area_col],
               c=color, label=label, alpha=0.6, edgecolors='white',
               linewidth=0.5, s=60)

ax.set_xlabel(radius_col.replace('_', ' ').title())
ax.set_ylabel(area_col.replace('_', ' ').title())
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig('fig5_scatter_radius_area.png', dpi=180, bbox_inches='tight')
plt.show()
print("✅ Figure 5 saved.")


# ─────────────────────────────────────────────
# FIGURE 6 — Worst Features Comparison
# ─────────────────────────────────────────────
worst_features = ['worst radius', 'worst perimeter',
                  'worst area', 'worst concavity', 'worst concave points']
available_worst = []
for f in worst_features:
    if f in df.columns:
        available_worst.append(f)
    else:
        alt = f.replace('worst ', '') + '_worst'
        if alt in df.columns:
            available_worst.append(alt)

fig, axes = plt.subplots(1, len(available_worst), figsize=(18, 6))
fig.suptitle('Figure 6 — Worst Feature Distribution by Diagnosis',
             fontsize=15, fontweight='bold')

for ax, feat in zip(axes, available_worst):
    sns.violinplot(x='diagnosis', y=feat, data=df,
                   palette=PALETTE, inner='box', ax=ax, linewidth=1.2)
    ax.set_title(feat.replace('_', ' ').title(), fontsize=10)
    ax.set_xlabel('')
    ax.set_ylabel('')

legend_elements = [Patch(facecolor=MALIGNANT_COLOR, label='Malignant (M)'),
                   Patch(facecolor=BENIGN_COLOR,    label='Benign (B)')]
fig.legend(handles=legend_elements, loc='upper right', fontsize=11)
plt.tight_layout()
plt.savefig('fig6_worst_features_violin.png', dpi=180, bbox_inches='tight')
plt.show()
print("✅ Figure 6 saved.")


# ─────────────────────────────────────────────
# FIGURE 7 — Pairplot (Bonus)
# ─────────────────────────────────────────────
pair_cols = [radius_col, area_col,
             'mean concavity'      if 'mean concavity'      in df.columns else 'concavity_mean',
             'mean concave points' if 'mean concave points' in df.columns else 'concave points_mean',
             'diagnosis']
pair_df = df[[c for c in pair_cols if c in df.columns]]

g = sns.pairplot(pair_df, hue='diagnosis', palette=PALETTE,
                 plot_kws={'alpha': 0.5, 'edgecolor': 'none', 's': 25},
                 diag_kind='kde')
g.figure.suptitle('Figure 7 — Pairplot of Key Features', y=1.02,
                  fontsize=15, fontweight='bold')

plt.savefig('fig7_pairplot.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Figure 7 saved.")

print("\n🎉 All figures saved! Add them to your research paper as Figure 1–7.")