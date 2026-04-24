import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report
)

# ─────────────────────────────────────────────
# COLORS & STYLE
# ─────────────────────────────────────────────
MALIGNANT_COLOR = '#E05C5C'
BENIGN_COLOR    = '#5B8FD4'
MODEL_COLORS    = ['#5B8FD4', '#2ecc71', '#e67e22']
MODEL_NAMES     = ['Logistic Regression', 'Random Forest', 'KNN (k=5)']

plt.rcParams.update({
    'figure.facecolor': '#F9F9F9',
    'axes.facecolor':   '#F9F9F9',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'font.family':       'DejaVu Sans',
})

# ─────────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────────
# Option A: sklearn built-in dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['diagnosis'] = data.target   # 0 = Malignant, 1 = Benign (sklearn reverses it)
# Fix: sklearn uses 0=malignant, 1=benign — we remap to match paper (1=M, 0=B)
df['diagnosis'] = df['diagnosis'].map({0: 1, 1: 0})

# Option B: Your own CSV — comment above 4 lines and use:
# df = pd.read_csv('your_file.csv')
# df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

print(f"Dataset shape: {df.shape}")
print(f"Class distribution:\n{y.value_counts().rename({1: 'Malignant', 0: 'Benign'})}\n")

# ─────────────────────────────────────────────
# STEP 2: TRAIN-TEST SPLIT (80/20, stratified)
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─────────────────────────────────────────────
# STEP 3: FEATURE SCALING
# Fit ONLY on training data to prevent data leakage
# ─────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# STEP 4: DEFINE MODELS
# ─────────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(C=1.0, max_iter=10000, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    'KNN (k=5)':           KNeighborsClassifier(n_neighbors=5)
}

# ─────────────────────────────────────────────
# STEP 5: TRAIN, EVALUATE, PRINT RESULTS TABLE
# ─────────────────────────────────────────────
results    = {}
cv_results = {}

print("=" * 65)
print(f"{'MODEL':<22} {'ACC':>6} {'PREC':>6} {'REC':>6} {'F1':>6} {'AUC':>6}")
print("=" * 65)

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred      = model.predict(X_test_scaled)
    y_prob      = model.predict_proba(X_test_scaled)[:, 1]

    acc   = accuracy_score(y_test, y_pred)
    prec  = precision_score(y_test, y_pred)
    rec   = recall_score(y_test, y_pred)
    f1    = f1_score(y_test, y_pred)
    auc   = roc_auc_score(y_test, y_prob)

    results[name] = {
        'model': model, 'y_pred': y_pred, 'y_prob': y_prob,
        'accuracy': acc, 'precision': prec,
        'recall': rec, 'f1': f1, 'auc': auc
    }

    # 5-fold cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    cv_results[name] = cv_scores

    print(f"{name:<22} {acc:>6.4f} {prec:>6.4f} {rec:>6.4f} {f1:>6.4f} {auc:>6.4f}")

print("=" * 65)

# ─────────────────────────────────────────────
# STEP 6: CROSS-VALIDATION TABLE
# ─────────────────────────────────────────────
print("\n5-Fold Cross-Validation Accuracy:")
print("-" * 45)
for name, scores in cv_results.items():
    print(f"{name:<22}  Mean: {scores.mean():.4f}  Std: ±{scores.std():.4f}")
print("-" * 45)

# ─────────────────────────────────────────────
# STEP 7: CLASSIFICATION REPORTS
# ─────────────────────────────────────────────
print("\nDetailed Classification Reports:")
for name, res in results.items():
    print(f"\n{'─'*40}")
    print(f"  {name}")
    print('─'*40)
    print(classification_report(y_test, res['y_pred'],
                                 target_names=['Benign (0)', 'Malignant (1)']))

# ─────────────────────────────────────────────
# FIGURE 8, 9, 10 — CONFUSION MATRICES
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Figures 8–10: Confusion Matrices for All Three Classifiers',
             fontsize=14, fontweight='bold')

for ax, (name, res), color in zip(axes, results.items(), MODEL_COLORS):
    cm = confusion_matrix(y_test, res['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'],
                linewidths=1, linecolor='white',
                annot_kws={'size': 14, 'weight': 'bold'},
                ax=ax)
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=10)
    ax.set_ylabel('True Label', fontsize=10)

    # Annotate TP, TN, FP, FN
    tn, fp, fn, tp = cm.ravel()
    ax.text(0.5, -0.18, f'TP={tp}  TN={tn}  FP={fp}  FN={fn}',
            transform=ax.transAxes, ha='center', fontsize=9, color='#555')

plt.tight_layout()
plt.savefig('fig8_9_10_confusion_matrices.png', dpi=180, bbox_inches='tight')
plt.show()
print("✅ Confusion matrices saved.")

# ─────────────────────────────────────────────
# FIGURE 11 — ROC CURVES (all 3 on same plot)
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
fig.suptitle('Figure 11: ROC Curves — All Classifiers',
             fontsize=14, fontweight='bold')

for (name, res), color in zip(results.items(), MODEL_COLORS):
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    ax.plot(fpr, tpr, color=color, lw=2,
            label=f"{name} (AUC = {res['auc']:.4f})")

# Random classifier baseline
ax.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Random Classifier (AUC = 0.50)')

ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate (Recall)', fontsize=12)
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.legend(loc='lower right', fontsize=11)
ax.fill_between([0, 1], [0, 1], alpha=0.05, color='grey')

plt.tight_layout()
plt.savefig('fig11_roc_curves.png', dpi=180, bbox_inches='tight')
plt.show()
print("✅ ROC curves saved.")

# ─────────────────────────────────────────────
# FIGURE 12 — FEATURE IMPORTANCE (Random Forest)
# ─────────────────────────────────────────────
rf_model       = results['Random Forest']['model']
importances    = rf_model.feature_importances_
feature_names  = X.columns.tolist()

feat_df = pd.DataFrame({
    'Feature':    feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False).head(15)

fig, ax = plt.subplots(figsize=(11, 7))
fig.suptitle('Figure 12: Top 15 Feature Importances — Random Forest',
             fontsize=14, fontweight='bold')

bars = ax.barh(feat_df['Feature'][::-1], feat_df['Importance'][::-1],
               color='#5B8FD4', edgecolor='white', linewidth=0.8)

for bar, val in zip(bars, feat_df['Importance'][::-1]):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=9)

ax.set_xlabel('Mean Decrease in Gini Impurity', fontsize=11)
ax.set_ylabel('Feature', fontsize=11)
ax.set_xlim(0, feat_df['Importance'].max() * 1.18)

plt.tight_layout()
plt.savefig('fig12_feature_importance.png', dpi=180, bbox_inches='tight')
plt.show()
print("✅ Feature importance chart saved.")

# ─────────────────────────────────────────────
# FIGURE 13 — CV SCORE COMPARISON (Bonus)
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle('Figure 13: 5-Fold Cross-Validation Accuracy Comparison',
             fontsize=14, fontweight='bold')

cv_means = [cv_results[n].mean() for n in MODEL_NAMES]
cv_stds  = [cv_results[n].std()  for n in MODEL_NAMES]

bars = ax.bar(MODEL_NAMES, cv_means, yerr=cv_stds,
              color=MODEL_COLORS, width=0.4,
              error_kw=dict(elinewidth=2, capsize=6, ecolor='black'),
              edgecolor='white', linewidth=1.5)

for bar, mean, std in zip(bars, cv_means, cv_stds):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + std + 0.003,
            f'{mean:.4f}', ha='center', fontsize=11, fontweight='bold')

ax.set_ylim(0.88, 1.02)
ax.set_ylabel('Mean CV Accuracy', fontsize=12)
ax.set_xlabel('Classifier', fontsize=12)

plt.tight_layout()
plt.savefig('fig13_cv_comparison.png', dpi=180, bbox_inches='tight')
plt.show()
print("✅ Cross-validation comparison saved.")

print("\n🎉 All result figures saved!")
print("📋 Copy the printed numbers into Tables 1 and 2 in your results section.")
