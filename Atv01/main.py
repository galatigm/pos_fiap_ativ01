# ==============================================
# SISTEMA DE DIAGNÓSTICO DE CÂNCER DE MAMA
# Tech Challenge - Fase 1
# ==============================================

# ==============================================
# 1. IMPORTAÇÃO DAS BIBLIOTECAS
# ==============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, roc_curve)

# Configurações de visualização
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)

print("="*80)
print(" SISTEMA DE DIAGNÓSTICO DE CÂNCER DE MAMA ".center(80, "="))
print("="*80)

# ==============================================
# 2. EXPLORAÇÃO DE DADOS
# ==============================================
print("\n" + "="*80)
print(" ETAPA 1: EXPLORAÇÃO DE DADOS ".center(80))
print("="*80)

print("\n" + "="*80)
print(" 1.1 CARREGAMENTO E EXPLORAÇÃO INICIAL ".center(80))
print("="*80)

df = pd.read_csv('Atv01/data.csv')
print(f"\nDados carregados: {df.shape[0]} linhas x {df.shape[1]} colunas")
print("\nPrimeiras 5 linhas do dataset:")
print(df.head())
print("\nInformações sobre as colunas:")
print(df.info())
print("\nEstatísticas descritivas das features numéricas:")
print(df.describe())

print("\n" + "="*80)
print(" 1.2. ANÁLISE DE QUALIDADE DOS DADOS ".center(80))
print("="*80)

print("\nVerificação de valores ausentes:")
missing = df.isnull().sum()
if missing.sum() > 0:
    print("Valores ausentes por coluna:")
    print(missing[missing > 0])
else:
    print("Nenhum valor ausente encontrado.")

duplicatas = df.duplicated().sum()
print(f"\nLinhas duplicadas: {duplicatas}")

# ==============================================
# 3. PRÉ-PROCESSAMENTO
# ==============================================
print("\n" + "="*80)
print(" ETAPA 2: PRÉ-PROCESSAMENTO DOS DADOS ".center(80))
print("="*80)

print("\n" + "="*80)
print(" 2.1. LIMPEZA DOS DADOS ".center(80))
print("="*80)

print("\nRemoção de colunas desnecessárias:")
df = df.drop(['Unnamed: 32', 'id'], axis=1, errors='ignore')
print(f"Colunas 'id' e 'Unnamed: 32' removidas")
print(f"Dimensões atualizadas: {df.shape[0]} linhas x {df.shape[1]} colunas")

print("\n" + "="*80)
print(" 2.2. ANÁLISE DA VARIÁVEL TARGET ".center(80))
print("="*80)

print("\nDistribuição da variável 'diagnosis':")
print(df['diagnosis'].value_counts())

print("\nProporção:")
for label, prop in (df['diagnosis'].value_counts(normalize=True) * 100).items():
    nome = 'Benigno' if label == 'B' else 'Maligno'
    print(f"  {label} ({nome}): {prop:.2f}%")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
target_counts = df['diagnosis'].value_counts()
target_counts.plot(kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c'])
axes[0].set_title('Distribuição dos Diagnósticos', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Diagnóstico')
axes[0].set_ylabel('Quantidade')
axes[0].set_xticklabels(['Benigno', 'Maligno'], rotation=0)
axes[0].grid(axis='y', alpha=0.3)

axes[1].pie(target_counts, labels=['Benigno', 'Maligno'], 
            autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'], startangle=90)
axes[1].set_title('Proporção dos Diagnósticos', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('01_distribuicao_target.png', dpi=300, bbox_inches='tight')
plt.show()

benignos = (df['diagnosis'] == 'B').sum()
malignos = (df['diagnosis'] == 'M').sum()
print(f"\nBalanceamento do dataset:")
print(f"  Benignos (B): {benignos} ({benignos/len(df)*100:.1f}%)")
print(f"  Malignos (M): {malignos} ({malignos/len(df)*100:.1f}%)")

print("\n" + "="*80)
print(" 2.3. CONVERSÃO DE VARIÁVEIS CATEGÓRICAS ".center(80))
print("="*80)

print("\nCodificação da variável 'diagnosis':")
df['diagnosis'] = LabelEncoder().fit_transform(df['diagnosis'])
print("Mapeamento realizado:")
print("  B (Benigno) -> 0")
print("  M (Maligno) -> 1")

print("\n" + "="*80)
print(" 2.4. ANÁLISE DE DISTRIBUIÇÕES ".center(80))
print("="*80)

print("\nAnalisando distribuições das principais features...")

mean_features = [col for col in df.columns if 'mean' in col][:6]
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for idx, col in enumerate(mean_features):
    axes[idx].hist([df[df['diagnosis']==0][col], df[df['diagnosis']==1][col]], 
                   bins=20, label=['Benigno', 'Maligno'], 
                   color=['#2ecc71', '#e74c3c'], alpha=0.7)
    axes[idx].set_title(col, fontweight='bold')
    axes[idx].set_xlabel('Valor')
    axes[idx].set_ylabel('Frequência')
    axes[idx].legend()
    axes[idx].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('02_distribuicoes_features.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nObservações:")
print("- As features apresentam separação visível entre as classes")
print("- Tumores malignos tendem a ter valores mais altos nas características geométricas")

print("\n" + "="*80)
print(" 2.5. ANÁLISE DE CORRELAÇÃO ".center(80))
print("="*80)

correlation_matrix = df.corr()
target_corr = correlation_matrix['diagnosis'].sort_values(ascending=False)

print("\nTop 15 features mais correlacionadas com o diagnóstico:")
print(target_corr.head(16)[1:])

mean_cols = ['diagnosis'] + mean_features
plt.figure(figsize=(10, 8))
sns.heatmap(df[mean_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={'label': 'Correlação'})
plt.title('Matriz de Correlação - Features Principais', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('03_matriz_correlacao.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nPrincipais observações:")
print("- Features geométricas (radius, perimeter, area) com alta correlação entre si")
print("- Concave points e concavity mostram forte correlação com malignidade")

# ==============================================
# 4. DIVISÃO E NORMALIZAÇÃO
# ==============================================
print("\n" + "="*80)
print(" ETAPA 3: DIVISÃO E NORMALIZAÇÃO ".center(80))
print("="*80)

X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

print(f"\nFeatures (X): {X.shape[1]} variáveis")
print(f"Target (y): {y.shape[0]} amostras")

print("\nDivisão: 70% Treino | 15% Validação | 15% Teste")

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"\nTREINO: {X_train.shape[0]} amostras ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"VALIDAÇÃO: {X_val.shape[0]} amostras ({X_val.shape[0]/len(X)*100:.1f}%)")
print(f"TESTE: {X_test.shape[0]} amostras ({X_test.shape[0]/len(X)*100:.1f}%)")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("\nStandardScaler aplicado com sucesso")

# ==============================================
# 5. MODELAGEM
# ==============================================
print("\n" + "="*80)
print(" ETAPA 4: TREINAMENTO DOS MODELOS ".center(80))
print("="*80)

models = {}
results_val = []
results_test = []

# SVM
print("\n" + "-"*80)
print(" MODELO 1: SVM ".center(80))
print("-"*80)

svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, probability=True)
svm_model.fit(X_train_scaled, y_train)
models['SVM'] = svm_model

y_pred_svm_val = svm_model.predict(X_val_scaled)
y_prob_svm_val = svm_model.predict_proba(X_val_scaled)[:, 1]

print("\nVALIDAÇÃO:")
print(f"Accuracy: {accuracy_score(y_val, y_pred_svm_val):.4f}")
print(f"Precision: {precision_score(y_val, y_pred_svm_val):.4f}")
print(f"Recall: {recall_score(y_val, y_pred_svm_val):.4f}")
print(f"F1-Score: {f1_score(y_val, y_pred_svm_val):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_val, y_prob_svm_val):.4f}")

results_val.append({
    'Modelo': 'SVM',
    'Accuracy': accuracy_score(y_val, y_pred_svm_val),
    'Precision': precision_score(y_val, y_pred_svm_val),
    'Recall': recall_score(y_val, y_pred_svm_val),
    'F1-Score': f1_score(y_val, y_pred_svm_val),
    'ROC-AUC': roc_auc_score(y_val, y_prob_svm_val)
})

y_pred_svm_test = svm_model.predict(X_test_scaled)
y_prob_svm_test = svm_model.predict_proba(X_test_scaled)[:, 1]

print("\nTESTE:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_svm_test):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_svm_test):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_svm_test):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_svm_test):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_svm_test):.4f}")

results_test.append({
    'Modelo': 'SVM',
    'Accuracy': accuracy_score(y_test, y_pred_svm_test),
    'Precision': precision_score(y_test, y_pred_svm_test),
    'Recall': recall_score(y_test, y_pred_svm_test),
    'F1-Score': f1_score(y_test, y_pred_svm_test),
    'ROC-AUC': roc_auc_score(y_test, y_prob_svm_test)
})

# Random Forest
print("\n" + "-"*80)
print(" MODELO 2: RANDOM FOREST ".center(80))
print("-"*80)

rf_model = RandomForestClassifier(
    n_estimators=100, max_depth=10, min_samples_split=5,
    min_samples_leaf=2, random_state=42
)
rf_model.fit(X_train_scaled, y_train)
models['Random Forest'] = rf_model

y_pred_rf_val = rf_model.predict(X_val_scaled)
y_prob_rf_val = rf_model.predict_proba(X_val_scaled)[:, 1]

print("\nVALIDAÇÃO:")
print(f"Accuracy: {accuracy_score(y_val, y_pred_rf_val):.4f}")
print(f"Precision: {precision_score(y_val, y_pred_rf_val):.4f}")
print(f"Recall: {recall_score(y_val, y_pred_rf_val):.4f}")
print(f"F1-Score: {f1_score(y_val, y_pred_rf_val):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_val, y_prob_rf_val):.4f}")

results_val.append({
    'Modelo': 'Random Forest',
    'Accuracy': accuracy_score(y_val, y_pred_rf_val),
    'Precision': precision_score(y_val, y_pred_rf_val),
    'Recall': recall_score(y_val, y_pred_rf_val),
    'F1-Score': f1_score(y_val, y_pred_rf_val),
    'ROC-AUC': roc_auc_score(y_val, y_prob_rf_val)
})

y_pred_rf_test = rf_model.predict(X_test_scaled)
y_prob_rf_test = rf_model.predict_proba(X_test_scaled)[:, 1]

print("\nTESTE:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf_test):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_rf_test):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_rf_test):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_rf_test):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_rf_test):.4f}")

results_test.append({
    'Modelo': 'Random Forest',
    'Accuracy': accuracy_score(y_test, y_pred_rf_test),
    'Precision': precision_score(y_test, y_pred_rf_test),
    'Recall': recall_score(y_test, y_pred_rf_test),
    'F1-Score': f1_score(y_test, y_pred_rf_test),
    'ROC-AUC': roc_auc_score(y_test, y_prob_rf_test)
})

# ==============================================
# 6. COMPARAÇÃO
# ==============================================
print("\n" + "="*80)
print(" ETAPA 5: COMPARAÇÃO DOS MODELOS ".center(80))
print("="*80)

comparison_val_df = pd.DataFrame(results_val)
comparison_test_df = pd.DataFrame(results_test)

print("\nVALIDAÇÃO:")
print(comparison_val_df.to_string(index=False))

print("\nTESTE:")
print(comparison_test_df.to_string(index=False))

# Gráficos de comparação
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
x_pos = np.arange(len(metrics))
width = 0.35

svm_val = [comparison_val_df.loc[0, m] for m in metrics]
rf_val = [comparison_val_df.loc[1, m] for m in metrics]

axes[0].bar(x_pos - width/2, svm_val, width, label='SVM', color='#3498db')
axes[0].bar(x_pos + width/2, rf_val, width, label='Random Forest', color='#2ecc71')
axes[0].set_title('Validação', fontweight='bold')
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(metrics, rotation=45, ha='right')
axes[0].legend()
axes[0].set_ylim([0, 1.1])
axes[0].grid(axis='y', alpha=0.3)

svm_test = [comparison_test_df.loc[0, m] for m in metrics]
rf_test = [comparison_test_df.loc[1, m] for m in metrics]

axes[1].bar(x_pos - width/2, svm_test, width, label='SVM', color='#3498db')
axes[1].bar(x_pos + width/2, rf_test, width, label='Random Forest', color='#2ecc71')
axes[1].set_title('Teste', fontweight='bold')
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(metrics, rotation=45, ha='right')
axes[1].legend()
axes[1].set_ylim([0, 1.1])
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('04_comparacao_modelos.png', dpi=300, bbox_inches='tight')
plt.show()

# ==============================================
# 7. MATRIZES DE CONFUSÃO
# ==============================================
print("\n" + "="*80)
print(" ETAPA 6: MATRIZES DE CONFUSÃO ".center(80))
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

cm_svm_val = confusion_matrix(y_val, y_pred_svm_val)
sns.heatmap(cm_svm_val, annot=True, fmt='d', cmap='Blues', ax=axes[0,0],
            xticklabels=['Benigno', 'Maligno'], yticklabels=['Benigno', 'Maligno'])
axes[0,0].set_title('SVM - Validação', fontweight='bold')

cm_svm_test = confusion_matrix(y_test, y_pred_svm_test)
sns.heatmap(cm_svm_test, annot=True, fmt='d', cmap='Blues', ax=axes[0,1],
            xticklabels=['Benigno', 'Maligno'], yticklabels=['Benigno', 'Maligno'])
axes[0,1].set_title('SVM - Teste', fontweight='bold')

cm_rf_val = confusion_matrix(y_val, y_pred_rf_val)
sns.heatmap(cm_rf_val, annot=True, fmt='d', cmap='Greens', ax=axes[1,0],
            xticklabels=['Benigno', 'Maligno'], yticklabels=['Benigno', 'Maligno'])
axes[1,0].set_title('Random Forest - Validação', fontweight='bold')

cm_rf_test = confusion_matrix(y_test, y_pred_rf_test)
sns.heatmap(cm_rf_test, annot=True, fmt='d', cmap='Greens', ax=axes[1,1],
            xticklabels=['Benigno', 'Maligno'], yticklabels=['Benigno', 'Maligno'])
axes[1,1].set_title('Random Forest - Teste', fontweight='bold')

plt.tight_layout()
plt.savefig('05_matrizes_confusao.png', dpi=300, bbox_inches='tight')
plt.show()

# ==============================================
# 8. CURVAS ROC
# ==============================================
print("\n" + "="*80)
print(" ETAPA 7: CURVAS ROC ".center(80))
print("="*80)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

fpr_svm_val, tpr_svm_val, _ = roc_curve(y_val, y_prob_svm_val)
fpr_rf_val, tpr_rf_val, _ = roc_curve(y_val, y_prob_rf_val)

axes[0].plot(fpr_svm_val, tpr_svm_val, 
             label=f'SVM (AUC = {roc_auc_score(y_val, y_prob_svm_val):.3f})', 
             color='#3498db', linewidth=2)
axes[0].plot(fpr_rf_val, tpr_rf_val, 
             label=f'Random Forest (AUC = {roc_auc_score(y_val, y_prob_rf_val):.3f})', 
             color='#2ecc71', linewidth=2)
axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1)
axes[0].set_title('Curva ROC - Validação', fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

fpr_svm_test, tpr_svm_test, _ = roc_curve(y_test, y_prob_svm_test)
fpr_rf_test, tpr_rf_test, _ = roc_curve(y_test, y_prob_rf_test)

axes[1].plot(fpr_svm_test, tpr_svm_test, 
             label=f'SVM (AUC = {roc_auc_score(y_test, y_prob_svm_test):.3f})', 
             color='#3498db', linewidth=2)
axes[1].plot(fpr_rf_test, tpr_rf_test, 
             label=f'Random Forest (AUC = {roc_auc_score(y_test, y_prob_rf_test):.3f})', 
             color='#2ecc71', linewidth=2)
axes[1].plot([0, 1], [0, 1], 'k--', linewidth=1)
axes[1].set_title('Curva ROC - Teste', fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('06_curvas_roc.png', dpi=300, bbox_inches='tight')
plt.show()

# ==============================================
# 9. CLASSIFICATION REPORTS
# ==============================================
print("\n" + "="*80)
print(" ETAPA 8: CLASSIFICATION REPORTS ".center(80))
print("="*80)

print("\nSVM - TESTE:")
print(classification_report(y_test, y_pred_svm_test, target_names=['Benigno', 'Maligno']))

print("\nRANDOM FOREST - TESTE:")
print(classification_report(y_test, y_pred_rf_test, target_names=['Benigno', 'Maligno']))

# ==============================================
# 10. FEATURE IMPORTANCE
# ==============================================
print("\n" + "="*80)
print(" ETAPA 9: FEATURE IMPORTANCE ".center(80))
print("="*80)

feature_imp = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 15 features:")
print(feature_imp.head(15).to_string(index=False))

plt.figure(figsize=(10, 8))
top_feat = feature_imp.head(15)
plt.barh(range(len(top_feat)), top_feat['Importance'], color='#2ecc71')
plt.yticks(range(len(top_feat)), top_feat['Feature'])
plt.xlabel('Importância')
plt.title('Top 15 Features - Random Forest', fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('07_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print(" ANÁLISE CONCLUÍDA ".center(80))
print("="*80)