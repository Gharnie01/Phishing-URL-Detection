                                ##################################################
                                #####                                        #####
                                ##### EDA, FEATURE ENGINEERING AND, MODELING #####
                                #####                                        #####
                                ##################################################


import pandas as pd
import matplotlib.pyplot as plt

# Load Dataset
manual_df = pd.read_csv('data/url_dataset.csv')
label_counts = manual_df['type'].value_counts()

# original dataset label distribution Bar Chart
plt.figure()
label_counts.plot(kind='bar', color=['skyblue', 'lightcoral'])
for i, value in enumerate(label_counts.values):
    plt.text(i, value, str(value), ha='center', va='bottom')
plt.title('Original Dataset label Distribution')
plt.xlabel('Labels (0 = Benign, 1 = Phishing)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('output/original_dataset_distribution_bar.png', dpi=500) 
plt.close()

# Load preprocessed Dataset
turl_df = pd.read_csv('output/NEWFSFiles/final_selected_features_data.csv')
label_counts = turl_df['phishing'].value_counts()

# Preprocessed dataset label distribution Bar Chart
plt.figure()
label_counts.plot(kind='bar', color=['green', 'red'])  #Alternate colors to display as you want for which label
for i, value in enumerate(label_counts.values):
    plt.text(i, value, str(value), ha='center', va='bottom')
plt.title('Preprocessed Dataset label Distribution')
plt.xlabel('Labels (0 = Benign, 1 = Phishing)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('output/preprocessed_datset_distribution_bar.png', dpi=500)
plt.close()

import seaborn as sns
# Correlation heatmap to identify highly correlated or low-variance features
tmp_corr = turl_df.drop(columns=['url','phishing']).corr()
plt.figure(figsize=(20,18))
sns.heatmap(tmp_corr, cmap='coolwarm', center=0)
plt.xticks(fontsize=14) 
plt.yticks(fontsize=14)
plt.title('Feature Correlation Matrix of all manual input features')
plt.savefig('output/manual_corr_heatmap_after_30f_overlap.png', dpi=500)
plt.close()

from wordcloud import WordCloud
import tldextract
# Word clouds
all_urls = ' '.join(turl_df['url'])
wc = WordCloud(width=800, height=400).generate(all_urls)
plt.figure(figsize=(10,5))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of URLs')
plt.savefig('output/url_wordcloud.png', dpi=500)
plt.close()
#Domain word cloud
domains = turl_df['url'].apply(lambda u: tldextract.extract(u).domain)
wc2 = WordCloud(width=800, height=400).generate(' '.join(domains))
plt.figure(figsize=(10,5))
plt.imshow(wc2, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Domains')
plt.savefig('output/domain_wordcloud.png', dpi=500)
plt.close()

#Dataframe 
turl_df['url_length'] = turl_df['url'].str.len()

# Separate phishing and legitimate URLs based on 'phishing' column
phishing_df = turl_df[turl_df['phishing'] == 'phishing']
legitimate_df = turl_df[turl_df['phishing'] == 'legitimate']

# Plotting Density Plot
plt.figure(figsize=(10, 6))
sns.kdeplot(legitimate_df['url_length'], fill=True, color='blue', label='Legitimate', alpha=0.6)
sns.kdeplot(phishing_df['url_length'], fill=True, color='orange', label='Phishing', alpha=0.6)
# Customizing the plot
plt.title('Density Plot of URL Length for Legitimate and Phishing URLs')
plt.xlabel('URL Length')
plt.ylabel('Density')
plt.legend()
plt.xlim(0, 500)
plt.tight_layout()
# Save the plot
plt.savefig('output/url_length_density_plot.png', dpi=500)
plt.show()

import numpy as np

# Calculate URL Lengths Histograms
# =========================
urls = turl_df['url'].tolist()
url_lengths = [len(url) for url in urls]
min_length = min(url_lengths)
max_length = max(url_lengths)
avg_length = sum(url_lengths) / len(url_lengths)
# Plotting Histogram
plt.figure(figsize=(12, 6))
sns.histplot(url_lengths, bins=30, kde=True, color='skyblue', edgecolor='black')
# Add Average, Min, Max Lines
plt.axvline(avg_length, color='green', linestyle='--', label=f'Average Length: {avg_length:.2f}')
plt.axvline(min_length, color='red', linestyle='--', label=f'Min Length: {min_length}')
plt.axvline(max_length, color='blue', linestyle='--', label=f'Max Length: {max_length}')
# Annotate bars
counts, edges = np.histogram(url_lengths, bins=30)
bin_centers = 0.5 * (edges[1:] + edges[:-1])
for count, x in zip(counts, bin_centers):
    if count > 0:
        plt.text(x, count + 2, str(count), fontsize=9, ha='center')
# Graph Customization
plt.title('Distribution of URL Lengths with Annotations')
plt.xlabel('URL Length')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
# Save and Display the Plot
plt.savefig('output/url_length_distribution.png', dpi=1000)
plt.show()

# =========================
# Scaling Features
# =========================
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1. Load the dataset (replace 'dataset.csv' with your file)
turl_df = pd.read_csv('data/features.csv')

# 2. Extract columns to preserve: 'url' and 'phishing'
url_column = turl_df['url']
label_column = turl_df['phishing']

# 3. Extract features (drop 'url' and 'phishing' columns)
features = turl_df.drop(['url', 'phishing'], axis=1)

# 4. Apply Standard Scaling (Z-score normalization)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# 5. Convert scaled features back to DataFrame
scaled_df = pd.DataFrame(scaled_features, columns=features.columns)

# 6. Reattach 'url' and 'phishing' columns
final_df = pd.concat([url_column, scaled_df, label_column], axis=1)

# 7. Save to 'scaled.csv' (with headers)
final_df.to_csv('output/scaled.csv', index=False)

print("Scaling complete! Output saved to 'scaled.csv'.")

# =========================
# Feature Engineering
# =========================

# 1. Libraries and Dependencies
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 2. Load Dataset
turl_df = pd.read_csv('data/scaled.csv')
X = turl_df.drop(columns=['url', 'phishing']).values
y = turl_df['phishing'].values
feature_names = turl_df.drop(columns=['url', 'phishing']).columns

# 3. Build MLP Model Function (for Feature Evaluation)
def build_mlp(input_dim):
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def evaluate_features(X_selected, y):
    """Performs 5-fold CV with the selected features and returns average F1 score."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    metrics_list = []

    for train_index, test_index in skf.split(X_selected, y):
        X_tr, X_te = X_selected[train_index], X_selected[test_index]
        y_tr, y_te = y[train_index], y[test_index]

        mlp = build_mlp(X_tr.shape[1])
        mlp.fit(X_tr, y_tr, epochs=10, batch_size=32, verbose=0)
        y_pred = (mlp.predict(X_te).ravel() > 0.5).astype(int)

        # Calculate F1 score
        f1 = f1_score(y_te, y_pred)
        metrics_list.append(f1)

    return np.mean(metrics_list)

# =========================
# 4. Optimizing Variance Threshold
# =========================
var_thresholds = [0.0, 0.001, 0.01, 0.02, 0.05]
var_scores = []

for threshold in var_thresholds:
    vt = VarianceThreshold(threshold=threshold)
    X_vt = vt.fit_transform(X)
    score = evaluate_features(X_vt, y)
    var_scores.append(score)
    print(f"Variance Threshold {threshold} - Features: {X_vt.shape[1]}, Avg F1: {score:.4f}")

# Best Variance Threshold
best_var_idx = np.argmax(var_scores)
best_var_threshold = var_thresholds[best_var_idx]
vt = VarianceThreshold(threshold=best_var_threshold)
X_vt = vt.fit_transform(X)
best_var_features = feature_names[vt.get_support(indices=True)]

plt.figure()
plt.plot(var_thresholds, var_scores, marker='o')
plt.title("Variance Threshold Optimization")
plt.xlabel("Threshold")
plt.ylabel("Avg F1 Score")
plt.savefig('output/variance_threshold_optimization.png', dpi=1000)

# =========================
# 5. Optimizing RFE
# =========================
feature_percentages = [0.3, 0.5, 0.7, 0.9]
rfe_scores = []
rfe_objects = []

for pct in feature_percentages:
    n_features = int(pct * X.shape[1])
    rfe_model = LogisticRegression(max_iter=500, random_state=42)
    rfe = RFE(estimator=rfe_model, n_features_to_select=n_features)
    X_rfe = rfe.fit_transform(X, y)
    score = evaluate_features(X_rfe, y)
    rfe_scores.append(score)
    rfe_objects.append(rfe)
    print(f"RFE with {n_features} features - Avg F1: {score:.4f}")

# Best RFE Feature Count
best_rfe_idx = np.argmax(rfe_scores)
best_rfe_pct = feature_percentages[best_rfe_idx]
best_rfe = rfe_objects[best_rfe_idx]
best_rfe_features = feature_names[best_rfe.get_support(indices=True)]

plt.figure()
plt.plot([int(pct * 100) for pct in feature_percentages], rfe_scores, marker='o')
plt.title("RFE Feature Optimization")
plt.xlabel("Percentage of Features")
plt.ylabel("Avg F1 Score")
plt.savefig('output/rfe_optimization.png', dpi=1000)

# =========================
# 6. Overlapping Features (Final Selection)
# =========================

# Display and save optimal features from Variance Threshold
print("\nOptimal Variance Threshold Features:")
print(f"Optimal Variance Threshold: {best_var_threshold}")
print(f"Number of Features Retained: {len(best_var_features)}")
pd.DataFrame({'Variance_Features': best_var_features}).to_csv('output/optimal_variance_features.csv', index=False)

# Display and save optimal features from RFE
print("\nOptimal RFE Features:")
print(f"Best RFE percentage: {int(best_rfe_pct * 100)}%")
print(f"Number of Features Retained: {len(best_rfe_features)}")
pd.DataFrame({'RFE_Features': best_rfe_features}).to_csv('output/optimal_rfe_features.csv', index=False)

# Calculate overlapping features
overlapping_features = list(set(best_var_features).intersection(set(best_rfe_features)))
print("\nFinal Overlapping Features (Retained for Final Model):")
print(f"Number of Overlapping Features: {len(overlapping_features)}")
pd.DataFrame({'Final_Selected_Features': overlapping_features}).to_csv('output/final_selected_features.csv', index=False)

# Save Overlapping Features with 'url' and 'phishing' columns
final_columns = ['url'] + overlapping_features + ['phishing']
final_X_df = turl_df[final_columns]
final_X_df.to_csv('output/final_selected_features_data.csv', index=False)

from matplotlib_venn import venn2
import matplotlib.pyplot as plt

# Create sets
var_set = set(best_var_features)
rfe_set = set(best_rfe_features)

# Plot Venn Diagram
plt.figure(figsize=(6, 6))
venn2([var_set, rfe_set], set_labels=('VarianceThreshold', 'RFE'), set_colors=('skyblue', 'lightcoral'), alpha=0.6)
plt.title("Venn Diagram of Selected Features")
plt.tight_layout()
plt.savefig('output/venn_feature_overlap.png', dpi=500)
plt.show()


import seaborn as sns
import pandas as pd

# Create DataFrame to show presence (1 = selected, 0 = not selected)
all_features = list(set(best_var_features).union(set(best_rfe_features)))
heatmap_data = pd.DataFrame({
    'VarianceThreshold': [1 if feat in var_set else 0 for feat in all_features] ,
    'RFE': [1 if feat in rfe_set else 0 for feat in all_features]
}, index=all_features)

# Plot Heatmap
plt.figure(figsize=(10, len(all_features) * 0.3))  # Dynamic height
sns.heatmap(heatmap_data, cmap='Blues', cbar=False, linewidths=0.5, linecolor='gray', annot=True, annot_kws={"size": 16})
plt.title("Feature Selection Heatmap (RFE vs VarianceThreshold)")
plt.xlabel("Selection Method", fontsize=18)
plt.ylabel("Feature Name", fontsize=18)
plt.xticks(fontsize=16) 
plt.yticks(fontsize=16)  # or any size you prefer
plt.tight_layout()
plt.savefig('output/feature_selection_heatmap.png', dpi=500)
plt.show()





            #========================= ABLATION STUDY =================#
            # Ablation Study: Evaluating the Impact of Manual Features #
            # =========================================================#

'''Ablation Study: Investigating the  performance of baseline models in comparison with
the proposed model for phishing detection.Aslo how it improves on the baseline models
(BERT-only model, manual features-only model, and TCN embeddings with manual features.)'''


                            # ===========================================#
                            # Proposed Classification Model (Full model) #
                            # ===========================================#
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np
import tensorflow as tf
import seaborn as sns
import pandas as pd
import time
import os 
import random
import matplotlib.pyplot as plt

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.keras.backend.clear_session()

# 🎯 Set Random Seed for Reproducibility
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Mixed Precision Policy
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# TUNING 6: Added L2 regularization
# Define L2 regularizer
l2_reg = tf.keras.regularizers.l2(1e-4)

# ============== Load Data ===================
df = pd.read_csv('data/final_selected_features_data.csv')
urls = df['url'].tolist()
manual_features = df.drop(columns=['url', 'phishing']).values.astype(np.float32)
labels = df['phishing'].values.astype(np.float32)
assert len(np.unique(labels)) == 2, "Binary classification expected with 2 label classes."

# # ============== BERT Embeddings =============
bert_embeddings = np.load('Embeddings/cls_embeddings.npy') 
print("Shape:", bert_embeddings.shape) # Load precomputed BERT embeddings if available
X_ids = bert_embeddings.astype(np.float32)
X_manual = manual_features
y = labels

# ============== Train-Test Split ==============
X_ids_train, X_ids_test, X_manual_train, X_manual_test, y_train, y_test = train_test_split(
    X_ids, X_manual, y, test_size=0.2, random_state=42, stratify=y
)

# ============== Cross-Validation Setup ==============
NUM_FOLDS = 4
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
train_metrics = []
fold_histories = []

# ============== Training Loop ===================
for fold, (train_idx, val_idx) in enumerate(skf.split(X_ids_train, y_train), 1):
    print(f"\n📦 Fold {fold}")

    ids_tr, ids_val = X_ids_train[train_idx], X_ids_train[val_idx]
    man_tr, man_val = X_manual_train[train_idx], X_manual_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    # Define inputs
    input_bert = tf.keras.Input(shape=(768,), dtype=tf.float32, name="bert_embeddings")
    input_manual = tf.keras.Input(shape=(X_manual.shape[1],), dtype=tf.float32, name="manual_features")
    #TUNING 1: Adjusted layers from 2 to 3 with 2 dropout layer of 0.4
    #monitor training and test results to avoid overfitting NB: try Dropout(0.5)
    # Process BERT path
    x_bert = tf.keras.layers.Dense(512, use_bias=False, kernel_regularizer=l2_reg)(input_bert)
    x_bert = tf.keras.layers.BatchNormalization()(x_bert)
    x_bert = tf.keras.layers.Activation('relu')(x_bert)
    x_bert = tf.keras.layers.Dropout(0.3)(x_bert)
    x_bert = tf.keras.layers.Dense(256, use_bias=False, kernel_regularizer=l2_reg)(x_bert)
    x_bert = tf.keras.layers.BatchNormalization()(x_bert)
    x_bert = tf.keras.layers.Activation('relu')(x_bert)
    x_bert = tf.keras.layers.Dropout(0.3)(x_bert)
    # Normalize manual features (but no dropout/dense)
    x_manual = tf.keras.layers.BatchNormalization()(input_manual)
    # Concatenate processed BERT + normalized manual features
    x = tf.keras.layers.Concatenate()([x_bert, x_manual])
    # Final classification layers
    x = tf.keras.layers.Dense(64, use_bias=False, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32')(x)

    # Define the model
    model = tf.keras.Model(inputs=[input_bert, input_manual], outputs=output)
    #TUNING: 1e-5, 2e-5, 3e-5
    model.compile(optimizer=tf.keras.optimizers.Adam(5e-5), loss='binary_crossentropy', metrics=['accuracy'])
    # TUNING 2: EarlyStopping callback (initialized inside the loop) adjusted epoch, 
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint('proposed_model.h5', monitor='val_loss', save_best_only=True)

    # ⏱️ Record prediction time for validation set (optional)
    train_start = time.time()
    # TUNING 3: Increased epochs from 5 to 10 and reducing batch size from 128 to 64
    # TUNING 4: Added ModelCheckpoint to save the best model
    # TUNING 5: Added EarlyStopping to prevent overfitting
    history = model.fit([ids_tr, man_tr], y_tr,
              validation_data=([ids_val, man_val], y_val),
              epochs=20, batch_size=64, verbose=1,callbacks=[early_stopping, checkpoint])
    fold_histories.append(history.history)

    # TUNING 4: Added ModelCheckpoint to save the best model
    # TUNING 5: Added EarlyStopping to prevent overfitting
    train_duration = time.time() - train_start
    val_start = time.time()
    y_pred = (model.predict([ids_val, man_val]) > 0.5).astype(int)
    val_duration = time.time() - val_start  # ⏱️ Prediction time (optional)
    metrics = {
        'Fold': f'Fold {fold}',
        'Accuracy': accuracy_score(y_val, y_pred),
        'Precision': precision_score(y_val, y_pred),
        'Recall': recall_score(y_val, y_pred),
        'F1 Score': f1_score(y_val, y_pred),
        'AUC-ROC': roc_auc_score(y_val, y_pred),
        'Training Time (s)': round(train_duration, 2),
        'Prediction Time (s)': round(val_duration, 2)    
    }
    train_metrics.append(metrics)

# ============== Save Training Metrics ==============
train_metrics_df = pd.DataFrame(train_metrics)
avg = train_metrics_df.mean(numeric_only=True)
avg['Fold'] = 'Average'
train_metrics_df = pd.concat([train_metrics_df, pd.DataFrame([avg])], ignore_index=True)
print(train_metrics_df)

# Save training metrics to CSV
def plot_fold_learning_curves(histories, metric, save_path, title_prefix):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.ravel()
    labels = ['(a)', '(b)', '(c)', '(d)']
    
    for i, history in enumerate(histories):
        axs[i].plot(history[metric], label='Training', color='blue')
        axs[i].plot(history[f'val_{metric}'], label='Validation', color='red')
        axs[i].set_title(f"{labels[i]} Fold {i+1} {title_prefix} Curve", loc='left', fontsize=10)
        axs[i].set_xlabel('Epochs')
        axs[i].set_ylabel(metric.capitalize())
        axs[i].legend()
        axs[i].grid(True, alpha=0.6)  # Can add this linestyle='--' for dotter lines if you prefer
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=1000)
    plt.close()
    print(f"✅ {title_prefix} curves saved to {save_path}")

# ============== Final Test Set Evaluation ==============
# ⏱️ Record prediction time on final test set
test_start = time.time()
y_test_probs = model.predict([X_ids_test, X_manual_test]).flatten()
y_test_pred = (y_test_probs > 0.5).astype(int)
test_duration = time.time() - test_start  # ⏱️ Test prediction time
test_metrics = {
    'Set': 'Test',
    'Accuracy': accuracy_score(y_test, y_test_pred),
    'Precision': precision_score(y_test, y_test_pred),
    'Recall': recall_score(y_test, y_test_pred),
    'F1 Score': f1_score(y_test, y_test_pred),
    'AUC-ROC': roc_auc_score(y_test, y_test_pred),
    'Prediction Time (s)': round(test_duration, 2)
}
test_metrics_df = pd.DataFrame([test_metrics])
print("\n🎯 Final Test Set Performance:")
print(test_metrics_df)

# ============== Save Visualizations ==============
def save_confusion_matrix(cm, path="output/confusion_matrix.png", title="Confusion Matrix"):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Legit', 'Phish'], yticklabels=['Legit', 'Phish'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=1000)
    plt.close()
    print(f"✅ Confusion matrix saved to {path}")

def save_metrics_table(df, path="output/metrics_table.png", title="Metrics Table"):
    fig, ax = plt.subplots(figsize=(max(6, df.shape[1]*1.5), max(1.5, df.shape[0]*0.5)))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.round(4).values,
                     colLabels=df.columns,
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=500)
    plt.close()
    print(f"✅ Table saved to {path}")
from sklearn.metrics import roc_curve

def save_roc_curve(y_true, y_probs, path="output/roc_curve.png", title="Receiver Operating Characteristic (ROC) Curve", model_name="DistilBERT"):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc_score = roc_auc_score(y_true, y_probs)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_score:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate/ Specificity")
    plt.ylabel("True Positive Rate/ Sensitivity")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=1000)
    plt.close()
    print(f"✅ ROC curve saved to {path}")

# Save results
save_metrics_table(train_metrics_df, "output/train_metrics_table.png", "Training Metrics (Cross-Validation)")
save_confusion_matrix(confusion_matrix(y_test, y_test_pred), "output/confusion_matrix_test.png")
save_metrics_table(test_metrics_df, "output/test_metrics_table.png", "Final Test Set Metrics")
plot_fold_learning_curves(fold_histories, 'accuracy', 'output/folds_accuracy_curves.png', 'Accuracy')
plot_fold_learning_curves(fold_histories, 'loss', 'output/folds_loss_curves.png', 'Loss')  
save_roc_curve(y_test, y_test_probs, "output/test_roc_curve.png")
np.save("output/proposed_model_y_test_true.npy", y_test)
np.save("output/proposed_model_y_test_pred.npy", y_test_pred)
np.save("output/proposed_model_y_test_probs.npy", y_test_probs)
np.save("output/proposed_model_fold_histories.npy", fold_histories)

######################################################################################################
#=== BASELINE: =======DISTILBERT WITH DEFAULT HEAD (W/o manual features + W/o custom head) ==========#
######################################################################################################
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import random
import time
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification



# Set random seed for reproducibility
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Set mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# ==================== Load and Prepare Data =====================
df = pd.read_csv('data/final_selected_features_data.csv')
urls = df['url'].tolist()
labels = df['phishing'].values.astype(np.float32)
assert len(np.unique(labels)) == 2, "Binary classification expected with 2 label classes."

# ==================== Tokenization =====================
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
encodings = tokenizer(urls, truncation=True, padding='max_length', max_length=64, return_tensors='np')
input_ids = encodings['input_ids']
attention_mask = encodings['attention_mask']

# ==================== Train-Test Split =====================
X_ids_train, X_ids_test, X_mask_train, X_mask_test, y_train, y_test = train_test_split(
    input_ids, attention_mask, labels, test_size=0.2, random_state=SEED, stratify=labels
)

# ==================== Cross-Validation Setup =====================
NUM_FOLDS = 4
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
train_metrics = []
fold_histories = []

from transformers import __version__ as transformers_version 
print("Transformers:", transformers_version)

# ==================== Training Loop =====================
for fold, (train_idx, val_idx) in enumerate(skf.split(X_ids_train, y_train), 1):
    print(f"\n📦 Fold {fold}")

    ids_tr, ids_val = X_ids_train[train_idx], X_ids_train[val_idx]
    mask_tr, mask_val = X_mask_train[train_idx], X_mask_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2,  # For binary classification
            problem_type="single_label_classification"
    )

    optimizer = tf.keras.optimizers.AdamW(learning_rate=4e-5)
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint('output/best_model_distilbert.keras', monitor='val_loss', save_best_only=True)
    
    train_start = time.time()
    history = model.fit(
        {'input_ids': ids_tr, 'attention_mask': mask_tr}, y_tr,
        validation_data=({'input_ids': ids_val, 'attention_mask': mask_val}, y_val),
        epochs=10, batch_size=32, callbacks=[early_stopping, checkpoint], verbose=2
    )
    fold_histories.append(history.history)
    train_duration = time.time() - train_start

    val_start = time.time()
    logits = model.predict({'input_ids': ids_val, 'attention_mask': mask_val}).logits
    probs = tf.nn.softmax(logits, axis=-1).numpy()
    y_val_pred = np.argmax(probs, axis=1)
    val_duration = time.time() - val_start

    metrics = {
        'Fold': f'Fold {fold}',
        'Accuracy': accuracy_score(y_val, y_val_pred),
        'Precision': precision_score(y_val, y_val_pred),
        'Recall': recall_score(y_val, y_val_pred),
        'F1 Score': f1_score(y_val, y_val_pred),
        'AUC-ROC': roc_auc_score(y_val, y_val_pred),
        'Training Time (s)': round(train_duration, 2),
        'Prediction Time (s)': round(val_duration, 2)
    }
    train_metrics.append(metrics)

# ==================== Save Training Metrics =====================
train_metrics_df = pd.DataFrame(train_metrics)
avg = train_metrics_df.mean(numeric_only=True)
avg['Fold'] = 'Average'
train_metrics_df = pd.concat([train_metrics_df, pd.DataFrame([avg])], ignore_index=True)
print(train_metrics_df)

# ==================== Final Test Set Evaluation =====================
test_start = time.time()
y_test_logits = model.predict({'input_ids': X_ids_test, 'attention_mask': X_mask_test}).logits
y_test_probs = tf.nn.softmax(y_test_logits, axis=-1).numpy()
y_test_pred = np.argmax(y_test_probs, axis=1)
test_duration = time.time() - test_start

test_metrics_df = pd.DataFrame([{
    'Set': 'Test',
    'Accuracy': accuracy_score(y_test, y_test_pred),
    'Precision': precision_score(y_test, y_test_pred),
    'Recall': recall_score(y_test, y_test_pred),
    'F1 Score': f1_score(y_test, y_test_pred),
    'AUC-ROC': roc_auc_score(y_test, y_test_probs[:, 1]),
    'Prediction Time (s)': round(test_duration, 2)
}])
print("\n🎯 Final Test Set Performance:")
print(test_metrics_df)

# ==================== Save Visualizations =====================
def save_confusion_matrix(cm, path="output/confusion_matrix_test_distilbert.png", title="Confusion Matrix"):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Legit', 'Phish'], yticklabels=['Legit', 'Phish'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=1000)
    plt.close()
    print(f"✅ Confusion matrix saved to {path}")

def save_roc_curve(y_true, y_probs, path="output/roc_curve_distilbert.png", title="Receiver Operating Characteristic (ROC) Curve", model_name="DistilBERT-DH"):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc_score = roc_auc_score(y_true, y_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_score:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')
    plt.xlabel("False Positive Rate/ Specificity")
    plt.ylabel("True Positive Rate/ Sensitivity")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=1000)
    plt.close()
    print(f"✅ ROC curve saved to {path}")

def save_metrics_table(df, path, title):
    fig, ax = plt.subplots(figsize=(max(6, df.shape[1]*1.5), max(1.5, df.shape[0]*0.5)))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.round(4).values, colLabels=df.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=500)
    plt.close()
    print(f"✅ Table saved to {path}")

def plot_fold_learning_curves(histories, metric, save_path, title_prefix):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.ravel()
    labels = ['(a)', '(b)', '(c)', '(d)']
    for i, history in enumerate(histories):
        axs[i].plot(history[metric], label='Training', color='blue')
        axs[i].plot(history[f'val_{metric}'], label='Validation', color='red')
        axs[i].set_title(f"{labels[i]} Fold {i+1} {title_prefix} Curve", loc='left', fontsize=10)
        axs[i].set_xlabel('Epochs')
        axs[i].set_ylabel(metric.capitalize())
        axs[i].legend()
        axs[i].grid(True, alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=1000)
    plt.close()
    print(f"✅ {title_prefix} curves saved to {save_path}")

# Save everything
save_metrics_table(train_metrics_df, "output/train_metrics_table_distilbert.png", "Training Metrics (DistilBERT)")
save_confusion_matrix(confusion_matrix(y_test, y_test_pred), "output/confusion_matrix_test_distilbert.png")
save_metrics_table(test_metrics_df, "output/test_metrics_table_distilbert.png", "Test Metrics (DistilBERT)")
plot_fold_learning_curves(fold_histories, 'accuracy', 'output/folds_accuracy_curves_distilbert.png', 'Accuracy')
plot_fold_learning_curves(fold_histories, 'loss', 'output/folds_loss_curves_distilbert.png', 'Loss')
save_roc_curve(y_test, y_test_probs, "output/roc_curve_distilbert.png")
np.save('output/distilbert_default_fold_histories.npy', fold_histories)
np.save("output/distilbert_default_y_test_probs.npy", y_test_probs)
np.save("output/distilbert_default_y_test_true.npy", y_test)
np.save("output/distilbert_default_y_test_pred.npy", y_test_pred)

########################################################################
# ======= DISTILBERT WITH CUSTOM HEAD (W/o manual features) ===========#
########################################################################

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import numpy as np
import tensorflow as tf
import seaborn as sns
import pandas as pd
import time
import random
import matplotlib.pyplot as plt

# Mixed Precision Policy
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Set random seed for reproducibility
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# L2 regularizer
l2_reg = tf.keras.regularizers.l2(1e-4)

# ============== Load Data ===================
df = pd.read_csv('data/final_selected_features_data.csv')
urls = df['url'].tolist()
labels = df['phishing'].values.astype(np.float32)
bert_embeddings = np.load('Embeddings/cls_embeddings.npy').astype(np.float32)

# ============== Train-Test Split ==============
X_train, X_test, y_train, y_test = train_test_split(
    bert_embeddings, labels, test_size=0.2, stratify=labels, random_state=SEED
)

# ============== Cross-Validation Setup ==============
NUM_FOLDS = 4
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
train_metrics = []
fold_histories = []

# ============== Training Loop ===================
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    print(f"\n📦 Fold {fold}")
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    input_bert = tf.keras.Input(shape=(768,), dtype=tf.float32, name="bert_embeddings")
    x = tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=l2_reg)(input_bert)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=l2_reg)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32')(x)

    model = tf.keras.Model(inputs=input_bert, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(5e-5), loss='binary_crossentropy', metrics=['accuracy'])

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint('dist_custom_head.h5', monitor='val_loss', save_best_only=True)

    train_start = time.time()
    history = model.fit(X_tr, y_tr,
              validation_data=(X_val, y_val),
              epochs=20, batch_size=64, verbose=1, callbacks=[early_stopping, checkpoint])
    fold_histories.append(history.history)
    train_duration = time.time() - train_start

    val_start = time.time()
    y_pred = (model.predict(X_val) > 0.5).astype(int)
    val_duration = time.time() - val_start

    metrics = {
        'Fold': f'Fold {fold}',
        'Accuracy': accuracy_score(y_val, y_pred),
        'Precision': precision_score(y_val, y_pred),
        'Recall': recall_score(y_val, y_pred),
        'F1 Score': f1_score(y_val, y_pred),
        'AUC-ROC': roc_auc_score(y_val, y_pred),
        'Training Time (s)': round(train_duration, 2),
        'Prediction Time (s)': round(val_duration, 2)
    }
    train_metrics.append(metrics)

# ============== Save Training Metrics ==============
train_metrics_df = pd.DataFrame(train_metrics)
avg = train_metrics_df.mean(numeric_only=True)
avg['Fold'] = 'Average'
train_metrics_df = pd.concat([train_metrics_df, pd.DataFrame([avg])], ignore_index=True)
print(train_metrics_df)

# ============== Final Test Set Evaluation ==============
test_start = time.time()
y_test_probs = model.predict(X_test).flatten()
y_test_pred = (y_test_probs > 0.5).astype(int)
test_duration = time.time() - test_start

test_metrics = {
    'Set': 'Test',
    'Accuracy': accuracy_score(y_test, y_test_pred),
    'Precision': precision_score(y_test, y_test_pred),
    'Recall': recall_score(y_test, y_test_pred),
    'F1 Score': f1_score(y_test, y_test_pred),
    'AUC-ROC': roc_auc_score(y_test, y_test_probs),
    'Prediction Time (s)': round(test_duration, 2)
}
test_metrics_df = pd.DataFrame([test_metrics])
print("\n🎯 Final Test Set Performance:")
print(test_metrics_df)

# ============== Save Visualizations ==============
def save_confusion_matrix(cm, path="output/confusion_matrix_ablation.png", title="Confusion Matrix"):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Legit', 'Phish'], yticklabels=['Legit', 'Phish'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=1000)
    plt.close()

def save_metrics_table(df, path="output/ablation_metrics_table.png", title="Metrics Table"):
    fig, ax = plt.subplots(figsize=(max(6, df.shape[1]*1.5), max(1.5, df.shape[0]*0.5)))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.round(4).values, colLabels=df.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=500)
    plt.close()

def save_roc_curve(y_true, y_probs, path="output/ablation_roc_curve.png", title="Receiver Operating Characteristic (ROC) Curve", model_name="DistilBERT-CUS-HEAD"):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc_score = roc_auc_score(y_true, y_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_score:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate/ Specificity")
    plt.ylabel("True Positive Rate/ Sensitivity")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=1000)
    plt.close()

def plot_fold_learning_curves(histories, metric, save_path, title_prefix):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.ravel()
    labels = ['(a)', '(b)', '(c)', '(d)']
    for i, history in enumerate(histories):
        axs[i].plot(history[metric], label='Training', color='blue')
        axs[i].plot(history[f'val_{metric}'], label='Validation', color='red')
        axs[i].set_title(f"{labels[i]} Fold {i+1} {title_prefix}", loc='left', fontsize=10)
        axs[i].set_xlabel('Epochs')
        axs[i].set_ylabel(metric.capitalize())
        axs[i].legend()
        axs[i].grid(True, alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=1000)
    plt.close()

# Save results
save_metrics_table(train_metrics_df, "output/CLSonly_train_metrics_table.png", "Training Metrics (No Manual Features)")
save_confusion_matrix(confusion_matrix(y_test, y_test_pred), "output/CLSonly_confusion_matrix.png")
save_metrics_table(test_metrics_df, "output/CLSonly_test_metrics_table.png", "Final Test Set Metrics (No Manual Features)")
plot_fold_learning_curves(fold_histories, 'accuracy', 'output/CLSonly_folds_accuracy_curves.png', 'Accuracy')
plot_fold_learning_curves(fold_histories, 'loss', 'output/CLSonly_folds_loss_curves.png', 'Loss')
save_roc_curve(y_test, y_test_probs, "output/CLSonly_ablation_test_roc_curve.png")
np.save('output/distilbert_custom_fold_histories.npy', fold_histories)
np.save("output/distilbert_custom_y_test_probs.npy", y_test_probs)
np.save("output/distilbert_custom_y_test_true.npy", y_test)
np.save("output/distilbert_custom_y_test_pred.npy", y_test_pred)




#########################################################################################
#=============== MAN_FEATURES ONLY WITH CUSTOM HEAD (W/o CLS Embeddings) ===============#
#########################################################################################

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import numpy as np
import tensorflow as tf
import seaborn as sns
import pandas as pd
import time
import random
import matplotlib.pyplot as plt

# Set random seed for reproducibility
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Mixed Precision Policy
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# L2 regularizer
l2_reg = tf.keras.regularizers.l2(1e-4)

# ============== Load Data ===================
df = pd.read_csv('data/final_selected_features_data.csv')
manual_features = df.drop(columns=['url', 'phishing']).values.astype(np.float32)
labels = df['phishing'].values.astype(np.float32)

X = manual_features
y = labels

# ============== Train-Test Split ==============
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============== Cross-Validation Setup ==============
NUM_FOLDS = 4
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
train_metrics = []
fold_histories = []

# ============== Training Loop ===================
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    print(f"\n📦 Fold {fold}")

    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    # Define input
    input_manual = tf.keras.Input(shape=(X.shape[1],), dtype=tf.float32, name="manual_features")
    x = tf.keras.layers.BatchNormalization()(input_manual)
    x = tf.keras.layers.Dense(64, use_bias = False, kernel_regularizer=l2_reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32')(x)

    # Define the model
    model = tf.keras.Model(inputs=input_manual, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(5e-5), loss='binary_crossentropy', metrics=['accuracy'])

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint('man_features_only.h5', monitor='val_loss', save_best_only=True)

    train_start = time.time()
    history = model.fit(X_tr, y_tr,
                        validation_data=(X_val, y_val),
                        epochs=20, batch_size=64, verbose=1,
                        callbacks=[early_stopping, checkpoint])
    fold_histories.append(history.history)

    train_duration = time.time() - train_start
    val_start = time.time()
    y_pred = (model.predict(X_val) > 0.5).astype(int)
    val_duration = time.time() - val_start

    metrics = {
        'Fold': f'Fold {fold}',
        'Accuracy': accuracy_score(y_val, y_pred),
        'Precision': precision_score(y_val, y_pred),
        'Recall': recall_score(y_val, y_pred),
        'F1 Score': f1_score(y_val, y_pred),
        'AUC-ROC': roc_auc_score(y_val, y_pred),
        'Training Time (s)': round(train_duration, 2),
        'Prediction Time (s)': round(val_duration, 2)
    }
    train_metrics.append(metrics)

# ============== Save Training Metrics ==============
train_metrics_df = pd.DataFrame(train_metrics)
avg = train_metrics_df.mean(numeric_only=True)
avg['Fold'] = 'Average'
train_metrics_df = pd.concat([train_metrics_df, pd.DataFrame([avg])], ignore_index=True)
print(train_metrics_df)


def plot_fold_learning_curves(histories, metric, save_path, title_prefix):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.ravel()
    labels = ['(a)', '(b)', '(c)', '(d)']

    for i, history in enumerate(histories):
        axs[i].plot(history[metric], label='Training', color='blue')
        axs[i].plot(history[f'val_{metric}'], label='Validation', color='red')
        axs[i].set_title(f"{labels[i]} Fold {i+1} {title_prefix} Curve", loc='left', fontsize=10)
        axs[i].set_xlabel('Epochs')
        axs[i].set_ylabel(metric.capitalize())
        axs[i].legend()
        axs[i].grid(True, alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=1000)
    plt.close()
    print(f"✅ {title_prefix} curves saved to {save_path}")

# ============== Final Test Set Evaluation ==============
test_start = time.time()
y_test_probs = model.predict(X_test).flatten()
y_test_pred = (y_test_probs > 0.5).astype(int)
test_duration = time.time() - test_start

test_metrics = {
    'Set': 'Test',
    'Accuracy': accuracy_score(y_test, y_test_pred),
    'Precision': precision_score(y_test, y_test_pred),
    'Recall': recall_score(y_test, y_test_pred),
    'F1 Score': f1_score(y_test, y_test_pred),
    'AUC-ROC': roc_auc_score(y_test, y_test_pred),
    'Prediction Time (s)': round(test_duration, 2)
}
test_metrics_df = pd.DataFrame([test_metrics])
print("\n🎯 Final Test Set Performance:")
print(test_metrics_df)

# ============== Save Visualizations ==============
def save_confusion_matrix(cm, path="output/confusion_matrix.png", title="Confusion Matrix"):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Legit', 'Phish'], yticklabels=['Legit', 'Phish'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=1000)
    plt.close()
    print(f"✅ Confusion matrix saved to {path}")

def save_metrics_table(df, path="output/metrics_table.png", title="Metrics Table"):
    fig, ax = plt.subplots(figsize=(max(6, df.shape[1]*1.5), max(1.5, df.shape[0]*0.5)))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.round(4).values,
                     colLabels=df.columns,
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=500)
    plt.close()
    print(f"✅ Table saved to {path}")

def save_roc_curve(y_true, y_probs, path="output/roc_curve.png", title="Receiver Operating Characteristic (ROC) Curve", model_name="MAN-FEAT-ONLY"):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc_score = roc_auc_score(y_true, y_probs)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_score:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate/ Specificity")
    plt.ylabel("True Positive Rate/ Sensitivity")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=1000)
    plt.close()
    print(f"✅ ROC curve saved to {path}")

save_metrics_table(train_metrics_df, "output/man_train_metrics_table.png", "Training Metrics (Cross-Validation)")
save_confusion_matrix(confusion_matrix(y_test, y_test_pred), "output/man_confusion_matrix_test.png")
save_metrics_table(test_metrics_df, "output/man_test_metrics_table.png", "Final Test Set Metrics")
plot_fold_learning_curves(fold_histories, 'accuracy', 'output/man_folds_accuracy_curves.png', 'Accuracy')
plot_fold_learning_curves(fold_histories, 'loss', 'output/man_folds_loss_curves.png', 'Loss')
save_roc_curve(y_test, y_test_probs, "output/man_test_roc_curve.png")
np.save('output/manual_features_fold_histories.npy', fold_histories)
np.save("output/manual_features_y_test_probs.npy", y_test_probs)
np.save("output/manual_features_y_test_true.npy", y_test)
np.save("output/manual_features_y_test_pred.npy", y_test_pred)


#---------------------------------------EMBEDDINGS COMPARISONS--------------------------------------------#

                            ####################################################
                            # =============== ELMo Embeddings   ===============#
                            ####################################################

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import numpy as np
import tensorflow as tf
import seaborn as sns
import pandas as pd
import time
import os 
import random
import matplotlib.pyplot as plt

# 🎯 Set Random Seed for Reproducibility
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Mixed Precision Policy
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Define L2 regularizer
l2_reg = tf.keras.regularizers.l2(1e-4)

# ============== Load Data ===================
df = pd.read_csv('data/final_selected_features_data.csv')
urls = df['url'].tolist()
manual_features = df.drop(columns=['url', 'phishing']).values.astype(np.float32)
labels = df['phishing'].values.astype(np.float32)

# ============== Load ELMo Embeddings ==============
elmo_embeddings = np.load('Embeddings/elmo_embeddings.npy')  # shape: (N, 1024)
X_ids = elmo_embeddings.astype(np.float32)
print("Shape:", elmo_embeddings.shape) # Load precomputed BERT embeddings if available
X_manual = manual_features
y = labels

# ============== Train-Test Split ==============
X_ids_train, X_ids_test, X_manual_train, X_manual_test, y_train, y_test = train_test_split(
    X_ids, X_manual, y, test_size=0.2, random_state=42, stratify=y
)

# ============== Cross-Validation Setup ==============
NUM_FOLDS = 4
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
train_metrics = []
fold_histories = []

# ============== Training Loop ===================
for fold, (train_idx, val_idx) in enumerate(skf.split(X_ids_train, y_train), 1):
    print(f"\n📦 Fold {fold}")

    ids_tr, ids_val = X_ids_train[train_idx], X_ids_train[val_idx]
    man_tr, man_val = X_manual_train[train_idx], X_manual_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    input_elmo = tf.keras.Input(shape=(1024,), dtype=tf.float32, name="elmo_embeddings")
    input_manual = tf.keras.Input(shape=(X_manual.shape[1],), dtype=tf.float32, name="manual_features")

    x_elmo = tf.keras.layers.Dense(512, use_bias = False, kernel_regularizer=l2_reg)(input_elmo)
    x_elmo = tf.keras.layers.BatchNormalization()(x_elmo)
    x_elmo = tf.keras.layers.Activation('relu')(x_elmo)
    x_elmo = tf.keras.layers.Dropout(0.3)(x_elmo)
    x_elmo = tf.keras.layers.Dense(256, use_bias = False, kernel_regularizer=l2_reg)(x_elmo)
    x_elmo = tf.keras.layers.BatchNormalization()(x_elmo)
    x_elmo = tf.keras.layers.Activation('relu')(x_elmo)
    x_elmo = tf.keras.layers.Dropout(0.3)(x_elmo)

    x_manual = tf.keras.layers.BatchNormalization()(input_manual)
    x = tf.keras.layers.Concatenate()([x_elmo, x_manual])

    x = tf.keras.layers.Dense(64, use_bias = False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32')(x)

    model = tf.keras.Model(inputs=[input_elmo, input_manual], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(5e-5), loss='binary_crossentropy', metrics=['accuracy'])

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint('elmo_model.h5', monitor='val_loss', save_best_only=True)

    train_start = time.time()
    history = model.fit([ids_tr, man_tr], y_tr,
                        validation_data=([ids_val, man_val], y_val),
                        epochs=20, batch_size=64, verbose=1,
                        callbacks=[early_stopping, checkpoint])
    train_duration = time.time() - train_start
    val_start = time.time()
    y_pred = (model.predict([ids_val, man_val]) > 0.5).astype(int)
    val_duration = time.time() - val_start

    fold_histories.append(history.history)
    metrics = {
        'Fold': f'Fold {fold}',
        'Accuracy': accuracy_score(y_val, y_pred),
        'Precision': precision_score(y_val, y_pred),
        'Recall': recall_score(y_val, y_pred),
        'F1 Score': f1_score(y_val, y_pred),
        'AUC-ROC': roc_auc_score(y_val, y_pred),
        'Training Time (s)': round(train_duration, 2),
        'Prediction Time (s)': round(val_duration, 2)    
    }
    train_metrics.append(metrics)

# ============== Save Training Metrics ==============
train_metrics_df = pd.DataFrame(train_metrics)
avg = train_metrics_df.mean(numeric_only=True)
avg['Fold'] = 'Average'
train_metrics_df = pd.concat([train_metrics_df, pd.DataFrame([avg])], ignore_index=True)
print(train_metrics_df)

# ============== Final Test Set Evaluation ==============
test_start = time.time()
y_test_probs = model.predict([X_ids_test, X_manual_test]).flatten()
y_test_pred = (y_test_probs > 0.5).astype(int)
test_duration = time.time() - test_start
test_metrics = {
    'Set': 'Test',
    'Accuracy': accuracy_score(y_test, y_test_pred),
    'Precision': precision_score(y_test, y_test_pred),
    'Recall': recall_score(y_test, y_test_pred),
    'F1 Score': f1_score(y_test, y_test_pred),
    'AUC-ROC': roc_auc_score(y_test, y_test_probs),
    'Prediction Time (s)': round(test_duration, 2)
}
test_metrics_df = pd.DataFrame([test_metrics])
print("\n🎯 Final Test Set Performance:")
print(test_metrics_df)

# ============== Save Visualizations ==============
def save_confusion_matrix(cm, path, title="Confusion Matrix"):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Legit', 'Phish'], yticklabels=['Legit', 'Phish'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=1000)
    plt.close()

def save_metrics_table(df, path, title="Metrics Table"):
    fig, ax = plt.subplots(figsize=(max(6, df.shape[1]*1.5), max(1.5, df.shape[0]*0.5)))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.round(4).values,
                     colLabels=df.columns,
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=500)
    plt.close()

def plot_fold_learning_curves(histories, metric, save_path, title_prefix):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.ravel()
    labels = ['(a)', '(b)', '(c)', '(d)']
    
    for i, history in enumerate(histories):
        axs[i].plot(history[metric], label='Training', color='blue')
        axs[i].plot(history[f'val_{metric}'], label='Validation', color='red')
        axs[i].set_title(f"{labels[i]} Fold {i+1} {title_prefix} Curve", loc='left', fontsize=10)
        axs[i].set_xlabel('Epochs')
        axs[i].set_ylabel(metric.capitalize())
        axs[i].legend()
        axs[i].grid(True, alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=1000)
    plt.close()

def save_roc_curve(y_true, y_probs, path, title="Receiver Operating Characteristic (ROC) Curve", model_name="ELMo"):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc_score = roc_auc_score(y_true, y_probs)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_score:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate/ Specificity")
    plt.ylabel("True Positive Rate/ Sensitivity")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=1000)
    plt.close()

# Save output files
save_metrics_table(train_metrics_df, "output/ELMo_train_metrics_table.png", "Training Metrics (Cross-Validation)")
save_confusion_matrix(confusion_matrix(y_test, y_test_pred), "output/ELMo_confusion_matrix_test.png")
save_metrics_table(test_metrics_df, "output/ELMo_test_metrics_table.png", "Final Test Set Metrics")
plot_fold_learning_curves(fold_histories, 'accuracy', 'output/ELMo_folds_accuracy_curves.png', 'Accuracy')
plot_fold_learning_curves(fold_histories, 'loss', 'output/ELMo_folds_loss_curves.png', 'Loss')  
save_roc_curve(y_test, y_test_probs, "output/ELMo_test_roc_curve.png")

# Save raw outputs
np.save("output/ELMo_y_test_true.npy", y_test)
np.save("output/ELMo_test_pred.npy", y_test_pred)
np.save("output/ELMo_y_test_probs.npy", y_test_probs)
np.save("output/ELMo_fold_histories.npy", fold_histories)



        # ========================================================================#
        # --------------------------FastText Embeddings---------------------------#
        # ========================================================================#

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import os, random, time

# 🎯 Reproducibility
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

tf.keras.mixed_precision.set_global_policy('mixed_float16')

# ============== Load Data ============== #
df = pd.read_csv('data/final_selected_features_data.csv')
urls = df['url'].tolist()
X_manual = df.drop(columns=['url', 'phishing']).values.astype(np.float32)
y = df['phishing'].values.astype(np.float32)

# ⚠️ Load FastText Embeddings (precomputed)
X_fasttext = np.load("Embeddings/fasttext_embeddings.npy").astype(np.float32)  # shape: (n_samples, 300)

# ============== Train-Test Split ============== #
X_ft_train, X_ft_test, X_manual_train, X_manual_test, y_train, y_test = train_test_split(
    X_fasttext, X_manual, y, test_size=0.2, stratify=y, random_state=42
)

# ============== Cross-Validation Setup ============== #
NUM_FOLDS = 4
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
train_metrics = []
fold_histories = []

# ============== Model Training Loop ============== #
l2_reg = tf.keras.regularizers.l2(1e-4)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_ft_train, y_train), 1):
    print(f"\n📦 Fold {fold}")

    ft_tr, ft_val = X_ft_train[train_idx], X_ft_train[val_idx]
    man_tr, man_val = X_manual_train[train_idx], X_manual_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    input_ft = tf.keras.Input(shape=(X_ft_train.shape[1],), dtype=tf.float32, name="fasttext_embeddings")
    input_manual = tf.keras.Input(shape=(X_manual.shape[1],), dtype=tf.float32, name="manual_features")

    # FastText path
    x_ft = tf.keras.layers.Dense(256, use_bias=False, kernel_regularizer=l2_reg)(input_ft)
    x_ft = tf.keras.layers.BatchNormalization()(x_ft)
    x_ft = tf.keras.layers.Activation('relu')(x_ft)
    x_ft = tf.keras.layers.Dropout(0.3)(x_ft)
    x_ft = tf.keras.layers.Dense(128, use_bias=False, kernel_regularizer=l2_reg)(x_ft)
    x_ft = tf.keras.layers.BatchNormalization()(x_ft)
    x_ft = tf.keras.layers.Activation('relu')(x_ft)
    x_ft = tf.keras.layers.Dropout(0.3)(x_ft)

    # Manual path
    x_manual = tf.keras.layers.BatchNormalization()(input_manual)

    # Fusion
    x = tf.keras.layers.Concatenate()([x_ft, x_manual])
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32')(x)

    model = tf.keras.Model(inputs=[input_ft, input_manual], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(5e-5), loss='binary_crossentropy', metrics=['accuracy'])

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint('fasttext_hybrid_model.h5', monitor='val_loss', save_best_only=True)

    train_start = time.time()
    history = model.fit([ft_tr, man_tr], y_tr,
                        validation_data=([ft_val, man_val], y_val),
                        epochs=20, batch_size=64, verbose=1,
                        callbacks=[early_stopping, checkpoint])
    train_duration = time.time() - train_start

    fold_histories.append(history.history)

    val_start = time.time()
    y_pred = (model.predict([ft_val, man_val]) > 0.5).astype(int)
    val_duration = time.time() - val_start

    metrics = {
        'Fold': f'Fold {fold}',
        'Accuracy': accuracy_score(y_val, y_pred),
        'Precision': precision_score(y_val, y_pred),
        'Recall': recall_score(y_val, y_pred),
        'F1 Score': f1_score(y_val, y_pred),
        'AUC-ROC': roc_auc_score(y_val, y_pred),
        'Training Time (s)': round(train_duration, 2),
        'Prediction Time (s)': round(val_duration, 2)
    }
    train_metrics.append(metrics)

# ============== Save Metrics ============== #
train_df = pd.DataFrame(train_metrics)
avg = train_df.mean(numeric_only=True)
avg['Fold'] = 'Average'
train_df = pd.concat([train_df, pd.DataFrame([avg])], ignore_index=True)
print(train_df)


# ============== Test Set Evaluation ============== #
test_start = time.time()
y_test_probs = model.predict([X_ft_test, X_manual_test]).flatten()
y_test_pred = (y_test_probs > 0.5).astype(int)
test_duration = time.time() - test_start

test_metrics_df = pd.DataFrame([{
    'Set': 'Test',
    'Accuracy': accuracy_score(y_test, y_test_pred),
    'Precision': precision_score(y_test, y_test_pred),
    'Recall': recall_score(y_test, y_test_pred),
    'F1 Score': f1_score(y_test, y_test_pred),
    'AUC-ROC': roc_auc_score(y_test, y_test_probs),
    'Prediction Time (s)': round(test_duration, 2)
}])
print(test_metrics_df)

# ========== Plot & Save Outputs ==========

# Save learning curves
def plot_fold_curves(histories, metric, filename, title_prefix):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.ravel()
    labels = ['(a)', '(b)', '(c)', '(d)']
    for i, h in enumerate(histories):
        axs[i].plot(h[metric], label='Training', color='blue')
        axs[i].plot(h[f'val_{metric}'], label='Validation', color='red')
        axs[i].set_title(f"{labels[i]} Fold {i+1} {title_prefix}", loc='left')
        axs[i].set_xlabel('Epochs')
        axs[i].set_ylabel(metric.capitalize())
        axs[i].legend()
        axs[i].grid(True, alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename, dpi=1000)
    plt.close()

def save_metrics_table(df, path, title):
    fig, ax = plt.subplots(figsize=(max(6, df.shape[1]*1.5), max(1.5, df.shape[0]*0.5)))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.round(4).values,
                     colLabels=df.columns,
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=500)
    plt.close()

# Save confusion matrix and ROC
def save_conf_matrix(cm, path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Legit', 'Phish'], yticklabels=['Legit', 'Phish'])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=1000)
    plt.close()

def save_roc(y_true, y_probs, path, model_name="FastText Hybrid", title="ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc = roc_auc_score(y_true, y_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=1000)
    plt.close()


save_metrics_table(train_df, "output/fasttext_train_metrics.png", "ELMo Training Metrics")
save_metrics_table(test_metrics_df, "output/fasttext_test_metrics.png", "ELMo Test Metrics")
save_conf_matrix(confusion_matrix(y_test, y_test_pred), "output/fasttext_model_confusion.png")
save_roc(y_test, y_test_probs, "output/fasttext_model_roc.png")
plot_fold_curves(fold_histories, 'accuracy', 'output/fasttext_model_accuracy.png', 'Accuracy')
plot_fold_curves(fold_histories, 'loss', 'output/fasttext_model_loss.png', 'Loss')

# ============== Save Artifacts ============== #
np.save("output/fasttext_y_test_true.npy", y_test)
np.save("output/fasttext_y_test_pred.npy", y_test_pred)
np.save("output/fasttext_y_test_probs.npy", y_test_probs)
np.save("output/fasttext_fold_histories_ft_model.npy", fold_histories)





                            ####################################################
                            # =============== GloVe Embeddings ================#
                            ####################################################


# ✅ GloVe Version of Proposed Model Training Pipeline

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import numpy as np
import tensorflow as tf
import seaborn as sns
import pandas as pd
import time
import os
import random
import matplotlib.pyplot as plt

# 🎯 Set Random Seed for Reproducibility
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Mixed Precision Policy
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# L2 regularizer
l2_reg = tf.keras.regularizers.l2(1e-4)

# ============== Load Data ===================
df = pd.read_csv('data/final_selected_features_data.csv')
urls = df['url'].tolist()
manual_features = df.drop(columns=['url', 'phishing']).values.astype(np.float32)
labels = df['phishing'].values.astype(np.float32)

# ============== Load GloVe Embeddings =============
glove_embeddings = np.load('Embeddings/glove_embeddings.npy')  # Shape: (num_samples, 300)
X_ids = glove_embeddings.astype(np.float32)
X_manual = manual_features
y = labels

# ============== Train-Test Split ==============
X_ids_train, X_ids_test, X_manual_train, X_manual_test, y_train, y_test = train_test_split(
    X_ids, X_manual, y, test_size=0.2, random_state=SEED, stratify=y
)

# ============== Cross-Validation Setup ==============
NUM_FOLDS = 4
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
train_metrics = []
fold_histories = []

# ============== Training Loop ===================
for fold, (train_idx, val_idx) in enumerate(skf.split(X_ids_train, y_train), 1):
    print(f"\n📦 Fold {fold}")

    ids_tr, ids_val = X_ids_train[train_idx], X_ids_train[val_idx]
    man_tr, man_val = X_manual_train[train_idx], X_manual_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    input_glove = tf.keras.Input(shape=(X_ids.shape[1],), dtype=tf.float32, name="glove_embeddings")
    input_manual = tf.keras.Input(shape=(X_manual.shape[1],), dtype=tf.float32, name="manual_features")

    # GloVe path
    x_glove = tf.keras.layers.Dense(256, use_bias=False, kernel_regularizer=l2_reg)(input_glove)
    x_glove = tf.keras.layers.BatchNormalization()(x_glove)
    x_glove = tf.keras.layers.Activation('relu')(x_glove)
    x_glove = tf.keras.layers.Dropout(0.3)(x_glove)
    x_glove = tf.keras.layers.Dense(128, use_bias=False, kernel_regularizer=l2_reg)(x_glove)
    x_glove = tf.keras.layers.BatchNormalization()(x_glove)
    x_glove = tf.keras.layers.Activation('relu')(x_glove)
    x_glove = tf.keras.layers.Dropout(0.3)(x_glove)

    x_manual = tf.keras.layers.BatchNormalization()(input_manual)
    x = tf.keras.layers.Concatenate()([x_glove, x_manual])
    x = tf.keras.layers.Dense(64, use_bias=False, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid', dtype='float32')(x)

    model = tf.keras.Model(inputs=[input_glove, input_manual], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(5e-5), loss='binary_crossentropy', metrics=['accuracy'])

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint('proposed_glove_model.h5', monitor='val_loss', save_best_only=True)

    train_start = time.time()
    history = model.fit([ids_tr, man_tr], y_tr,
                        validation_data=([ids_val, man_val], y_val),
                        epochs=20, batch_size=64, verbose=1,
                        callbacks=[early_stopping, checkpoint])
    train_duration = time.time() - train_start
    fold_histories.append(history.history)

    val_start = time.time()
    y_pred = (model.predict([ids_val, man_val]) > 0.5).astype(int)
    val_duration = time.time() - val_start

    metrics = {
        'Fold': f'Fold {fold}',
        'Accuracy': accuracy_score(y_val, y_pred),
        'Precision': precision_score(y_val, y_pred),
        'Recall': recall_score(y_val, y_pred),
        'F1 Score': f1_score(y_val, y_pred),
        'AUC-ROC': roc_auc_score(y_val, y_pred),
        'Training Time (s)': round(train_duration, 2),
        'Prediction Time (s)': round(val_duration, 2)
    }
    train_metrics.append(metrics)

train_metrics_df = pd.DataFrame(train_metrics)
avg = train_metrics_df.mean(numeric_only=True)
avg['Fold'] = 'Average'
train_metrics_df = pd.concat([train_metrics_df, pd.DataFrame([avg])], ignore_index=True)
print(train_metrics_df)

# ============== Final Test Set Evaluation ==============
test_start = time.time()
y_test_probs = model.predict([X_ids_test, X_manual_test]).flatten()
y_test_pred = (y_test_probs > 0.5).astype(int)
test_duration = time.time() - test_start

# Save training metrics to CSV
def plot_fold_learning_curves(histories, metric, save_path, title_prefix):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.ravel()
    labels = ['(a)', '(b)', '(c)', '(d)']
    
    for i, history in enumerate(histories):
        axs[i].plot(history[metric], label='Training', color='blue')
        axs[i].plot(history[f'val_{metric}'], label='Validation', color='red')
        axs[i].set_title(f"{labels[i]} Fold {i+1} {title_prefix} Curve", loc='left', fontsize=10)
        axs[i].set_xlabel('Epochs')
        axs[i].set_ylabel(metric.capitalize())
        axs[i].legend()
        axs[i].grid(True, alpha=0.6)  # Can add this linestyle='--' for dotter lines if you prefer
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=1000)
    plt.close()
    print(f"✅ {title_prefix} curves saved to {save_path}")

test_metrics_df = pd.DataFrame([{ 
    'Set': 'Test',
    'Accuracy': accuracy_score(y_test, y_test_pred),
    'Precision': precision_score(y_test, y_test_pred),
    'Recall': recall_score(y_test, y_test_pred),
    'F1 Score': f1_score(y_test, y_test_pred),
    'AUC-ROC': roc_auc_score(y_test, y_test_probs),
    'Prediction Time (s)': round(test_duration, 2)
}])

print("\n🎯 Final Test Set Performance:")
print(test_metrics_df)

#============== Save Visualizations ==============
def save_confusion_matrix(cm, path="output/confusion_matrix.png", title="Confusion Matrix"):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Legit', 'Phish'], yticklabels=['Legit', 'Phish'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=1000)
    plt.close()
    print(f"✅ Confusion matrix saved to {path}")

def save_metrics_table(df, path="output/metrics_table.png", title="Metrics Table"):
    fig, ax = plt.subplots(figsize=(max(6, df.shape[1]*1.5), max(1.5, df.shape[0]*0.5)))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.round(4).values,
                     colLabels=df.columns,
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=500)
    plt.close()
    print(f"✅ Table saved to {path}")
from sklearn.metrics import roc_curve

def save_roc_curve(y_true, y_probs, path="output/roc_curve.png", title="Receiver Operating Characteristic (ROC) Curve", model_name="DistilBERT"):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc_score = roc_auc_score(y_true, y_probs)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_score:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate/ Specificity")
    plt.ylabel("True Positive Rate/ Sensitivity")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=1000)
    plt.close()
    print(f"✅ ROC curve saved to {path}")

# Save results
save_metrics_table(train_metrics_df, "output/glove_train_metrics_table.png", "Training Metrics (Cross-Validation)")
save_confusion_matrix(confusion_matrix(y_test, y_test_pred), "output/glove_confusion_matrix_test.png")
save_metrics_table(test_metrics_df, "output/glove_test_metrics_table.png", "Final Test Set Metrics")
plot_fold_learning_curves(fold_histories, 'accuracy', 'output/glove_folds_accuracy_curves.png', 'Accuracy')
plot_fold_learning_curves(fold_histories, 'loss', 'output/glove_folds_loss_curves.png', 'Loss')  
save_roc_curve(y_test, y_test_probs, "output/glove_test_roc_curve.png")

# Save artifacts to .npy
np.save("output/glove_y_test_true.npy", y_test)
np.save("output/glove_y_test_pred.npy", y_test_pred)
np.save("output/glove_y_test_probs.npy", y_test_probs)
np.save("output/glove_fold_histories.npy", fold_histories)



#=========================================END OF Embedding Comparison ========================================#


#========================================  BASELINE  MODELS  ===========================================#

#################################################################
# ==================== BERT BASELINE MODEL =====================#
#################################################################
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from transformers import BertTokenizerFast, TFBertForSequenceClassification, AdamW

# Set random seed for reproducibility
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Set mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# ==================== Load and Prepare Data =====================
df = pd.read_csv('data/final_selected_features_data.csv')
urls = df['url'].tolist()
labels = df['phishing'].values.astype(np.float32)
assert len(np.unique(labels)) == 2, "Binary classification expected with 2 label classes."

# Tokenization using BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
encodings = tokenizer(urls, truncation=True, padding='max_length', max_length=200, return_tensors='np')
input_ids = encodings['input_ids']
attention_mask = encodings['attention_mask']

# Train-test split
X_ids_train, X_ids_test, X_mask_train, X_mask_test, y_train, y_test = train_test_split(
    input_ids, attention_mask, labels, test_size=0.2, random_state=SEED, stratify=labels
)

# Cross-validation setup
NUM_FOLDS = 4
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
train_metrics = []
fold_histories = []

# Training loop
for fold, (train_idx, val_idx) in enumerate(skf.split(X_ids_train, y_train), 1):
    print(f"\n📦 Fold {fold} - BERT")

    ids_tr, ids_val = X_ids_train[train_idx], X_ids_train[val_idx]
    mask_tr, mask_val = X_mask_train[train_idx], X_mask_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    model = TFBertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2,
        problem_type="single_label_classification"
    )

    optimizer = AdamW(learning_rate=5e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint('BERT_model.keras', monitor='val_loss', save_best_only=True)

    train_start = time.time()
    history = model.fit(
        {'input_ids': ids_tr, 'attention_mask': mask_tr}, y_tr,
        validation_data=({'input_ids': ids_val, 'attention_mask': mask_val}, y_val),
        epochs=20, batch_size=64, callbacks=[early_stopping, checkpoint], verbose=1
    )
    train_duration = time.time() - train_start

    val_start = time.time()
    logits = model.predict({'input_ids': ids_val, 'attention_mask': mask_val}).logits
    probs = tf.nn.softmax(logits, axis=-1).numpy()
    y_val_pred = np.argmax(probs, axis=1)
    val_duration = time.time() - val_start

    metrics = {
        'Fold': f'Fold {fold}',
        'Accuracy': accuracy_score(y_val, y_val_pred),
        'Precision': precision_score(y_val, y_val_pred),
        'Recall': recall_score(y_val, y_val_pred),
        'F1 Score': f1_score(y_val, y_val_pred),
        'AUC-ROC': roc_auc_score(y_val, y_val_pred),
        'Training Time (s)': round(train_duration, 2),
        'Prediction Time (s)': round(val_duration, 2)
    }
    train_metrics.append(metrics)
    fold_histories.append(history.history)

# ==================== Save Training Metrics =====================
train_metrics_df = pd.DataFrame(train_metrics)
avg = train_metrics_df.mean(numeric_only=True)
avg['Fold'] = 'Average'
train_metrics_df = pd.concat([train_metrics_df, pd.DataFrame([avg])], ignore_index=True)
print(train_metrics_df)
train_metrics_df.to_csv("output/train_metrics_bert.csv", index=False)

# ==================== Final Test Set Evaluation =====================
test_start = time.time()
y_test_logits = model.predict({'input_ids': X_ids_test, 'attention_mask': X_mask_test}).logits
y_test_probs = tf.nn.softmax(y_test_logits, axis=-1).numpy()
y_test_pred = np.argmax(y_test_probs, axis=1)
test_duration = time.time() - test_start

test_metrics_df = pd.DataFrame([{
    'Set': 'Test',
    'Accuracy': accuracy_score(y_test, y_test_pred),
    'Precision': precision_score(y_test, y_test_pred),
    'Recall': recall_score(y_test, y_test_pred),
    'F1 Score': f1_score(y_test, y_test_pred),
    'AUC-ROC': roc_auc_score(y_test, y_test_probs[:, 1]),
    'Prediction Time (s)': round(test_duration, 2)
}])

print("\n🎯 Final Test Set Performance (BERT):")
print(test_metrics_df)
test_metrics_df.to_csv("output/test_metrics_bert.csv", index=False)

# ==================== Save Visualizations =====================
def save_confusion_matrix(cm, path="output/confusion_matrix_test_distilbert.png", title="Confusion Matrix"):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Legit', 'Phish'], yticklabels=['Legit', 'Phish'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=1000)
    plt.close()
    print(f"✅ Confusion matrix saved to {path}")

def save_roc_curve(y_true, y_probs, path="output/roc_curve_distilbert.png", title="Receiver Operating Characteristic (ROC) Curve", model_name="BERT"):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc_score = roc_auc_score(y_true, y_probs)
    plt.figure(figsize=(6, 5)) 
    plt.plot(fpr, tpr, label =f"{model_name} (AUC = {auc_score:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate/ Specificity")
    plt.ylabel("True Positive Rate/ Sensitivity")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=1000)
    plt.close()
    print(f"✅ ROC curve saved to {path}")

def save_metrics_table(df, path, title):
    fig, ax = plt.subplots(figsize=(max(6, df.shape[1]*1.5), max(1.5, df.shape[0]*0.5)))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.round(4).values, colLabels=df.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=500)
    plt.close()
    print(f"✅ Table saved to {path}")

def plot_fold_learning_curves(histories, metric, save_path, title_prefix):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.ravel()
    labels = ['(a)', '(b)', '(c)', '(d)']
    for i, history in enumerate(histories):
        axs[i].plot(history[metric], label='Training', color='blue')
        axs[i].plot(history[f'val_{metric}'], label='Validation', color='red')
        axs[i].set_title(f"{labels[i]} Fold {i+1} {title_prefix} Curve", loc='left', fontsize=10)
        axs[i].set_xlabel('Epochs')
        axs[i].set_ylabel(metric.capitalize())
        axs[i].legend()
        axs[i].grid(True, alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=1000)
    plt.close()
    print(f"✅ {title_prefix} curves saved to {save_path}")

# Save results
save_metrics_table(train_metrics_df, "output/train_metrics_table_bert.png", "Training Metrics (BERT)")
save_confusion_matrix(confusion_matrix(y_test, y_test_pred), "output/confusion_matrix_test_bert.png")
save_metrics_table(test_metrics_df, "output/test_metrics_table_bert.png", "Test Metrics (BERT)")
plot_fold_learning_curves(fold_histories, 'accuracy', 'output/folds_accuracy_curves_bert.png', 'Accuracy')
plot_fold_learning_curves(fold_histories, 'loss', 'output/folds_loss_curves_bert.png', 'Loss')
save_roc_curve(y_test, y_test_probs, "output/roc_curve_bert.png")
np.save('output/BERT_fold_histories.npy', fold_histories)
np.save("output/BERT_y_test_probs.npy", y_test_probs)
np.save("output/BERT_y_test_true.npy", y_test)
np.save("output/BERT_y_test_pred.npy", y_test_pred)


####################################################################
# ==================== ELECTRA BASELINE MODEL =====================#
####################################################################



from transformers import TFElectraModel, ElectraTokenizer, TFElectraForSequenceClassification
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

# Set random seed for reproducibility
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Mixed precision for performance
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Load data
df = pd.read_csv("data/final_selected_features_data.csv")
urls = df['url'].tolist()
labels = df['phishing'].values.astype(np.float32)

# Load tokenizer and model
tokenizer = ElectraTokenizer.from_pretrained("google/electra-small-discriminator")
model_name = "google/electra-small-discriminator"

# Tokenize URLs
encodings = tokenizer(urls, truncation=True, padding='max_length', max_length=200, return_tensors='tf')
X_input_ids = encodings['input_ids']
X_attention_mask = encodings['attention_mask']

y = labels

# Train-test split
X_ids_train, X_ids_test, X_mask_train, X_mask_test, y_train, y_test = train_test_split(
    X_input_ids, X_attention_mask, y, test_size=0.2, random_state=42, stratify=y
)

# Stratified K-Fold
NUM_FOLDS = 4
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
train_metrics = []
fold_histories = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_ids_train, y_train), 1):
    print(f"\n📦 Fold {fold}")

    input_ids_tr, input_ids_val = tf.gather(X_ids_train, train_idx), tf.gather(X_ids_train, val_idx)
    att_mask_tr, att_mask_val = tf.gather(X_mask_train, train_idx), tf.gather(X_mask_train, val_idx)
    y_tr, y_val = tf.gather(y_train, train_idx), tf.gather(y_train, val_idx)

    model = TFElectraForSequenceClassification.from_pretrained(model_name, num_labels=1)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(f'electra_model.h5',
                                                    monitor='val_loss', save_best_only=True)

    history = model.fit(
        {'input_ids': input_ids_tr, 'attention_mask': att_mask_tr}, y_tr,
        validation_data=({'input_ids': input_ids_val, 'attention_mask': att_mask_val}, y_val),
        epochs=20, batch_size=64, callbacks=[early_stopping, checkpoint], verbose=1
    )

    fold_histories.append(history.history)
    y_pred = (tf.sigmoid(model.predict({'input_ids': input_ids_val, 'attention_mask': att_mask_val}).logits).numpy().flatten() > 0.5).astype(int)

    metrics = {
        'Fold': f'Fold {fold}',
        'Accuracy': accuracy_score(y_val, y_pred),
        'Precision': precision_score(y_val, y_pred),
        'Recall': recall_score(y_val, y_pred),
        'F1 Score': f1_score(y_val, y_pred),
        'AUC-ROC': roc_auc_score(y_val, y_pred)
    }
    train_metrics.append(metrics)

# Save training metrics
train_metrics_df = pd.DataFrame(train_metrics)
avg = train_metrics_df.mean(numeric_only=True)
avg['Fold'] = 'Average'
train_metrics_df = pd.concat([train_metrics_df, pd.DataFrame([avg])], ignore_index=True)
print(train_metrics_df)

# Final evaluation on test set
y_test_probs = tf.sigmoid(model.predict({'input_ids': X_ids_test, 'attention_mask': X_mask_test}).logits).numpy().flatten()
y_test_pred = (y_test_probs > 0.5).astype(int)

cm = confusion_matrix(y_test, y_test_pred)
test_metrics = {
    'Set': 'Test',
    'Accuracy': accuracy_score(y_test, y_test_pred),
    'Precision': precision_score(y_test, y_test_pred),
    'Recall': recall_score(y_test, y_test_pred),
    'F1 Score': f1_score(y_test, y_test_pred),
    'AUC-ROC': roc_auc_score(y_test, y_test_probs),
    'Prediction Time (s)': round(time.time(), 2)
}
test_metrics_df = pd.DataFrame([test_metrics])
print("\n🎯 Final Test Set Performance:")
print(test_metrics_df)

def save_metrics_table(df, path="output/metrics_table.png", title="Metrics Table"):
    fig, ax = plt.subplots(figsize=(max(6, df.shape[1]*1.5), max(1.5, df.shape[0]*0.5)))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.round(4).values,
                     colLabels=df.columns,
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=500)
    plt.close()
    print(f"✅ Table saved to {path}")

def plot_fold_learning_curves(histories, metric, save_path, title_prefix):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.ravel()
    labels = ['(a)', '(b)', '(c)', '(d)']

    for i, history in enumerate(histories):
        axs[i].plot(history[metric], label='Training', color='blue')
        axs[i].plot(history[f'val_{metric}'], label='Validation', color='red')
        axs[i].set_title(f"{labels[i]} Fold {i+1} {title_prefix} Curve", loc='left', fontsize=10)
        axs[i].set_xlabel('Epochs')
        axs[i].set_ylabel(metric.capitalize())
        axs[i].legend()
        axs[i].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=1000)
    plt.close()
    print(f"✅ {title_prefix} curves saved to {save_path}")

def save_confusion_matrix(cm, path="output/confusion_matrix.png", title="Confusion Matrix"):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Legit', 'Phish'], yticklabels=['Legit', 'Phish'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=1000)
    plt.close()
    print(f"✅ Confusion matrix saved to {path}")

def save_roc_curve(y_true, y_probs, path="output/roc_curve.png", title="Receiver Operating Characteristic (ROC) Curve", model_name="ELECTRA"):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc_score = roc_auc_score(y_true, y_probs)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_score:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate/ Specificity")
    plt.ylabel("True Positive Rate/ Sensitivity")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=1000)
    plt.close()
    print(f"✅ ROC curve saved to {path}")

# Save outputs
save_metrics_table(train_metrics_df, "output/electra_train_metrics_table.png", "Training Metrics (Cross-Validation)")
save_confusion_matrix(cm, "output/electra_confusion_matrix_test.png")
save_metrics_table(test_metrics_df, "output/electra_test_metrics_table.png", "Final Test Set Metrics")
plot_fold_learning_curves(fold_histories, 'accuracy', 'output/electra_folds_accuracy_curves.png', 'Accuracy')
plot_fold_learning_curves(fold_histories, 'loss', 'output/electra_folds_loss_curves.png', 'Loss')
save_roc_curve(y_test, y_test_probs, "output/electra_test_roc_curve.png")
np.save('output/ELECTRA_fold_histories.npy', fold_histories)
np.save("output/ELECTRA_y_test_probs.npy", y_test_probs)
np.save("output/ELECTRA_y_test_true.npy", y_test)
np.save("output/ELECTRA_y_test_pred.npy", y_test_pred)

####################################################################
# ==================== ROBERTA BASELINE MODEL =====================#
####################################################################
import numpy as np
import pandas as pd
import tensorflow as tf
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from transformers import DistilRobertaTokenizerFast, TFDistilRobertaForSequenceClassification

# ===================== Load Data =====================
df = pd.read_csv('data/final_selected_features_data.csv')
urls = df['url'].tolist()
y = df['phishing'].values.astype(np.float32)

# ===================== Tokenizer & Model =====================
tokenizer = DistilRobertaTokenizerFast.from_pretrained("distilroberta-base")
model_name = "distilroberta-base"

# ===================== Encode Inputs =====================
X_encoded = tokenizer(urls, padding=True, truncation=True, max_length=200, return_tensors='np')
input_ids = X_encoded['input_ids']
attention_mask = X_encoded['attention_mask']

# ===================== Train-Test Split =====================
X_ids_train, X_ids_test, X_mask_train, X_mask_test, y_train, y_test = train_test_split(
    input_ids, attention_mask, y, test_size=0.2, stratify=y, random_state=42
)

# ===================== Cross-Validation =====================
NUM_FOLDS = 4
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
train_metrics = []
fold_histories = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_ids_train, y_train), 1):
    print(f"\n📦 Fold {fold}")

    X_id_tr, X_id_val = X_ids_train[train_idx], X_ids_train[val_idx]
    X_mask_tr, X_mask_val = X_mask_train[train_idx], X_mask_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    model = TFDistilRobertaForSequenceClassification.from_pretrained(model_name)
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    model.compile(optimizer=optimizer, loss=model.compute_loss, metrics=['accuracy'])

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    history = model.fit(
        {'input_ids': X_id_tr, 'attention_mask': X_mask_tr}, y_tr,
        validation_data=({'input_ids': X_id_val, 'attention_mask': X_mask_val}, y_val),
        epochs=20, batch_size=64, callbacks=[early_stopping], verbose=1
    )
    fold_histories.append(history.history)

    y_val_pred = (model.predict({'input_ids': X_id_val, 'attention_mask': X_mask_val}).logits > 0).astype(int)

    metrics = {
        'Fold': f'Fold {fold}',
        'Accuracy': accuracy_score(y_val, y_val_pred),
        'Precision': precision_score(y_val, y_val_pred),
        'Recall': recall_score(y_val, y_val_pred),
        'F1 Score': f1_score(y_val, y_val_pred),
        'AUC-ROC': roc_auc_score(y_val, y_val_pred)
    }
    train_metrics.append(metrics)

# ===================== Save Training Metrics =====================
train_metrics_df = pd.DataFrame(train_metrics)
avg = train_metrics_df.mean(numeric_only=True)
avg['Fold'] = 'Average'
train_metrics_df = pd.concat([train_metrics_df, pd.DataFrame([avg])], ignore_index=True)
print(train_metrics_df)

# ===================== Final Test Evaluation =====================
y_test_probs = model.predict({'input_ids': X_ids_test, 'attention_mask': X_mask_test}).logits
y_test_probs = tf.nn.sigmoid(y_test_probs).numpy().flatten()
y_test_pred = (y_test_probs > 0.5).astype(int)

cm = confusion_matrix(y_test, y_test_pred)
test_metrics_df = pd.DataFrame([{
    'Set': 'Test',
    'Accuracy': accuracy_score(y_test, y_test_pred),
    'Precision': precision_score(y_test, y_test_pred),
    'Recall': recall_score(y_test, y_test_pred),
    'F1 Score': f1_score(y_test, y_test_pred),
    'AUC-ROC': roc_auc_score(y_test, y_test_probs),
}])
print("✅ DistilRoBERTa Test evaluation complete.")

# ===================== Plot Fold Curves =====================
def plot_fold_learning_curves(histories, metric, save_path, title_prefix):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.ravel()
    labels = ['(a)', '(b)', '(c)', '(d)']
    for i, history in enumerate(histories):
        axs[i].plot(history[metric], label='Training', color='blue')
        axs[i].plot(history[f'val_{metric}'], label='Validation', color='red')
        axs[i].set_title(f"{labels[i]} Fold {i+1} {title_prefix} Curve", loc='left', fontsize=10)
        axs[i].set_xlabel('Epochs')
        axs[i].set_ylabel(metric.capitalize())
        axs[i].legend()
        axs[i].grid(True, alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=1000)
    plt.close()
    print(f"✅ {title_prefix} curves saved to {save_path}")

def save_confusion_matrix(cm, path="output/confusion_matrix.png", title="Confusion Matrix"):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Legit', 'Phish'], yticklabels=['Legit', 'Phish'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=1000)
    plt.close()
    print(f"✅ Confusion matrix saved to {path}")

def save_roc_curve(y_true, y_probs, path="output/roc_curve.png", title="Receiver Operating Characteristic (ROC) Curve", model_name="RoBERTa"):
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc_score = roc_auc_score(y_true, y_probs)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_score:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate / Specificity")
    plt.ylabel("True Positive Rate / Sensitivity")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=1000)
    plt.close()
    print(f"✅ ROC curve saved to {path}")

def save_metrics_table(df, path="output/metrics_table.png", title="Metrics Table"):
    fig, ax = plt.subplots(figsize=(max(6, df.shape[1]*1.5), max(1.5, df.shape[0]*0.5)))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.round(4).values,
                     colLabels=df.columns,
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=500)
    plt.close()
    print(f"✅ Table saved to {path}")

save_metrics_table(train_metrics_df, path="output/distilroberta_train_metrics.png", title="DistilRoBERTa Training Metrics")
save_metrics_table(test_metrics_df, path="output/distilroberta_test_metrics.png", title="DistilRoBERTa Test Set Metrics")
save_confusion_matrix(cm, "output/roberta_confusion_matrix_test.png")
save_roc_curve(y_test, y_test_probs, path="output/roberta_roc_curve_test.png")
plot_fold_learning_curves(fold_histories, 'accuracy', 'output/distilroberta_folds_accuracy_curves.png', 'Accuracy')
plot_fold_learning_curves(fold_histories, 'loss', 'output/distilroberta_folds_loss_curves.png', 'Loss')
np.save('output/ROBERTA_fold_histories.npy', fold_histories)
np.save("output/ROBERTA_y_test_probs.npy", y_test_probs)
np.save("output/ROBERTA_y_test_true.npy", y_test)
np.save("output/ROBERTA_y_test_pred.npy", y_test_pred)

