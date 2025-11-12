#Prepare Data# SCRIPT 06: EVALUATE MODELS
# GOAL: Compare the RAC model vs. the Baseline on the test set.

# 1. Import libraries
#    (e.g., torch, pandas, sklearn.metrics)
#    (from models import ..., from utils import ...)

# 2. Load the metadata and get the *test* split
#    - (e.g., df_target = df[df['dataset_source'] == 'TaoWu'])
#    - You need to have a held-out test set (e.g., from a CSV)
#    - test_dataset = FCDataset(df_test)
#    - test_dataloader = DataLoader(test_dataset, ...)

# 3. Load the trained RAC model
#    - (Initialize RAC_Model, load 'rac_model.pth')
#    - (Also need to load GAE encoder and FAISS index for its forward pass)
#    - rac_model.eval()

# 4. Load the trained Baseline model
#    - (Initialize Baseline_GNN, load 'baseline_model.pth')
#    - baseline_model.eval()

# 5. Initialize lists to store results
#    - all_labels = []
#    - rac_predictions = []
#    - baseline_predictions = []

# 6. Start the evaluation loop
#    - with torch.no_grad():
#    -   for (data, label) in test_dataloader:
#    -     # ... 1. Get RAC prediction (requires retrieval step)
#    -     #      rac_pred = rac_model(...)
#    -     # ... 2. Get Baseline prediction
#    -     #      baseline_pred = baseline_model(data)
#    -     # ... 3. Store results
#    -     #      all_labels.append(label)
#    -     #      rac_predictions.append(rac_pred)
#    -     #      baseline_predictions.append(baseline_pred)

# 7. Post-process results (convert logits to probabilities/classes)
#    - ...

# 8. Calculate and print metrics
#    - (e.g., from sklearn.metrics import accuracy_score, roc_auc_score)
#    - print("--- RAC Model Results ---")
#    - print(f"Accuracy: {accuracy_score(all_labels, rac_classes)}")
#    - print(f"AUC: {roc_auc_score(all_labels, rac_probs)}")
#    -
#    - print("--- Baseline Model Results ---")
#    - print(f"Accuracy: {accuracy_score(all_labels, baseline_classes)}")
#    - print(f"AUC: {roc_auc_score(all_labels, baseline_probs)}")

# 9. (Optional) Save test results to a file