def evaluate_binary_predictions(true_y, pred_scores):
    auc = roc_auc_score(true_y, pred_scores)
    acc = accuracy_score(true_y, getLabels(pred_scores))  # using default cutoff of 0.5
    mcc = matthews_corrcoef(true_y, getLabels(pred_scores))
    return auc, mcc, acc
    
score1 = evaluate_binary_predictions(y_result,Y_Pred)
print("AUC:", score1[0])
print("Mcc:", score1[1])
