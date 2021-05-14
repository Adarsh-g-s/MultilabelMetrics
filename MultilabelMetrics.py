
'Without using any external libraries, writing the common machine learning metrics for multi-label classification'
'TP, TN, FP, FN for the corresponding classes are below'

'---For class A---'
'TP for A'
def true_positive_A(y_true, y_pred):
    tp = 0

    for yt, yp in zip(y_true, y_pred):
        if yt == "A" and yp == "A":
            tp += 1
    return tp

'TN for A'
def true_negative_A(y_true, y_pred):
    tn = 0

    for yt, yp in zip(y_true, y_pred):
        if yt == "B" and yp == "B" or yt == "C" and yp == "C":
        # if yt == 0 and yp == 0:
            tn += 1
    return tn

'FP for A'
def false_positive_A(y_true, y_pred):
    fp = 0

    for yt, yp in zip(y_true, y_pred):
        if yt != "A" and yp == "A":
            fp += 1
    return fp

'FN for A'
def false_negative_A(y_true, y_pred):
    fn = 0

    for yt, yp in zip(y_true, y_pred):
        if yt == "A" and yp != "A":
            fn += 1
    return fn


'---For class B---'
'TP for B'
def true_positive_B(y_true, y_pred):
    tp = 0

    for yt, yp in zip(y_true, y_pred):
        if yt == "B" and yp == "B":
            tp += 1
    return tp

'TN for B'
def true_negative_B(y_true, y_pred):
    tn = 0

    for yt, yp in zip(y_true, y_pred):
        if yt == "A" and yp == "A" or yt == "B" and yp == "B":
        # if yt == 0 and yp == 0:
            tn += 1
    return tn

'FP for B'
def false_positive_B(y_true, y_pred):
    fp = 0

    for yt, yp in zip(y_true, y_pred):
        if yt != "B" and yp == "B":
            fp += 1
    return fp

'FN for B'
def false_negative_B(y_true, y_pred):
    fn = 0

    for yt, yp in zip(y_true, y_pred):
        if yt == "B" and yp != "B":
            fn += 1
    return fn

'---For class C---'
'TP for C'
def true_positive_C(y_true, y_pred):
    tp = 0

    for yt, yp in zip(y_true, y_pred):
        if yt == "C" and yp == "C":
            tp += 1
    return tp

'TN for C'
def true_negative_C(y_true, y_pred):
    tn = 0

    for yt, yp in zip(y_true, y_pred):
        if yt == "A" and yp == "A" or yt == "B" and yp == "B":
        # if yt == 0 and yp == 0:
            tn += 1
    return tn

'FP for C'
def false_positive_C(y_true, y_pred):
    fp = 0

    for yt, yp in zip(y_true, y_pred):
        if yt != "C" and yp == "C":
            fp += 1
    return fp

'FN for C'
def false_negative_C(y_true, y_pred):
    fn = 0

    for yt, yp in zip(y_true, y_pred):
        if yt == "C" and yp != "C":
            fn += 1
    return fn

'--- Precision and Recall for the respective classes ---'
'Precision for A'
def precision_A(y_true, y_pred):
    tp = true_positive_A(y_true, y_pred)
    fp = false_positive_A(y_true, y_pred)
    precision_A = tp / (tp + fp)
    return precision_A

'Recall for A'
def recall_A(y_true, y_pred):
    tp = true_positive_A(y_true, y_pred)
    fn = false_negative_A(y_true, y_pred)
    recall_A = tp / (tp + fn)
    return recall_A

'Precision for B'
def precision_B(y_true, y_pred):
    tp = true_positive_B(y_true, y_pred)
    fp = false_positive_B(y_true, y_pred)
    precision_B = tp / (tp + fp)
    return precision_B

'Recall for B'
def recall_B(y_true, y_pred):
    tp = true_positive_B(y_true, y_pred)
    fn = false_negative_B(y_true, y_pred)
    recall_B = tp / (tp + fn)
    return recall_B

'Precision for C'
def precision_C(y_true, y_pred):
    tp = true_positive_C(y_true, y_pred)
    fp = false_positive_C(y_true, y_pred)
    precision_C = tp / (tp + fp)
    return precision_C

'Recall for C'
def recall_C(y_true, y_pred):
    tp = true_positive_C(y_true, y_pred)
    fn = false_negative_C(y_true, y_pred)
    recall_C = tp / (tp + fn)
    return recall_C

'F1 score for A'
def f1_A(y_true, y_pred):
    p = precision_A(y_true, y_pred)
    r = recall_A(y_true, y_pred)
    score = 2 * p * r / (p + r)
    return score, p, r

'F1 score for B'
def f1_B(y_true, y_pred):
    p = precision_B(y_true, y_pred)
    r = recall_B(y_true, y_pred)
    score = 2 * p * r / (p + r)
    return score,  p, r

'F1 score for C'
def f1_C(y_true, y_pred):
    p = precision_C(y_true, y_pred)
    r = recall_C(y_true, y_pred)
    score = 2 * p * r / (p + r)
    return score, p, r

'Weighted F1/ Macro avrege loss function'
def compute_weighted_f1(f1_scores, f1_weights, weighted_F1):
    for key in f1_scores:
        value = f1_scores[key] * f1_weights[key]
        # print(value)
        weighted_F1 = weighted_F1 + value
    return weighted_F1



def weighted_f1(y_true, y_pred, f1_weights):

    'F1, precision and recall score for A'
    f1_score_A, precision_A, recall_A = f1_A(y_true, y_pred)

    'F1 , precision and recall score for B'
    f1_score_B, precision_B, recall_B = f1_B(y_true, y_pred)

    'F1, precision and recall score for C'
    f1_score_C, precision_C, recall_C = f1_C(y_true, y_pred)

    # print("Precision {},{},{}".format(precision_A, precision_B, precision_C))
    # print("Recall {},{},{}".format(recall_A, recall_B, recall_C))
    # print("F1 Score {},{},{}".format(f1_score_A, f1_score_B, f1_score_C))

    precision_scores = {'A': precision_A, 'B': precision_B, 'C': precision_C}
    recall_scores = {'A': recall_A, 'B': recall_B, 'C': recall_C}
    f1_scores = {'A':f1_score_A, 'B': f1_score_B, 'C': f1_score_C}

    # sum_f1_scores = f1_score_A+f1_score_B+f1_score_C
    # print(sum_f1_scores)
    # weighted_F1 = np.average(sum_f1_scores, weights = f1_weights)



    weighted_F1 = 0

    weighted_F1 = compute_weighted_f1(f1_scores, f1_weights, weighted_F1)

    # print(weighted_F1)

    result = {'precision':precision_scores,
              'recall':recall_scores,
              'F1':f1_scores,
              'weighted_F1':weighted_F1}

    print(result)
    return result

weighted_f1(y_true = ["A", "B", "C", "A", "A", "B", "A", "C", "A", "A", "B", "C", "C"],
         y_pred = ["A", "B", "C", "A", "B", "C", "B", "C", "A", "A", "B", "C", "C"], f1_weights = {'A':0.7, 'B':0.2, 'C':0.1})