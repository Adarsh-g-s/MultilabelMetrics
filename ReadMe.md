
### Custom loss function

It's quite common to write your own loss functions when you are doing ML research. 
In this codebase, a loss function weighted_f1 is written which calculates modifications of the
F1 score for multi-label classification. 

### Challenge

In this code, no external libraries like numpy, pandas, sklearn among others have not been used for writing the loss function.
The loss functions are written from scratch using basic math functions in Python.

#### Input

Given the below input labels, \
y_true = ["A", "B", "C", "A", "A", "B", "A", "C", "A", "A", "B", "C", "C"] \
y_pred = ["A", "B", "C", "A", "B", "C", "B", "C", "A", "A", "B", "C", "C"] \
weights = {'A':0.7, 'B':0.2, 'C':0.1}

Corresponding metrics like Precision, Recall, F1 score is computed for each label and then finally
the macro averaged F1 is computed from the F1 scores of the corrsponding labels.