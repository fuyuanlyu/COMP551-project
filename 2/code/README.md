The code is organized as follows:

Validation pipeline.ipynb: code for evaluation different models and hyper-parameter tuning

dataset/dataset.py: code for dataset preparation

dataset/IMDB.pickle: IMDB dataset. We transform the original 50000 txt files into a pickle dataset for the sake of simplicity. The transformation is done via dataset/dataset.py:119 prepare_IMDB_dataset() function. No other processing is done in this step.

lstm-exp-architecture/: code for different LSTMs' architechture analysis (include:biLSTM,convLSTM,LSTM)

lstm-exp-hyperparameter/: code for LSTM's hyperparameters analysis (include: epoch, max review length) 


-----------------------------------------

This is the result of different methods on Both dataset:

| Dataset       | LR    | Decision Tree | SVM   | Ada Boost | Random forest | MLP(10, 256) | MLP(30, 1024) | XG Boost | LSTM   |
| ------------- | ----- | ------------- | ----- | --------- | ------------- | ------------ | ------------- | -------- | ------ |
| 20 news group | 0.776 | 0.47          | 0.804 | 0.5       | 0.688         | 0.7058       | 0.7408        | 0.6164   | 0.6761 |
| IMDB Reviews  | 0.883 | 0.702         | 0.877 | 0.83      | 0.8367        | 0.8757       | 0.8801        | 0.7398   | 0.8916 |





