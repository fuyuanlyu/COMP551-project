from models import logistic, naivebayes
from models import myutility
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from matplotlib import pyplot as plt

k_fold = 5

datasets = ['iris_data_cleaned', 'car_data_cleaned',\
 'adult_data_cleaned', 'ionosphere_cleaned']



def main():
    fig, axes= plt.subplots(2,2,figsize=(10,10))
    fig.subplots_adjust(bottom=-0.8)
    acclr_models=[]*len(datasets)
    accnb_models=[]*len(datasets)
    size_of_datasets=[]*len(datasets)
    for i, dataset in enumerate(datasets):
        print("===", dataset, "===")
        print("Predicting based on logistic regression")
        acclr_model,size_of_dataset = main_lr(dataset)
        print('Predicting based on Naive Bayes')
        accnb_model= main_nb(dataset)
        acclr_models.append(acclr_model)
        accnb_models.append(accnb_model)
        size_of_datasets.append(size_of_dataset)

    axes[0,0].plot(size_of_datasets[0],acclr_models[0],label="LR")
    axes[0,0].plot(size_of_datasets[0],accnb_models[0],label="NB")
    axes[0,0].set_xlabel('number of samples')
    axes[0,0].set_ylabel('Accuracy')
    axes[0,0].set_title('iris')
    axes[0,0].legend(loc='lower right')

    axes[0,1].plot(size_of_datasets[1],acclr_models[1],label="LR")
    axes[0,1].plot(size_of_datasets[1],accnb_models[1],label="NB")
    axes[0,1].set_xlabel('number of samples')
    axes[0,1].set_ylabel('Accuracy')
    axes[0,1].set_title('car')
    axes[0,1].legend(loc='lower right')

    axes[1,0].plot(size_of_datasets[2],acclr_models[2],label="LR")
    axes[1,0].plot(size_of_datasets[2],accnb_models[2],label="NB")
    axes[1,0].set_xlabel('number of samples')
    axes[1,0].set_ylabel('Accuracy')
    axes[1,0].set_title('adult')
    axes[1,0].legend(loc='lower right')

    axes[1,1].plot(size_of_datasets[3],acclr_models[3],label="LR")
    axes[1,1].plot(size_of_datasets[3],accnb_models[3],label="NB")
    axes[1,1].set_xlabel('number of samples')
    axes[1,1].set_ylabel('Accuracy')
    axes[1,1].set_title('inosphere')
    axes[1,1].legend(loc='lower right')
    plt.show()


## Predicting based on logistic regression
def main_lr(dataset):
    data = np.load('Datasets/' + dataset + '.npy')
    np.random.shuffle(data)
    num_samples = data.shape[0]
    size_of_dataset = []
    acclr_model=[]
    acclr_sk=[]
    for m in range(num_samples):
        if ((m % 10 == 0) & (m!=0)):
            data_x = data[:m,:-1]
            data_y = data[:m,-1]
            step = data_x.shape[0] // k_fold
            num_of_class = int(np.max(data_y)+1)
            acc = 0.

            for i in range(k_fold):
                test_x, test_y = data_x[i:i+step], data_y[i:i+step]
                train_x, train_y = np.concatenate((data_x[0:i], \
                        data_x[i+step:]), axis=0), \
                    myutility.convertToOneHot(np.concatenate((data_y[0:i], \
                        data_y[i+step:]), axis=0), num_of_class)

                LR = logistic.logistic_regression(train_x.shape[1], train_y.shape[1])
                LR.fit(train_x, train_y)
                predict_y = LR.predict(test_x)
                acc += np.sum(predict_y == test_y) / test_y.shape[0]
            
            acclr_model.append(acc/k_fold)
            # print(acc/k_fold)

            # acc = 0.
            # for i in range(k_fold):
            #     test_x, test_y = data_x[i:i+step], data_y[i:i+step]
            #     train_x, train_y = np.concatenate((data_x[0:i], \
            #             data_x[i+step:]), axis=0), \
            #         np.concatenate((data_y[0:i], data_y[i+step:]), axis=0)
                
            #     LR2 = LogisticRegression(solver='lbfgs', multi_class='auto')
            #     LR2.fit(train_x, train_y)
            #     predict_y = LR2.predict(test_x)
            #     acc += np.sum(predict_y == test_y) / test_y.shape[0]

            # # print(acc/k_fold)

            # acclr_sk.append(acc/k_fold)

            size_of_dataset.append(m)
    return acclr_model, size_of_dataset

## Predicting based on naive bayes
def main_nb(dataset):
    data = np.load('Datasets/' + dataset + '.npy')
    np.random.shuffle(data)
    num_samples = data.shape[0]
    # accnb_model=[]
    accnb_sk=[] 
    acc = 0.
    for m in range(num_samples):
        if ((m % 10 == 0) & (m!=0)):
            data_x = data[:m,:-1]
            data_y = data[:m,-1]
            step = data_x.shape[0] // k_fold
            num_of_class = int(np.max(data_y)+1)
            acc = 0. 
            # for i in range(k_fold):
            #     test_x, test_y = data_x[i:i+step], data_y[i:i+step]
            #     train_x, train_y = np.concatenate((data_x[0:i], \
            #             data_x[i+step:]), axis=0), \
            #         myutility.convertToOneHot(np.concatenate((data_y[0:i], \
            #             data_y[i+step:]), axis=0), num_of_class)

            #     NB = naivebayes.NaiveBayes(train_x.shape[1], train_y.shape[1])
            #     NB.fit(train_x, train_y)
            #     predict_y = NB.predict(test_x)
            #     acc += np.sum(predict_y == test_y) / test_y.shape[0]
            # # print(acc/k_fold)
            # accnb_model.append(acc/k_fold)

            acc = 0.
            for i in range(k_fold):
                test_x, test_y = data_x[i:i+step], data_y[i:i+step]
                train_x, train_y = np.concatenate((data_x[0:i], data_x[i+step:]), axis=0), \
                    np.concatenate((data_y[0:i], data_y[i+step:]), axis=0)
                
                NB2 = MultinomialNB()
                NB2.fit(train_x, train_y)
                predict_y = NB2.predict(test_x)
                acc += np.sum(predict_y == test_y) / test_y.shape[0]
            # print(acc/k_fold)
            accnb_sk.append(acc/k_fold)

    return accnb_sk





if __name__ == '__main__':
	main()





