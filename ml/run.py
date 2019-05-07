import qiskit_dbc as QK
import pyquil_dbc as PQ
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, normalize
import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

if __name__ == "__main__":

    QK_classifier = QK.QK_DistanceBasedClassifier()
    PQ_classifier = PQ.PQ_DistanceBasedClassifier()

    x_prime = [-0.549, 0.836] # x' in publication
    x_double_prime = [0.053 , 0.999] # x'' in publication

    # training set must contain tuples: (vector, class)
    training_set = [
        ([0, 1], 0), # class 0 training vector
        ([0.78861006, 0.61489363], 1) # class 1 training vector
    ]

    def feed_the_classifier(dataset, classifier):
        results = []
        for vec in dataset:
            cl = classifier.classify(vec, training_set)
            vec.append(cl)
            results.append(vec)
        return results

    # print([x_prime])
    # res1 = feed_the_classifier([x_prime], PQ_classifier)
    # print(res1)

    iris_data = load_iris().data[:3,:2]
    data = iris_data
    scalar = StandardScaler()
    scalar.fit(data)
    data1 = scalar.transform(data)
    data2 = np.ndarray.tolist(normalize(data1))

    res = np.asarray(feed_the_classifier(data2, PQ_classifier))
    print(res)

    for el in res[:, 2]:


    df  = pd.DataFrame(res, columns = ['x', 'y', 'label'])
    print(df)
    # df.plot()  # plots all columns against index
    sns.relplot(x="x", y="y", hue="label", data=df);
    # df.plot(kind='density')  # estimate density function
    plt.show()
    '''
    print("--------------------- QisKit version ----------------------")
    print(f"Classifying x' = {x_prime} with noisy simulator backend")
    class_result = QK_classifier.classify(test_vector=x_prime, training_set=training_set)
    print(f"Test vector x'' was classified as class {class_result}\n")

    print('===============================================\n')

    print(f"Classifying x'' = {x_double_prime} with noisy simulator backend")
    class_result = QK_classifier.classify(test_vector=x_double_prime, training_set=training_set)
    print(f"Test vector x' was classified as class {class_result}\n")

    print("--------------------- PyQuil version ----------------------")

    print(f"Classifying x' = {x_prime} with noisy simulator backend")
    class_result = PQ_classifier.classify(test_vector=x_prime, training_set=training_set)
    print(f"Test vector x'' was classified as class {class_result}\n")

    print('===============================================\n')

    print(f"Classifying x'' = {x_double_prime} with noisy simulator backend")
    class_result = PQ_classifier.classify(test_vector=x_double_prime, training_set=training_set)
    print(f"Test vector x' was classified as class {class_result}")
    '''

