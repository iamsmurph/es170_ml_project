import qiskit_dbc as QK
import pyquil_dbc as PQ
import vis
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, normalize
import numpy as np
from copy import deepcopy
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

    def get_iris():
        iris_data = np.c_[load_iris().data[0:100,1:2],load_iris().data[0:100,3:4]]

        #iris_data = np.c_[d1, d2]
        scalar = StandardScaler()
        scalar.fit(iris_data)
        data1 = scalar.transform(iris_data)
        ar_data = np.ndarray.tolist(normalize(data1))
        return ar_data

    def get_iris_label():
        return load_iris().target[0:100]
    '''
    a = get_iris()[0:2]
    b = get_iris()[50:52]
    a.append(b[0])
    a.append(b[1])
    c = deepcopy(a)
    '''

    visual = vis.Visual()

    print("Classifying with Rigetti system")
    output1 = np.asarray(feed_the_classifier(get_iris(), PQ_classifier))

    print("Classifying with IBM system")
    output2 = np.asarray(feed_the_classifier(get_iris(), QK_classifier))

    visual.dddplot(output1, get_iris_label())
    visual.dddplot(output2, get_iris_label())
    plt.show()

    #visual.to_file(output)
    #visual.coloured_scatter(output)



    '''
    print("--------------------- PyQuil version ----------------------")

    print(f"Classifying small_dataset = {a} with noisy simulator backend")
    class_result = np.asarray(feed_the_classifier(a, PQ_classifier))
    print(f"Test vector x' was classified as class \n{class_result}\n")

    print("--------------------- QisKit version ----------------------")

    print(f"Classifying x' = {c} with noisy simulator backend")
    class_result1 = np.asarray(feed_the_classifier(c, QK_classifier))
    print(f"Test vector x' was classified as class {class_result1}\n")
    '''

    '''
    print("--------------------- PyQuil version ----------------------")

    print(f"Classifying x' = {x_prime} with noisy simulator backend")
    class_result = PQ_classifier.classify(test_vector=x_prime, training_set=training_set)
    print(f"Test vector x' was classified as class {class_result}\n")

    print('===============================================\n')

    print(f"Classifying x'' = {x_double_prime} with noisy simulator backend")
    class_result = PQ_classifier.classify(test_vector=x_double_prime, training_set=training_set)
    print(f"Test vector x'' was classified as class {class_result}")


    print("--------------------- QisKit version ----------------------")
    print(f"Classifying x' = {x_prime} with noisy simulator backend")
    class_result = QK_classifier.classify(test_vector=x_prime, training_set=training_set)
    print(f"Test vector x' was classified as class {class_result}\n")

    print('===============================================\n')

    print(f"Classifying x'' = {x_double_prime} with noisy simulator backend")
    class_result = QK_classifier.classify(test_vector=x_double_prime, training_set=training_set)
    print(f"Test vector x'' was classified as class {class_result}\n")
    '''


