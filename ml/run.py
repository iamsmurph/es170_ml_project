import qiskit_dbc as QK
import pyquil_dbc as PQ
import vis
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, normalize
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    QK_classifier = QK.QK_DistanceBasedClassifier()
    PQ_classifier = PQ.PQ_DistanceBasedClassifier()
    visual = vis.Visual()

    # these are the values used in the publication, we expanded this area to use
    # against all the iris values
    x_prime = [-0.549, 0.836] # x' in publication
    x_double_prime = [0.053 , 0.999] # x'' in publication

    # training set must contain tuples: (vector, class)
    training_set = [
        ([0, 1], 0), # class 0 training vector
        ([0.78861006, 0.61489363], 1) # class 1 training vector
    ]

    # this function inputs each data point into the quantum classifier
    def feed_the_classifier(dataset, classifier):
        results = []
        for vec in dataset:
            cl = classifier.classify(vec, training_set)
            vec.append(cl)
            results.append(vec)
        return results

    # Gets data set from sklearn server
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

    # Runs iris against rigetti circuit
    print("Classifying with Rigetti system")
    output1 = np.asarray(feed_the_classifier(get_iris(), PQ_classifier))

    # Runs iris against IBM circuit
    print("Classifying with IBM system")
    output2 = np.asarray(feed_the_classifier(get_iris(), QK_classifier))

    # use display module to generate 3d graphs
    visual.dddplot(output1, get_iris_label())
    visual.dddplot(output2, get_iris_label())
    plt.show()


