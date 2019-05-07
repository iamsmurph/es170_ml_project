import qiskit_dbc as QK
import pyquil_dbc as PQ
import vis
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, normalize
import numpy as np

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
        iris_data = load_iris().data[0:50,:2]
        data = iris_data
        scalar = StandardScaler()
        scalar.fit(iris_data)
        data1 = scalar.transform(iris_data)
        ar_data = np.ndarray.tolist(normalize(data1))
        return ar_data


    output = np.asarray(feed_the_classifier(get_iris(), PQ_classifier))
    print(output)

    visual = vis.Visual()
    # read = visual.from_file()
    visual.to_file(output)
    visual.coloured_scatter(output)
    visual.dddplot(output)

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

