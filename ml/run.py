import pyquil_dbc as QK
import pyquil_dbc as PQ

if __name__ == "__main__":

    # initiate an instance of the distance-based classifier
    QK_classifier = QK.QK_DistanceBasedClassifier()
    PQ_classifier = QK.QK_DistanceBasedClassifier()

    x_prime = [-0.549, 0.836] # x' in publication
    x_double_prime = [0.053 , 0.999] # x'' in publication

    # training set must contain tuples: (vector, class)
    training_set = [
        ([0, 1], 0), # class 0 training vector
        ([0.78861006, 0.61489363], 1) # class 1 training vector
    ]

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
