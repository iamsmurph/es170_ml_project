from pyquil import Program, get_qc
from pyquil.gates import *
import numpy as np

class PQ_DistanceBasedClassifier:

    def create_circuit(self, angles):

        p = Program()
        ro = p.declare("ro", memory_type='BIT', memory_size=4)

        p += H(0)
        p += H(1)

        p += CNOT(0, 2)
        p += RY(-angles[0], 2)
        p += CNOT(0, 2)
        p += RY(angles[0], 2)
        p += X(0)

        # training 1
        p += CCNOT(0, 1, 2)
        p += X(1)

        # training 2
        p += CCNOT(0, 1, 2)
        p += CNOT(1, 2)
        p += RY(angles[1], 2)
        p += CNOT(1, 2)
        p += RY(-angles[1], 2)

        p += CCNOT(0, 1, 2)
        p += CNOT(1, 2)
        p += RY(-angles[1], 2)
        p += CNOT(1, 2)
        p += RY(angles[1], 2)

        # Final Entaglement
        p += CNOT(1, 3)
        p += H(0)

        # Measure
        p += MEASURE(0, ro[0])
        p += MEASURE(1, ro[1])
        p += MEASURE(2, ro[3])
        p += MEASURE(3, ro[2])

        return p

    def simulate(self, program):

        # this function takes the pyquil simulation format and reformats into
        # the style used by qiskit

        def sort(arr):
          (uni, counts) = np.unique(arr, axis=0, return_counts = True)
          a = []
          for i in range(np.shape(uni)[0]):
            a.append(''.join(map(str, (uni[i]).tolist())))
          final = dict(zip(a, counts))
          return final
        # get a simulator here we have a noisy 4 qubit simulator
        qc = get_qc('4q-noisy-qvm')

        # These are the simulation settings
        program.wrap_in_numshots_loop(shots=100)
        comp = qc.compile(program)
        results = qc.run(comp)

        return sort(results)

    def get_angle_for_vec(self, vec):
        return float(np.arccos(vec[0]))*2.0

    def interpret_results(self, result_counts):

        total_samples = sum(result_counts.values())

        # define lambda function that retrieves only results where the ancilla is in the |0> state
        post_select = lambda counts: [(state, occurences) for state, occurences in counts.items() if state[-1] == '0']

        # perform the postselection
        postselection = dict(post_select(result_counts))
        postselected_samples = sum(postselection.values())

        # reads the first bit, if 1 return 1, else return 0
        retrieve_class = lambda binary_class: [occurences for state, occurences in postselection.items() if state[0] == str(binary_class)]

        # Calculate the number of 1 resutls vs number of 2 resutls
        prob_class0 = sum(retrieve_class(0))/postselected_samples
        prob_class1 = sum(retrieve_class(1))/postselected_samples
        return prob_class0, prob_class1

    def classify(self, test_vector, training_set):

        training_vectors = [tuple_[0] for tuple_ in training_set]

        angles = []
        angles.append(self.get_angle_for_vec(test_vector)/2)
        angles.append(self.get_angle_for_vec(training_vectors[1])/4)

        # create the quantum circuit
        program = self.create_circuit(angles=angles)

        # simulate and get the results
        result = self.simulate(program)

        prob_class0, prob_class1 = self.interpret_results(result)

        if prob_class0 > prob_class1:
            return 1
        elif prob_class0 < prob_class1:
            return 0
        # in the unlikely situation that there are exactly as many 1s as 0s
        # return a value 2, this will be dealt with in the display module
        else:
            return 2
