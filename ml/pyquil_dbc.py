from pyquil import Program, get_qc
from pyquil.gates import *
import numpy as np

class PQ_DistanceBasedClassifier:

    def create_circuit(self, angles):
        """
        Creating the quantum circuit
        by filling in the gaps with the
        defined `angles` that are required
        to load the test and training vectors.
        """

        p = Program()
        ro = p.declare("ro", memory_type='BIT', memory_size=4)

        # name the individual qubits for more clarity
        # ancilla_qubit = [0]
        # index_qubit = [1]
        # data_qubit = [2]
        # class_qubit = [3]

        #######################################
        #START of the state preparation routine

        # put the ancilla and the index qubits into uniform superposition
        p += H(0)
        p += H(1)

        def u3(program, theta, qubit):
            program += RZ(3*np.pi, qubit)
            program += RX(np.pi/2, qubit)
            program += RZ(theta+np.pi, qubit)
            program += RX(np.pi/2, qubit)
            program += RZ(0, qubit)
            return program

        # loading the test vector (which we wish to classify)
        p += CNOT(0, 2)
        p = u3(p, -angles[0], 2)
        p += CNOT(0, 2)
        p = u3(p, angles[0], 2)

        # flipping the ancilla qubit > this moves the input vector to the |0> state of the ancilla
        p += X(0)

        # loading the first training vector
        # [0,1] -> class 0
        # we can load this with a straightforward Toffoli

        p += CCNOT(0, 1, 2)

        # flip the index qubit > moves the first training vector to the |0> state of the index qubit
        p += X(1)

        # loading the second training vector
        # [0.78861, 0.61489] -> class 1

        p += CCNOT(0, 1, 2)

        p += CNOT(1, 2)
        p = u3(p, angles[1], 2)
        p += CNOT(1, 2)
        p = u3(p, -angles[1], 2)

        p += CCNOT(0, 1, 2)

        p += CNOT(1, 2)
        p = u3(p, -angles[1], 2)
        p += CNOT(1, 2)
        p = u3(p, angles[1], 2)

        # END of state preparation routine
        ####################################################

        # at this point we would usually swap the data and class qubit
        # however, we can be lazy and let the Qiskit compiler take care of it

        # flip the class label for training vector #2

        p += CNOT(1, 3)

        #############################################
        # START of the mini distance-based classifier

        # interfere the input vector with the training vectors
        p += H(0)

        # Measure all qubits and record the results in the classical registers
        p += MEASURE(0, ro[0])
        p += MEASURE(1, ro[1])
        p += MEASURE(2, ro[2])
        p += MEASURE(3, ro[3])

        # END of the mini distance-based classifier
        #############################################

        return p

    def simulate(self, program):
        """
        Compile and run the quantum circuit
        on a simulator backend.
        """
        # Create quantum computer (simulation)
        qc = get_qc('4q-qvm')

        # perhaps another way to do it locally?
        # qc = get_qc(QPU_LATTICE_NAME, as_qvm=True)

        #shown somewhere as a different way
        #qvm = QVMconnection()

        # Measure the qubits specified by classical_register (qubits 0 and 1) a number of times
        program.wrap_in_numshots_loop(shots=20)

        comp = qc.compile(program)
        # Compile and run the Program (returns a 20 element array of arrays of qubit results)
        results = qc.run(comp)

        def sort(arr):
          (uni, counts) = np.unique(arr, axis=0, return_counts = True)
          a = []
          for i in range(np.shape(uni)[0]):
            a.append(''.join(map(str, (uni[i]).tolist())))
          final = dict(zip(a, counts))
          return final

        # retrieve the results from the simulation
        return sort(results)

    def get_angles(self, test_vector, training_vectors):
        """
        Return the angles associated with
        the `test_vector` and the `training_vectors`.
        Note: if you want to extend this classifier
        for other test and training vectors you need to
        specify the angles here!
        """
        angles = []

        if test_vector == [-0.549, 0.836]:
            angles.append(4.30417579487669/2)
        elif test_vector == [0.053 , 0.999]:
            angles.append(3.0357101997648965/2)
        else:
            print('No angle defined for this test vector.')

        if training_vectors[0] == [0, 1] and training_vectors[1] == [0.78861006, 0.61489363]:
            angles.append(1.3245021469658966/4)
        else:
            print('No angles defined for these training vectors.')

        return angles

    def interpret_results(self, result_counts):
        """
        Post-selecting only the results where
        the ancilla was measured in the |0> state.
        Then computing the statistics of the class
        qubit.
        """

        total_samples = sum(result_counts.values())

        # define lambda function that retrieves only results where the ancilla is in the |0> state
        post_select = lambda counts: [(state, occurences) for state, occurences in counts.items() if state[-1] == '0']

        # perform the postselection
        postselection = dict(post_select(result_counts))
        postselected_samples = sum(postselection.values())

        print(f'Ancilla post-selection probability was found to be {postselected_samples/total_samples}')

        retrieve_class = lambda binary_class: [occurences for state, occurences in postselection.items() if state[0] == str(binary_class)]

        prob_class0 = sum(retrieve_class(0))/postselected_samples
        prob_class1 = sum(retrieve_class(1))/postselected_samples

        print(f'Probability for class 0 is {prob_class0}')
        print(f'Probability for class 1 is {prob_class1}')

        return prob_class0, prob_class1

    def classify(self, test_vector, training_set):
        """
        Classifies the `test_vector` with the
        distance-based classifier using the `training_vectors`
        as the training set.
        This functions combines all other functions of this class
        in order to execute the quantum classification.
        """

        # extract training vectors
        training_vectors = [tuple_[0] for tuple_ in training_set]


        # get the angles needed to load the data into the quantum state
        angles = self.get_angles(
                test_vector=test_vector,
                training_vectors=training_vectors
        )

        # create the quantum circuit
        program = self.create_circuit(angles=angles)

        # simulate and get the results
        result = self.simulate(program)

        prob_class0, prob_class1 = self.interpret_results(result)

        if prob_class0 > prob_class1:
            return 0
        elif prob_class0 < prob_class1:
            return 1
        else:
            return 'inconclusive. 50/50 results'
