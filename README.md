# Supervised Clustering on MNIST dataset with Rigetti PyQuil and IBM Qiskit

### How to use:
1. Must have a couple modules installed: mpl_toolkits.mplot3d, matplotlib.pyplot,
   sklearn.datasets, sklearn.preprocessing, numpy, qiskit, pyquil.
2. Start pyquil servers, using:
'''
qvm -S
quilc -S
'''
3. Run run.py with python3


### Main Files Contained in ml Folder:

* PyQuil_DBC.py: Our Pyquil implementation of Schuld's paper
* Qiskit_DBC.py: Our Qiskit implementation of Schuld's paper
* Run.py: Runs both the Pyquil and Qiskit implementations side by side to compare results on simulated quantum computer
