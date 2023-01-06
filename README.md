
# Interior point solver for transportation problem

This repository applies the Mehrotra Predictor-Corrector algorithm to the transportation problem.

The structure of the problem allows for much faster matrix inversion, yielding a nearly linear time algorithm when the number of destinations is small.

Initial algorithm implementation is taken from https://github.com/martinResearch/PySparseLP

