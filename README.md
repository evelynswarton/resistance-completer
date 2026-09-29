# Effective Resistance Graph Completion

### Recover unknown graph structure from effective resistance measurments.

This repository implements the algorithms from Graph Inference with Effective Resistance Queries for reconstructing missing edge weights from resistance-distance measurements.

Given:
  - Partial graph Laplacian information
  - Effective resistance measurement queries 
  - The location of the unknown Laplacian data 
the algorithm reconstructs the missing edge weights via convex optimization

**Paper**: (Graph Inference with Effective Resistance Queries)[https://proceedings.mlr.press/v313/warton26a.html]

**Status**: Research prototype

## Getting Started 

First, install all of the necessary dependencies.
Then,


```
git clone https://github.com/evelynswarton/resistance-completer
cd resistance-completer
python3 main.py
```

## Requirements

- Python
- Numpy
- NetworkX
- CVXPY

## Methods

For every entry in the graph Laplacian that is missing we take exactly one effective resistance measurement.
Then, using convex optimization, we minimize the function that is the difference between the inner product of the inputs and the measurements with the log determinant of the inverse Laplacian corresponding to the the inputs.
This difference is minimized exactly when the inputs are the missing edge weights, due to the Laplacian and all pairs effective resistance matrix varying independently.

## Status

This is a research prototype, for ongoing research projects under the direction of myself and my advisors. Please use with caution and double check results for any production environments.
