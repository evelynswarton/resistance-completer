# Effective Resistance Graph Completion

Recover missing edges of a graph from effective resistance measurements using convex optimization.

## Problem

Given a partially complete graph Laplacian and effective resistance measurements for certain vertex pairs, infer the unknown edge weights and return the entire graph Laplacian.

## Method

For every entry in the graph Laplacian that is missing we take exactly one effective resistance measurement.
Then, using convex optimization, we minimize the function that is the difference between the inner product of the inputs and the measurements with the log determinant of the inverse Laplacian corresponding to the the inputs.
This difference is minimized exactly when the inputs are the missing edge weights, due to the Laplacian and all pairs effective resistance matrix varying independently.

## Structure

- `main.py` Main experiment driver.
- `laplace_utils.py` Linear algebra helper functions.
- `graph_utils.py` Graph theoretic helper functions for implementing combinatorial structures.

## Requirements

- Python
- Numpy
- NetworkX
- CVXPY

## Related Work

This implements the final algorithms of section 6 of (Graph Inference with Effective Resistance Queries)[https://proceedings.mlr.press/v313/warton26a.html].

## Status

This is a research prototype, for ongoing research projects under the direction of myself and my advisors. Please use with caution and double check results for any production environments.
