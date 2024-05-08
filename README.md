# Neural Networks Determination of Nematic Elastic Constants

This notebook presents a collection of Python functions used in a method for determining the nematic elastic constants. The method is described in [J. Zaplotnik, M. Škarabot, M. Ravnik. Neural networks determination of material parameters and structures in nematic complex fluids, submitted in 2023].

## Overview

The notebook covers the following steps:
1. **Creation of a Training Set**: Generating a training dataset by simulating liquid crystal dynamics and light transmission.
2. **Random Initial Director Configuration**: Generating random initial director configurations.
3. **Function for Random Function**: Creating a smooth random function discretized in points with specified boundary values.
4. **Training a Simple Sequential Neural Network**: Implementing and training a simple sequential neural network.
5. **Importing Experimentally Measured Data**: Importing experimental data for comparison.
6. **Determining Elastic Constants**: Using the neural network to determine elastic constants of the 5CB liquid crystal.

## Required Python Libraries

- numpy
- scipy
- numba
- matplotlib
- tensorflow
- pandas
- csv
- time

## Creation of a Training Set

The nematic geometry is assumed to be parametrized as n(z,t)=(cosθ(z,t),0,sinθ(z,t)), where the tilt angle θ(z,t) varies in space (along z) and time t, determining the director configuration.

### Random Initial Director Configuration

For each training pair, a random initial state is generated, determined by θ(z,t=0).

The function `random_function(Nnew, AL, AR)` returns an array with a smooth random function discretized in `Nnew` points with `AL` and `AR` values on the left and right boundary, respectively. It is based on 1D quadratic interpolation between random values between `(AL−π/2)` and `(AR+π/2)` at random points within the `[0,1]` interval.

