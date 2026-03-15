// SwiftGrad - A tiny autograd engine and neural network library in Swift.
// Inspired by Andrej Karpathy's micrograd (https://github.com/karpathy/micrograd).
//
// Components:
//   - Engine.swift : Value type with automatic differentiation (reverse-mode autodiff)
//   - NN.swift     : Neuron, Layer, MLP built on top of Value
//   - Losses.swift : Loss functions (MSE, Hinge) and SGD optimizer
