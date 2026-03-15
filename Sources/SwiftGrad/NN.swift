import Foundation

// MARK: - Module Protocol

/// Base protocol for all neural network components (mirrors micrograd's Module).
public protocol Module {
    /// All trainable parameters in this module.
    func parameters() -> [Value]
}

extension Module {
    /// Reset all gradients to zero before a new training step.
    public func zeroGrad() {
        for p in parameters() {
            p.grad = 0
        }
    }
}

// MARK: - Neuron

/// A single neuron: computes w·x + b, optionally followed by an activation.
public final class Neuron: Module {

    public let w: [Value]
    public let b: Value
    public let nonlin: Bool

    public init(inputSize: Int, nonlin: Bool = true) {
        self.w = (0..<inputSize).map { _ in Value(Double.random(in: -1...1)) }
        self.b = Value(0)
        self.nonlin = nonlin
    }

    /// Forward pass.
    public func callAsFunction(_ x: [Value]) -> Value {
        // w·x + b
        let act = zip(w, x).reduce(b) { acc, pair in
            acc + pair.0 * pair.1
        }
        return nonlin ? act.relu() : act
    }

    public func parameters() -> [Value] {
        w + [b]
    }
}

extension Neuron: CustomStringConvertible {
    public var description: String {
        "\(nonlin ? "ReLU" : "Linear")Neuron(\(w.count))"
    }
}

// MARK: - Layer

/// A layer of neurons.
public final class Layer: Module {

    public let neurons: [Neuron]

    public init(inputSize: Int, outputSize: Int, nonlin: Bool = true) {
        self.neurons = (0..<outputSize).map { _ in Neuron(inputSize: inputSize, nonlin: nonlin) }
    }

    /// Forward pass - returns array of outputs, one per neuron.
    public func callAsFunction(_ x: [Value]) -> [Value] {
        neurons.map { $0(x) }
    }

    public func parameters() -> [Value] {
        neurons.flatMap { $0.parameters() }
    }
}

extension Layer: CustomStringConvertible {
    public var description: String {
        "Layer of [\(neurons.map(\.description).joined(separator: ", "))]"
    }
}

// MARK: - MLP (Multi-Layer Perceptron)

/// A multi-layer perceptron: a sequence of layers.
public final class MLP: Module {

    public let layers: [Layer]

    /// Create an MLP.
    /// - Parameters:
    ///   - inputSize: Number of input features.
    ///   - layerSizes: Number of neurons in each subsequent layer.
    ///     The last layer is linear (no activation); all others use ReLU.
    public init(inputSize: Int, layerSizes: [Int]) {
        let sizes = [inputSize] + layerSizes
        self.layers = (0..<layerSizes.count).map { i in
            Layer(
                inputSize: sizes[i],
                outputSize: sizes[i + 1],
                nonlin: i != layerSizes.count - 1
            )
        }
    }

    /// Forward pass.
    public func callAsFunction(_ x: [Value]) -> [Value] {
        var current = x
        for layer in layers {
            current = layer(current)
        }
        return current
    }

    /// Convenience for single-output networks.
    public func forward(_ x: [Value]) -> Value {
        callAsFunction(x)[0]
    }

    public func parameters() -> [Value] {
        layers.flatMap { $0.parameters() }
    }
}

extension MLP: CustomStringConvertible {
    public var description: String {
        "MLP of [\(layers.map(\.description).joined(separator: ", "))]"
    }
}
