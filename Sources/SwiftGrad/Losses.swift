import Foundation

// MARK: - Loss Functions

/// Common loss functions for training.
public enum Loss {

    /// Mean Squared Error: (1/n) * Σ(pred - target)²
    public static func mse(predicted: [Value], targets: [Double]) -> Value {
        let n = Double(predicted.count)
        let sumSquares = zip(predicted, targets).reduce(Value(0.0)) { acc, pair in
            let diff = pair.0 - pair.1
            return acc + diff * diff
        }
        return sumSquares * (1.0 / n)
    }

    /// SVM "max-margin" hinge loss (as used in micrograd's demo).
    /// targets should be +1 or -1.
    public static func hingeLoss(predicted: [Value], targets: [Double]) -> Value {
        let n = Double(predicted.count)
        let losses = zip(predicted, targets).map { pred, y in
            let margin = 1.0 - y * pred
            return margin.relu()
        }
        return losses.reduce(Value(0.0), +) * (1.0 / n)
    }
}

// MARK: - SGD Optimizer

/// Simple stochastic gradient descent optimizer.
public final class SGD {

    public let parameters: [Value]
    public var learningRate: Double

    public init(parameters: [Value], learningRate: Double = 0.01) {
        self.parameters = parameters
        self.learningRate = learningRate
    }

    /// Perform one optimization step: p = p - lr * p.grad
    public func step() {
        for p in parameters {
            p.data -= learningRate * p.grad
        }
    }
}
