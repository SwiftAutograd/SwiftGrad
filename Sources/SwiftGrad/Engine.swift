import Foundation

/// A single scalar value that tracks its computational graph for automatic differentiation.
/// This is the core building block — equivalent to micrograd's `Value` class.
public final class Value {

    /// The scalar data this node holds.
    public var data: Double

    /// The gradient (derivative) of the final output with respect to this value.
    public var grad: Double = 0.0

    /// The operation that produced this node (for debugging / visualization).
    public let op: String

    /// The child nodes that were combined to produce this node.
    public let children: [Value]

    /// Closure that computes the local gradient contribution during backprop.
    var _backward: () -> Void = {}

    // MARK: - Init

    public init(_ data: Double, children: [Value] = [], op: String = "") {
        self.data = data
        self.children = children
        self.op = op
    }

    // MARK: - Core Operations

    /// Addition: self + other
    public func adding(_ other: Value) -> Value {
        let out = Value(data + other.data, children: [self, other], op: "+")
        out._backward = { [weak self, weak other, weak out] in
            guard let self, let other, let out else { return }
            self.grad += out.grad
            other.grad += out.grad
        }
        return out
    }

    /// Multiplication: self * other
    public func multiplying(_ other: Value) -> Value {
        let out = Value(data * other.data, children: [self, other], op: "*")
        out._backward = { [weak self, weak other, weak out] in
            guard let self, let other, let out else { return }
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        }
        return out
    }

    /// Power: self ** exponent
    public func power(_ exponent: Double) -> Value {
        let out = Value(pow(data, exponent), children: [self], op: "**\(exponent)")
        let base = data
        out._backward = { [weak self, weak out] in
            guard let self, let out else { return }
            self.grad += (exponent * pow(base, exponent - 1)) * out.grad
        }
        return out
    }

    /// ReLU activation: max(0, self)
    public func relu() -> Value {
        let out = Value(data < 0 ? 0 : data, children: [self], op: "ReLU")
        out._backward = { [weak self, weak out] in
            guard let self, let out else { return }
            self.grad += (out.data > 0 ? 1.0 : 0.0) * out.grad
        }
        return out
    }

    /// Tanh activation
    public func tanh() -> Value {
        let t = Darwin.tanh(data)
        let out = Value(t, children: [self], op: "tanh")
        out._backward = { [weak self, weak out] in
            guard let self, let out else { return }
            self.grad += (1 - out.data * out.data) * out.grad
        }
        return out
    }

    /// Exponential: e^self
    public func exp() -> Value {
        let e = Darwin.exp(data)
        let out = Value(e, children: [self], op: "exp")
        out._backward = { [weak self, weak out] in
            guard let self, let out else { return }
            self.grad += out.data * out.grad
        }
        return out
    }

    // MARK: - Backpropagation

    /// Compute gradients for all nodes in the graph via reverse-mode autodiff.
    public func backward() {
        // Build topological order
        var topo: [Value] = []
        var visited = Set<ObjectIdentifier>()

        func buildTopo(_ v: Value) {
            let id = ObjectIdentifier(v)
            guard !visited.contains(id) else { return }
            visited.insert(id)
            for child in v.children {
                buildTopo(child)
            }
            topo.append(v)
        }

        buildTopo(self)

        // Set output gradient to 1 and propagate backwards
        self.grad = 1.0
        for v in topo.reversed() {
            v._backward()
        }
    }
}

// MARK: - Operator Overloads

/// Value + Value
public func + (lhs: Value, rhs: Value) -> Value {
    lhs.adding(rhs)
}

/// Value + Double
public func + (lhs: Value, rhs: Double) -> Value {
    lhs.adding(Value(rhs))
}

/// Double + Value
public func + (lhs: Double, rhs: Value) -> Value {
    Value(lhs).adding(rhs)
}

/// Value * Value
public func * (lhs: Value, rhs: Value) -> Value {
    lhs.multiplying(rhs)
}

/// Value * Double
public func * (lhs: Value, rhs: Double) -> Value {
    lhs.multiplying(Value(rhs))
}

/// Double * Value
public func * (lhs: Double, rhs: Value) -> Value {
    Value(lhs).multiplying(rhs)
}

/// Negation: -Value
public prefix func - (v: Value) -> Value {
    v * (-1.0)
}

/// Value - Value
public func - (lhs: Value, rhs: Value) -> Value {
    lhs + (-rhs)
}

/// Value - Double
public func - (lhs: Value, rhs: Double) -> Value {
    lhs + (-rhs)
}

/// Double - Value
public func - (lhs: Double, rhs: Value) -> Value {
    lhs + (-rhs)
}

/// Value / Value
public func / (lhs: Value, rhs: Value) -> Value {
    lhs * rhs.power(-1)
}

/// Value / Double
public func / (lhs: Value, rhs: Double) -> Value {
    lhs * Value(rhs).power(-1)
}

/// Double / Value
public func / (lhs: Double, rhs: Value) -> Value {
    Value(lhs) * rhs.power(-1)
}

// MARK: - CustomStringConvertible

extension Value: CustomStringConvertible {
    public var description: String {
        "Value(data=\(data), grad=\(grad))"
    }
}
