# SwiftGrad

A tiny, fully functional **autograd engine** and **neural network library** in pure Swift. Inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd).

SwiftGrad implements reverse-mode automatic differentiation (backpropagation) over a dynamically built computation graph. It's the same algorithm that powers PyTorch and TensorFlow - just on scalars instead of tensors, in ~250 lines of Swift.

## Demos

SwiftGrad powers [SwiftRL](https://github.com/SwiftAutograd/SwiftRL), an on-device reinforcement learning library. Every gradient computation in these demos flows through SwiftGrad's `backward()`.

### GridWorld - DQN

A Deep Q-Network learns to navigate from corner to corner on a grid. The agent starts with random exploration (epsilon = 1.0) and gradually shifts to exploiting learned Q-values. This is the same algorithm DeepMind used to play Atari games at superhuman level (Nature, 2015) - running entirely on-device in pure Swift.

![GridWorld Demo](gridworld-demo.gif)

### Snake - DQN

The agent observes 11 features (danger in 3 directions, current heading, food direction) and chooses to go straight, turn left, or turn right. It learns to chase food and avoid walls and its own body through trial and error. This demonstrates that SwiftGrad's autograd engine can handle real game AI training - the kind of adaptive NPC behavior that game studios currently need Python and cloud GPUs for.

![Snake Demo](snake-demo.gif)

### CartPole - REINFORCE

A pole is balanced on a cart using the REINFORCE policy gradient algorithm. The agent receives 4 continuous observations (position, velocity, angle, angular velocity) and must decide to push left or right each frame. Unlike DQN which learns a value function, REINFORCE directly optimizes the policy by differentiating through the log-probability of actions - this is only possible because SwiftGrad tracks gradients through softmax and log operations automatically.

![CartPole Demo](cartpole-demo.gif)

### Why on-device training matters

These demos are not pre-trained models running inference. The neural networks start with random weights and **learn from scratch on your Mac** in seconds. This matters because:

- **Privacy**: An RL agent that adapts to a user's playstyle never needs to send behavior data to a server
- **Personalization**: Every user gets a unique agent trained on their own interactions, not a one-size-fits-all model
- **Latency**: Policy updates happen within the app's process - no network round-trip to a cloud GPU
- **Offline**: Training works without an internet connection, on planes, in subways, anywhere

Today these run on scalars. Replace the `Value` type with a tensor backed by Accelerate or Metal, and the same algorithms scale to production game AI on iPhone and Vision Pro. The architecture is the hard part - and it's done.

<details>
<summary>CLI demo (terminal)</summary>

![SwiftRL CLI Demo](swiftrl-demo.gif)

</details>

## Why SwiftGrad?

- **Educational**: Small enough to read in one sitting, complete enough to train real networks
- **Pure Swift**: Zero dependencies, no Python, no bridging - runs natively on Apple Silicon
- **Foundation**: The autograd engine that powers [SwiftRL](https://github.com/SwiftAutograd/SwiftRL), an on-device reinforcement learning library

## Installation

Add SwiftGrad to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/SwiftAutograd/SwiftGrad.git", from: "0.2.0")
]
```

## Quick Start

```swift
import SwiftGrad

// Create values and build a computation graph
let a = Value(-4.0)
let b = Value(2.0)
let c = a + b
let d = a * b + b.power(3)
let e = c - d
let f = e.power(2)

// Compute gradients automatically
f.backward()

print(a.grad) // df/da
print(b.grad) // df/db
```

### Train a Neural Network

```swift
import SwiftGrad

// 2-layer MLP: 3 inputs, 4 hidden, 1 output
let model = MLP(inputSize: 3, layerSizes: [4, 4, 1])
let optimizer = SGD(parameters: model.parameters(), learningRate: 0.05)

let xs: [[Value]] = [
    [Value(2.0), Value(3.0), Value(-1.0)],
    [Value(3.0), Value(-1.0), Value(0.5)],
    [Value(0.5), Value(1.0), Value(1.0)],
    [Value(1.0), Value(1.0), Value(-1.0)],
]
let targets = [1.0, -1.0, -1.0, 1.0]

for epoch in 0..<100 {
    let predictions = xs.map { model.forward($0) }
    let loss = Loss.mse(predicted: predictions, targets: targets)

    model.zeroGrad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0 {
        print("Epoch \(epoch): loss = \(loss.data)")
    }
}
```

## Architecture

SwiftGrad maps 1:1 to micrograd's architecture:

| Component | File | Lines | Description |
|---|---|---|---|
| **Engine** | `Engine.swift` | ~150 | `Value` class with autograd, tracks computation graph, runs `backward()` via topological sort |
| **Neural Net** | `NN.swift` | ~80 | `Neuron`, `Layer`, `MLP` built on `Value`, uses `callAsFunction` for Pythonic syntax |
| **Training** | `Losses.swift` | ~40 | `Loss.mse`, `Loss.hingeLoss`, `SGD` optimizer |

### How It Works

Every `Value` remembers how it was created. When you write `a + b`, the result stores references to `a` and `b` as children, along with the operation (`+`) and a closure that knows how to compute the local gradient.

Calling `.backward()` on any value:
1. Builds a topological ordering of the entire computation graph
2. Sets the output gradient to 1.0
3. Walks the graph in reverse, applying the chain rule at each node

This is the exact same algorithm (reverse-mode autodiff) used by PyTorch, JAX, and every modern ML framework.

### Swift-Specific Design Choices

- **`Value` is a `class`** (reference type) - nodes in the computation graph are shared and aliased, just like Python
- **Operator overloads** (`+`, `*`, `-`, `/`) replace Python's `__add__`, `__mul__`, etc.
- **`callAsFunction`** enables `neuron(x)` and `model(x)` syntax, matching Python's `__call__`
- **`weak` captures** in `_backward` closures prevent retain cycles in the computation graph
- **`Module` protocol** with `parameters()` and `zeroGrad()` provides a clean training interface

### Operations Supported

| Operation | Forward | Backward (gradient) |
|---|---|---|
| `a + b` | `a.data + b.data` | `dout/da = 1`, `dout/db = 1` |
| `a * b` | `a.data * b.data` | `dout/da = b`, `dout/db = a` |
| `a.power(n)` | `a.data^n` | `dout/da = n * a^(n-1)` |
| `a.relu()` | `max(0, a.data)` | `dout/da = (a > 0) ? 1 : 0` |
| `a.tanh()` | `tanh(a.data)` | `dout/da = 1 - tanh^2(a)` |
| `a.exp()` | `e^a.data` | `dout/da = e^a` |
| `a.log()` | `ln(a.data)` | `dout/da = 1/a` |
| `-a`, `a - b`, `a / b` | Composed from above | Chain rule through composition |

## Tests

```bash
swift test
```

9 tests covering:
- Value addition, multiplication, ReLU
- Full backpropagation chain (verified against PyTorch)
- Neuron/MLP forward and backward passes
- Zero gradient reset
- Training loop convergence

## Part of the SwiftAutograd Organization

| Repository | Description | Status |
|---|---|---|
| **[SwiftGrad](https://github.com/SwiftAutograd/SwiftGrad)** | Autograd engine (you are here) | v0.2.0 |
| [SwiftRL](https://github.com/SwiftAutograd/SwiftRL) | On-device reinforcement learning | v0.2.0 |
| [SwiftRLDemos](https://github.com/SwiftAutograd/SwiftRLDemos) | Demo apps (GridWorld, Snake, CartPole) | In development |

## Acknowledgments

- [micrograd](https://github.com/karpathy/micrograd) by Andrej Karpathy - the direct inspiration for this project
- [Karpathy's micrograd lecture](https://www.youtube.com/watch?v=VMj-3S1tku0) - "The spelled-out intro to neural networks and backpropagation"

## License

MIT - see [LICENSE](LICENSE).
