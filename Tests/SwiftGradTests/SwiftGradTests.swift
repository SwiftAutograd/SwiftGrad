import Testing
@testable import SwiftGrad

// MARK: - Engine Tests

@Test func testValueAddition() {
    let a = Value(2.0)
    let b = Value(3.0)
    let c = a + b
    #expect(c.data == 5.0)

    c.backward()
    #expect(a.grad == 1.0)
    #expect(b.grad == 1.0)
}

@Test func testValueMultiplication() {
    let a = Value(2.0)
    let b = Value(-3.0)
    let c = a * b
    #expect(c.data == -6.0)

    c.backward()
    #expect(a.grad == -3.0) // dc/da = b
    #expect(b.grad == 2.0)  // dc/db = a
}

@Test func testReLU() {
    let a = Value(-2.0)
    let b = Value(3.0)
    let ra = a.relu()
    let rb = b.relu()
    #expect(ra.data == 0.0)
    #expect(rb.data == 3.0)
}

@Test func testBackpropChain() {
    // Replicate the micrograd sanity check:
    // a = Value(-4.0); b = Value(2.0)
    // c = a + b;  d = a * b + b**3
    // c += c + 1; c += 1 + c + (-a)
    // d += d * 2 + (b + a).relu()
    // d += 3 * d + (b - a).relu()
    // e = c - d; f = e**2
    // g = f / 2.0; g += 10.0 / f
    // g.backward()
    let a = Value(-4.0)
    let b = Value(2.0)
    var c = a + b
    var d = a * b + b.power(3)
    c = c + c + 1.0
    c = c + 1.0 + c + (-a)
    d = d + d * 2.0 + (b + a).relu()
    d = d + 3.0 * d + (b - a).relu()
    let e = c - d
    let f = e.power(2)
    var g = f / 2.0
    g = g + 10.0 / f
    g.backward()

    // Expected values from micrograd / PyTorch
    #expect(abs(g.data - 24.7041) < 0.001)
    #expect(abs(a.grad - 138.8338) < 0.001)
    #expect(abs(b.grad - 645.5773) < 0.001)
}

// MARK: - NN Tests

@Test func testNeuronForward() {
    let n = Neuron(inputSize: 3)
    let x: [Value] = [Value(1.0), Value(2.0), Value(3.0)]
    let out = n(x)
    // Just check it produces a value (exact result depends on random weights)
    #expect(out.data.isFinite)
}

@Test func testMLPForwardAndBackward() {
    let model = MLP(inputSize: 3, layerSizes: [4, 4, 1])
    let x: [Value] = [Value(2.0), Value(3.0), Value(-1.0)]
    let out = model.forward(x)

    #expect(out.data.isFinite)

    // Should have parameters
    let params = model.parameters()
    #expect(params.count == (3*4 + 4) + (4*4 + 4) + (4*1 + 1)) // 41

    // Backward should not crash
    out.backward()

    // At least some gradients should be non-zero
    let hasNonZeroGrad = params.contains { $0.grad != 0 }
    #expect(hasNonZeroGrad)
}

@Test func testZeroGrad() {
    let model = MLP(inputSize: 2, layerSizes: [2, 1])
    let x: [Value] = [Value(1.0), Value(2.0)]
    let out = model.forward(x)
    out.backward()

    model.zeroGrad()

    for p in model.parameters() {
        #expect(p.grad == 0.0)
    }
}

// MARK: - Training Loop Test

@Test func testTrainingReducesLoss() {
    // Simple regression: learn f(x) ≈ 1.0 for a few inputs
    let model = MLP(inputSize: 2, layerSizes: [4, 1])
    let optimizer = SGD(parameters: model.parameters(), learningRate: 0.05)

    let xs: [[Value]] = [
        [Value(1.0), Value(2.0)],
        [Value(2.0), Value(3.0)],
        [Value(-1.0), Value(0.5)],
    ]
    let targets = [1.0, 1.0, 1.0]

    var firstLoss: Double = 0
    var lastLoss: Double = 0

    for epoch in 0..<50 {
        // Forward
        let preds = xs.map { model.forward($0) }
        let loss = Loss.mse(predicted: preds, targets: targets)

        if epoch == 0 { firstLoss = loss.data }
        if epoch == 49 { lastLoss = loss.data }

        // Backward
        model.zeroGrad()
        loss.backward()

        // Update
        optimizer.step()
    }

    // Loss should decrease
    #expect(lastLoss < firstLoss)
}

// MARK: - Operator Tests

@Test func testOperators() {
    let a = Value(6.0)
    let b = Value(3.0)

    #expect((a - b).data == 3.0)
    #expect((a / b).data == 2.0)
    #expect((-a).data == -6.0)
    #expect((2.0 + a).data == 8.0)
    #expect((2.0 * a).data == 12.0)
    #expect((10.0 - a).data == 4.0)
}
