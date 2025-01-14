import numpy as np

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])
y_or = np.array([0, 1, 1, 1])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train_gate(x, y, epochs=100000, lr=0.01):
    w1, w2, bias = 0.1, 0.9, 0.25
    for _ in range(epochs):
        for i in range(4):
            result = sigmoid(x[i][0] * w1 + x[i][1] * w2 + bias)
            w1 += lr * (y[i] - result) * x[i][0]
            w2 += lr * (y[i] - result) * x[i][1]
            bias += lr * (y[i] - result)
    return w1, w2, bias

def predict_gate(x, w1, w2, bias):
    return [1 if sigmoid(x[i][0] * w1 + x[i][1] * w2 + bias) > 0.5 else 0 for i in range(4)]

w1_and, w2_and, bias_and = train_gate(x, y_and)
w1_or, w2_or, bias_or = train_gate(x, y_or)

print("AND Gate Predictions:")
for i, p in enumerate(predict_gate(x, w1_and, w2_and, bias_and)):
    print(f"Input: {x[i]} -> Predicted: {p}")

print("\nOR Gate Predictions:")
for i, p in enumerate(predict_gate(x, w1_or, w2_or, bias_or)):
    print(f"Input: {x[i]} -> Predicted: {p}")
