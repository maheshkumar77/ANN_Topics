import math

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Derivative of sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Training data (X, Y)
X = [0, 1]   # input
Y = [0, 1]   # expected output

# Initialize weights and bias randomly
w1 = 0.5   # input → hidden
b1 = 0.0

w2 = 0.5   # hidden → output
b2 = 0.0

learning_rate = 0.1
epochs = 10

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}")

    for x, y_true in zip(X, Y):

        # -------- FORWARD PROPAGATION --------
        z1 = w1 * x + b1
        a1 = sigmoid(z1)   # hidden neuron output

        z2 = w2 * a1 + b2
        y_pred = sigmoid(z2)  # final output

        # -------- LOSS (MSE) --------
        loss = (y_true - y_pred) ** 2
        print(f"Input: {x}, Predicted: {y_pred:.4f}, Loss: {loss:.4f}")

        # -------- BACKPROPAGATION --------

        # Output layer gradient
        d_loss_y = -2 * (y_true - y_pred)
        d_y_z2 = sigmoid_derivative(y_pred)
        d_z2_w2 = a1

        # Update w2 and b2
        w2 = w2 - learning_rate * d_loss_y * d_y_z2 * d_z2_w2
        b2 = b2 - learning_rate * d_loss_y * d_y_z2

        # Hidden layer gradient
        d_z2_a1 = w2
        d_a1_z1 = sigmoid_derivative(a1)
        d_z1_w1 = x

        # Update w1 and b1
        w1 = w1 - learning_rate * d_loss_y * d_y_z2 * d_z2_a1 * d_a1_z1 * d_z1_w1
        b1 = b1 - learning_rate * d_loss_y * d_y_z2 * d_z2_a1 * d_a1_z1

print("\nFinal weights:")
print("w1:", w1, "b1:", b1)
print("w2:", w2, "b2:", b2)
