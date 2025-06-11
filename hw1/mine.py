import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# 加载MNIST数据集
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data.astype('float32') / 255.0
y = mnist.target.astype('int64')
X = X.values.reshape(-1, 1, 28, 28)  # 转换为 (N, C, H, W) 格式
y = y.values

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 将标签转换为one-hot编码
def one_hot(y, num_classes):
    return np.eye(num_classes)[y]

y_train_onehot = one_hot(y_train, 10)
y_test_onehot = one_hot(y_test, 10)

# 定义卷积层
class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.1
        self.bias = np.zeros(out_channels)

    def forward(self, x):
        self.input = x
        batch_size, in_channels, in_height, in_width = x.shape
        out_height = (in_height - self.kernel_size + 2 * self.padding) // self.stride + 1
        out_width = (in_width - self.kernel_size + 2 * self.padding) // self.stride + 1
        output = np.zeros((batch_size, self.out_channels, out_height, out_width))

        # 添加padding
        if self.padding > 0:
            x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        else:
            x_padded = x

        # 卷积操作
        for b in range(batch_size):
            for c_out in range(self.out_channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        receptive_field = x_padded[b, :, h_start:h_start+self.kernel_size, w_start:w_start+self.kernel_size]
                        output[b, c_out, h, w] = np.sum(receptive_field * self.weights[c_out]) + self.bias[c_out]
        return output

    def backward(self, grad_output, learning_rate):
        batch_size, in_channels, in_height, in_width = self.input.shape
        grad_input = np.zeros_like(self.input)
        grad_weights = np.zeros_like(self.weights)
        grad_bias = np.zeros_like(self.bias)

        # 添加padding
        if self.padding > 0:
            x_padded = np.pad(self.input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        else:
            x_padded = self.input

        # 计算梯度
        for b in range(batch_size):
            for c_out in range(self.out_channels):
                for h in range(grad_output.shape[2]):
                    for w in range(grad_output.shape[3]):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        receptive_field = x_padded[b, :, h_start:h_start+self.kernel_size, w_start:w_start+self.kernel_size]
                        grad_input[b, :, h_start:h_start+self.kernel_size, w_start:w_start+self.kernel_size] += grad_output[b, c_out, h, w] * self.weights[c_out]
                        grad_weights[c_out] += grad_output[b, c_out, h, w] * receptive_field
                        grad_bias[c_out] += grad_output[b, c_out, h, w]

        # 更新参数
        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias

        return grad_input

# 定义最大池化层
class MaxPool2D:
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        self.input = x
        batch_size, channels, in_height, in_width = x.shape
        out_height = (in_height - self.kernel_size) // self.stride + 1
        out_width = (in_width - self.kernel_size) // self.stride + 1
        output = np.zeros((batch_size, channels, out_height, out_width))

        for b in range(batch_size):
            for c in range(channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        receptive_field = x[b, c, h_start:h_start+self.kernel_size, w_start:w_start+self.kernel_size]
                        output[b, c, h, w] = np.max(receptive_field)
        return output

    def backward(self, grad_output):
        batch_size, channels, in_height, in_width = self.input.shape
        grad_input = np.zeros_like(self.input)

        for b in range(batch_size):
            for c in range(channels):
                for h in range(grad_output.shape[2]):
                    for w in range(grad_output.shape[3]):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        receptive_field = self.input[b, c, h_start:h_start+self.kernel_size, w_start:w_start+self.kernel_size]
                        max_val = np.max(receptive_field)
                        grad_input[b, c, h_start:h_start+self.kernel_size, w_start:w_start+self.kernel_size] += (receptive_field == max_val) * grad_output[b, c, h, w]
        return grad_input

# 定义全连接层
class Linear:
    def __init__(self, in_features, out_features):
        self.weights = np.random.randn(in_features, out_features) * 0.1
        self.bias = np.zeros(out_features)

    def forward(self, x):
        self.input = x
        return np.dot(x, self.weights) + self.bias

    def backward(self, grad_output, learning_rate):
        grad_input = np.dot(grad_output, self.weights.T)
        grad_weights = np.dot(self.input.T, grad_output)
        grad_bias = np.sum(grad_output, axis=0)

        # 更新参数
        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias

        return grad_input

# 定义ReLU激活函数
class ReLU:
    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    def backward(self, grad_output):
        return grad_output * (self.input > 0)

# 定义Softmax交叉熵损失函数
class SoftmaxCrossEntropyLoss:
    def forward(self, x, y):
        self.y = y
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.softmax = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        loss = -np.sum(y * np.log(self.softmax + 1e-8)) / x.shape[0]
        return loss

    def backward(self):
        return (self.softmax - self.y) / self.y.shape[0]

# 定义网络
class Net:
    def __init__(self):
        self.conv1 = Conv2D(1, 32, 3, padding=1)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D(2, 2)
        self.conv2 = Conv2D(32, 64, 3, padding=1)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D(2, 2)
        self.fc1 = Linear(64 * 7 * 7, 128)
        self.relu3 = ReLU()
        self.fc2 = Linear(128, 10)

    def forward(self, x):
        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)
        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)
        x = x.reshape(x.shape[0], -1)  # Flatten
        x = self.fc1.forward(x)
        x = self.relu3.forward(x)
        x = self.fc2.forward(x)
        return x

    def backward(self, grad_output, learning_rate):
        grad_output = self.fc2.backward(grad_output, learning_rate)
        grad_output = self.relu3.backward(grad_output)
        grad_output = self.fc1.backward(grad_output, learning_rate)
        grad_output = grad_output.reshape(grad_output.shape[0], 64, 7, 7)  # Reshape
        grad_output = self.pool2.backward(grad_output)
        grad_output = self.relu2.backward(grad_output)
        grad_output = self.conv2.backward(grad_output, learning_rate)
        grad_output = self.pool1.backward(grad_output)
        grad_output = self.relu1.backward(grad_output)
        grad_output = self.conv1.backward(grad_output, learning_rate)

# 训练网络
def train(net, X_train, y_train_onehot, epochs=10, learning_rate=0.01, batch_size=64):
    losses = []
    for epoch in range(epochs):
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train_onehot[i:i+batch_size]

            # 前向传播
            output = net.forward(X_batch)
            loss = loss_fn.forward(output, y_batch)
            losses.append(loss)

            # 反向传播
            grad_output = loss_fn.backward()
            net.backward(grad_output, learning_rate)

        print(f'Epoch {epoch + 1}, Loss: {loss}')

    # 可视化Loss
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()

# 测试网络
def test(net, X_test, y_test):
    output = net.forward(X_test)
    pred = np.argmax(output, axis=1)
    accuracy = np.mean(pred == y_test)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

# 初始化网络和损失函数
net = Net()
loss_fn = SoftmaxCrossEntropyLoss()

# 训练网络
train(net, X_train, y_train_onehot)

# 测试网络
test(net, X_test, y_test)