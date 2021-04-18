import random
import math


def randn(size):
    """난수 생성
    """
    return [random.random() for _ in range(size+1)]


class MultiLayerPerceptron:
    def __init__(self, ni, nh, no, dataset) -> None:
        """퍼셉트론 네트워크 초기화
        """
        self.model = []
        hidden_layer = [
            {'weights': randn(ni)} for _ in range(nh)
        ]
        output_layer = [
            {'weights': randn(nh)}for _ in range(no)
        ]
        self.model.append(hidden_layer)
        self.model.append(output_layer)
        self.dataset = dataset

    def weight_sum(self, weights, inputs):
        """각각의 가중치계산 후 결과값을 리턴합니다.
        """
        sum = weights[-1]  # 바이어스 값은 노드가 1이므로 미리 할당합니다.
        for i in range(len(weights)-1):
            sum += weights[i]*inputs[i]
        return sum

    def activation_func(self, x):
        """활성함수입니다.
        """
        return 1.0/(1.0+math.exp(-x))  # sigmoid

    def activation_func_grad(self, x):
        """활성함수 미분값입니다.
        """
        return x*(1.0-x)  # sigmoid 미분값

    def feed_foward(self, data):
        """순전파 수행
        """
        inputs = data
        # 각각의 layer들을 지남
        for layer in self.model:
            outputs = []
            # layer들 중 노드들을 통해 가중치 계산
            for node in layer:
                zsum_or_osum = self.weight_sum(node['weights'], inputs)
                node['output'] = self.activation_func(zsum_or_osum)
                outputs.append(node['output'])
            inputs = outputs
        return inputs

    def backward(self, label):
        """역전파 알고리즘입니다.
        스토캐스틱 경사하강법(SGD)을 채택하였습니다.
        """
        # 출력층 -> 입력층 순
        for i in reversed(range(len(self.model))):
            layer = self.model[i]
            errors = []
            if i != len(self.model)-1:  # 출력층이 아닌 경우
                for j in range(len(layer)):
                    error = 0.0
                    for node in self.model[i+1]:  # 다음 레이어에 대해
                        error += (node['weights'][j]*node['delta'])
                    errors.append(error)
            else:
                for j in range(len(layer)):
                    node = layer[j]
                    errors.append(label[j] - node['output'])
            for j in range(len(layer)):
                node = layer[j]
                node['delta'] = errors[j] * \
                    self.activation_func_grad(node['output'])

    def update(self, train_set, lr):
        """weight update 함수
        """
        for i in range(len(self.model)):
            x_train = train_set[:-1]
            if i != 0:
                x_train = [node['output'] for node in self.model[i-1]]
            for node in self.model[i]:
                for j in range(len(x_train)):
                    node['weights'][j] += lr * node['delta']*x_train[j]
                node['weights'][-1] += lr * \
                    node['delta']  # bias의 노드는 항상 1임

    def train(self, epochs, lr=0.5, verbose=False):
        """주어진 dataset을 가지고 학습합니다.
        """
        for epoch in range(epochs):
            error = 0
            for train_set in self.dataset:
                # forward
                outputs = self.feed_foward(train_set)

                # one hot vector로 구성
                label = [0 for _ in range(
                    len(set([row[-1] for row in self.dataset])))]
                label[train_set[-1]] = 1

                # 표기할 오차
                error += sum((label[i]-outputs[i]) **
                             2 for i in range(len(label)))

                # backward
                self.backward(label)

                # update
                self.update(train_set, lr)

            if verbose and epoch % 100 == 0:
                print(f"epoch: {epoch}, error: {error:.3f}")


if __name__ == "__main__":
    random.seed(1)
    dataset = [[3.5064385449265267, 2.34547092892632525, 0],
               [4.384621956392097, 3.4530853889904205, 0],
               [4.841442919897487, 4.02507852317520154, 0],
               [3.5985868973088437, 4.1621314217538705, 0],
               [2.887219775424049, 3.31523082529190005, 0],
               [9.79822645535526, 1.1052409596099566, 1],
               [7.8261241795117422, 0.6711054766067182, 1],
               [2.5026163932400305, 5.800780055043912, 1],
               [5.032436157202415, 8.650625621472184, 1],
               [4.095084253434162, 7.69104329159447, 1]]

    network = MultiLayerPerceptron(2, 2, 2, dataset)
    network.train(2000)
