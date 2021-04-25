import random
import math

random.seed(1)


def randn(size):
    """난수 생성
    """
    return [random.random() for _ in range(size+1)]


class MultiLayerPerceptron:
    def __init__(self, ni, nh, no, dataset, epochs=5000) -> None:
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
        self.epochs = epochs

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
        # return max(0, x)    # ReLU
        return 1.0/(1.0+math.exp(-x))  # sigmoid

    def activation_func_grad(self, x):
        """활성함수 미분계수 값입니다.
        """
        # return 1 if x > 0 else 0  # ReLU
        return x*(1.0-x)  # sigmoid

    def feed_foward(self, data):
        """순전파 수행
        """
        inputs = data
        # 각각의 layer를 지남
        for layer in self.model:
            outputs = []
            # layer들 중 노드들을 통해 가중치 계산
            for node in layer:
                zsum_or_osum = self.weight_sum(node['weights'], inputs)
                # 각 가중치 합을 계산하여 활성함수를 적용한 output을 해당 layer-node에 output을 key값으로 하여 적용
                node['output'] = self.activation_func(zsum_or_osum)
                outputs.append(node['output'])
            inputs = outputs
        return inputs

    def backward(self, label):
        """역전파 알고리즘입니다.
        스토캐스틱 경사하강법(SGD)을 채택하였습니다.
        """
        # 출력 레이어(층) -> 입력 레이어(층) 순서로 역전파 진행
        for i in reversed(range(len(self.model))):
            layer = self.model[i]
            errors = []  # 계산할 에러

            if i == len(self.model)-1:  # 출력층인 경우
                for j in range(len(layer)):
                    node = layer[j]
                    errors.append(label[j] - node['output'])
            else:
                for j in range(len(layer)):
                    error = 0.0
                    for node in self.model[i+1]:  # 다음 레이어에 대해
                        error += (node['weights'][j]*node['delta'])
                    errors.append(error)
            for j in range(len(layer)):
                node = layer[j]
                node['delta'] = errors[j] * \
                    self.activation_func_grad(node['output'])

    def update(self, train_set, lr):
        """weight update 함수
        """
        for i in range(len(self.model)):
            inputs = train_set[:-1] if i == 0 else [node['output']
                                                    for node in self.model[i-1]]
            for node in self.model[i]:
                for j in range(len(inputs)):
                    node['weights'][j] += lr * node['delta'] * \
                        inputs[j]  # 역전파 할 때 곱해야할 노드값까지 계산
                node['weights'][-1] += lr * \
                    node['delta']  # bias의 노드는 항상 1임

    def train(self, lr=0.5, verbose=False):
        """주어진 dataset을 가지고 학습합니다.

        Parameters
        ----------

        lr : float
        verbose : bool
            epoch과 에러율을 보여줍니다.
        """
        for epoch in range(self.epochs):
            error = 0.0
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
                print(f"epoch: {epoch}, error: {error/len(self.dataset):.3f}")


if __name__ == "__main__":
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

    len_input_nodes = len(dataset[0])-1
    len_hidden_nodes = 2
    len_output_nodes = len(set(map(lambda x: x[-1], dataset)))

    epochs = int(input("epochs: "))
    network = MultiLayerPerceptron(
        len_input_nodes, len_hidden_nodes, len_output_nodes, dataset, epochs)
    network.train(verbose=True)
