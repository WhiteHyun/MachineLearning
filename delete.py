import random
import math
random.seed(1)


class tensor(list):
    """
    tensor
    """

    @classmethod
    def randn(cls, row_size, col_size=1):
        """가우시안 표준 정규 분포에서 난수 생성
        """
        if col_size != 1:
            return tensor(tensor(random.gauss(0.0, 1.0) for _ in range(col_size))for _ in range(row_size))
        else:
            return tensor(random.gauss(0.0, 1.0) for _ in range(row_size))

    def __dot(self, a, b):
        """행렬곱을 할 때 사용됩니다.


        Parameters
        ----------
        a : list
            계산될 행렬
        b : list
            계산될 행렬

        Examples
        --------
        1차원 크기의 행렬이라도 2차원 리스트로 인자를 주어야 합니다.

        >>> print(dot([[1, 2, 3], [4, 5, 6]], [[1, 2], [3, 4], [5, 6]]))
        [[22, 28], [49, 64]]

        Returns
        -------
        list
            행렬곱된 행렬

        """

        result = tensor()
        col = list(zip(*b))
        idx = 0
        # 행렬의 크기가 1xn일 때
        if len(a.shape()) == 1:
            for j in col:
                result.append(sum(x*y for x, y in zip(a, j)))
        else:
            for i in a:
                result.append(tensor())
                for j in col:
                    result[idx].append(sum(x*y for x, y in zip(i, j)))
                idx += 1
        return result

    def __add(self, a, b):
        """행렬합을 할 때 사용됩니다.


        Parameters
        ----------
        a : list
            계산될 행렬
        b : list
            계산될 행렬

        Examples
        --------
        1차원 크기의 행렬이라도 2차원 리스트로 인자를 주어야 합니다.

        >>> print(add([[22, 28], [49, 64]] + [[1, 2], [3, 4]]))
        [[23, 30], [52, 68]]

        Returns
        -------
        list
            행렬합된 행렬

        """
        result = tensor()
        idx = 0
        # 같은 크기의 행렬합인 경우
        a_shape = a.shape()
        b_shape = b.shape()
        if a_shape == b_shape:
            # 1xn 행렬일 때
            if len(a_shape) == 1:
                for x, y in zip(a, b):
                    result.append(x+y)
            # nxm 행렬일 때
            else:
                for x, y in zip(a, b):
                    result.append(tensor())
                    for i, j in zip(x, y):
                        result[idx].append(i + j)
                    idx += 1
        # 행렬과 bias 합인 경우
        elif a_shape[1] == b_shape:
            for row in a:
                result.append(tensor())
                result[idx] = tensor(map(lambda x, y: x + y, row, b))
                idx += 1
        return result

    def __rtruediv__(self, x):
        assert type(x) is not tensor
        result = tensor()
        idx = 0
        # 1xn 행렬일 때
        if len(self.shape()) == 1:
            for value in self:
                result.append(x/value)
        # nxm 행렬일 때
        else:
            for row in self:
                result.append(tensor())
                for value in row:
                    result[idx].append(x/value)
                idx += 1

        return result

    def __mul__(self, x):
        if type(x) is int:
            return super().__mul__(x)
        # 행렬곱
        elif self.shape() != x.shape():
            return self.__dot(self, x)
        # 같은 크기인 경우 각각 요소별 곱
        else:
            result = tensor()
            for val1, val2 in zip(self, x):
                result.append(val1*val2)
            return result

    def __rmul__(self, x):
        if type(x) is int:
            return super().__rmul__(x)
        elif type(x) is float:
            result = tensor()
            idx = 0
            for i in self:
                if type(i) is not tensor:
                    result.append(x*i)
                else:
                    result.append(tensor())
                    for j in i:
                        result[idx].append(x*j)
                    idx += 1
            return result

    def __add__(self, x):
        if type(x) is not int:
            return self.__add(self, x)
        # sum 적용 예외처리
        elif x == 0:
            return tensor(self.copy())
        # x가 정수일 때
        result = tensor()
        idx = 0
        if len(self.shape()) == 1:
            for value in self:
                result.append(value + x)
        else:
            for row in self:
                result.append(tensor())
                for value in row:
                    result[idx].append(value + x)
                idx += 1
        return result

    def __radd__(self, x):
        if type(x) is not int:
            return self.__add(self, x)
        return self.__add__(x)

    def __sub__(self, x):
        """행렬간 뺄셈
        """
        result = tensor()
        if len(self.shape()) == 1:
            for val1, val2 in zip(self, x):
                result.append(float.__sub__(val1, val2))
        else:
            for mtrx1, mtrx2 in zip(self, x):
                result.append(tensor(map(float.__sub__, mtrx1, mtrx2)))
        return result

    def __rsub__(self, x):
        """행렬간 뺄셈
        """
        result = tensor()
        if type(x) is tensor:
            self.__sub__(x)
        else:
            result = tensor()
            idx = 0
            # 1xn 행렬일 때
            if len(self.shape()) == 1:
                for value in self:
                    result.append(x-value)
            else:
                for mtrx in self:
                    result.append(tensor())
                    for i in mtrx:
                        result[idx].append(x-i)
                    idx += 1
            return result

    def __neg__(self):
        result = tensor()
        if len(self.shape()) == 1:
            for x in self:
                result.append(-x)
        else:
            idx = 0
            for x in self:
                result.append(tensor())
                for value in x:
                    result[idx].append(-value)
                idx += 1
        return result

    def __str__(self) -> str:
        if len(self.shape()) == 1:
            return super().__str__()
        result = "["
        for i in range(len(self)):
            if i == len(self)-1:
                result += f"{self[i]}]"
            else:
                result += f"{self[i]},\n "
        return result

    def __pow__(self, x):
        """행렬 거듭제곱
        """
        assert type(x) is int
        result = tensor()
        if len(self.shape()) == 1:
            for i in self:
                tensor.append(i ** x)
        else:
            idx = 0
            for i in self:
                result.append(tensor())
                for j in i:
                    result[idx].append(j ** x)
                idx += 1
        return result

    def T(self):
        """전치행렬
        """
        if len(self.shape()) == 1:
            return self.copy()
        else:
            return tensor(tensor(e) for e in zip(*self))

    def shape(self):
        """행렬의 크기
        """
        result = list()
        temp = self
        while type(temp) is tensor:
            if len(temp) == 1:
                break
            result.append(len(temp))
            temp = temp[0]
        return tuple(result)


class MultiLayerPerceptron:
    """2-layer 퍼셉트론,
    비용함수: MSE(Mean Squared Error)
    활성함수: sigmoid
    """

    def __init__(self, ni, nh, no) -> None:
        """

        Args:
            ni (int): input node
            nh (int): hidden node
            no (int): output node
        """
        self.weight = {}
        self.weight['w1'] = tensor.randn(ni, nh)
        self.weight['b1'] = tensor.randn(nh)
        self.weight['w2'] = tensor.randn(nh, no)
        self.weight['b2'] = tensor.randn(no)

    def predict(self, input_data):
        """
        Args:
            input_data: train dataset
        """
        if type(input_data) is list:
            input_data = tensor(input_data)

        w1, w2 = self.weight['w1'], self.weight['w2']
        b1, b2 = self.weight['b1'], self.weight['b2']
        a1 = input_data*w1 + b1  # input층->은닉층으로의 계산
        z1 = self.sigmoid(a1)   # 활성함수
        a2 = z1*w2 + b2         # 은닉층->출력층으로부터의 계산
        y = self.sigmoid(a2)    # 활성함수 적용

        return y

    def loss(self, input_data, label):
        predicted = self.predict(input_data)

        return self.__mse(predicted, label)

    def __mse(self, predicted, label):
        return 0.5 * sum((predicted-label)**2)

    def sigmoid(self, x):
        if len(x.shape()) == 1:
            e = tensor(map(lambda args: math.exp(args), -x))
        else:
            e = tensor(map(lambda args: tensor(math.exp(v) for v in args), -x))
        return 1/(1+e)

    def sigmoid_grad(self, output):
        """sigmoid grad라 써놓고 다르게 사용한다! ㅎㅎ;;

        sigmoid'(A^[l]) = sigmoid(A^[l])*(1-sigmoid(A^[l]))
                        = Z^[l]*(1-Z^[l])
        이기 때문에 output에 올바른 값을 넣도록 하자!

        Example
        -------

        a2에 대한 sigmoid 미분값을 구하기 위해서는 output에 y값을 주어야 한다.

        """
        return output*(1.0 - output)

    def gradient(self, input_data, label):
        """
        """
        if type(input_data) is list:
            input_data = tensor(input_data)

        w1, w2 = self.weight['w1'], self.weight['w2']
        b1, b2 = self.weight['b1'], self.weight['b2']
        grads = {}

        # forward
        a1 = input_data*w1 + b1
        z1 = self.sigmoid(a1)  # output
        a2 = z1*w2 + b2
        y = self.sigmoid(a2)  # output

        # backward
        dy = y - label  # MSE로 계산한 오차

        # hidden-output 층 간 기울기 계산
        grad_w2_weights = tensor()
        idx = 0
        for z in z1:
            grad_w2_weights.append(tensor())
            for sigmoid_grad, y_label in zip(self.sigmoid_grad(y), dy):
                grad_w2_weights[idx].append(y_label*sigmoid_grad*z)
            idx += 1
        grads_b2_weights = tensor()
        for j, dl in zip(self.sigmoid_grad(y), dy):
            grads_b2_weights.append(dl*j)
        # hidden-output 층 간 기울기 계산 끝

        grads['w2'] = grad_w2_weights
        grads['b2'] = grads_b2_weights

        # grad_w1_weights = w2 * grad_w2_weights
        # for test in grad_w1_weights:

        grads['w1'] = w2 * grad_w2_weights
        grads['b1'] = w2 * grad_w2_weights
        # input-hidden 층 간 기울기 계산

        # input-hidden 층 간 기울기 계산 끝

        # da1 = dy*w2.T()
        # dz1 = self.sigmoid_grad(a1)*da1
        # grads['w1'] = input_data.T()*dz1
        # grads['b1'] = sum(dz1)
        # print(grads['b1'], grads['b2'])
        return grads


if __name__ == "__main__":

    dataset = [[3.5064385449265267, 2.34547092892632525, 0], [4.384621956392097, 3.4530853889904205, 0], [4.841442919897487, 4.02507852317520154, 0], [3.5985868973088437, 4.1621314217538705, 0], [2.887219775424049, 3.31523082529190005, 0], [
        9.79822645535526, 1.1052409596099566, 1], [7.8261241795117422, 0.6711054766067182, 1], [2.5026163932400305, 5.800780055043912, 1], [5.032436157202415, 8.650625621472184, 1], [4.095084253434162, 7.69104329159447, 1]]
    x_train = tensor()
    y_train = tensor()
    for i in dataset:
        x_train.append(tensor(i[:-1]))
        y_train.append(tensor([1, 0])if i[-1] == 0 else tensor([0, 1]))
    perceptron = MultiLayerPerceptron(2, 2, 2)

    train_loss_list = []
    for i in range(300):
        if i % 300 == 0:
            print(i, end="..")
        rand = random.randint(0, len(x_train))
        grad = perceptron.gradient(x_train[rand], y_train[rand])

        for key in ('w1', 'b1', 'w2', 'b2'):
            perceptron.weight[key] -= 0.1 * grad[key]   # lr = 0.1

        loss = perceptron.loss(x_train, y_train)
        train_loss_list.append(loss)

    print(perceptron.predict(x_train))
    import numpy as np
    import matplotlib.pyplot as plt
    x = np.arange(len(train_loss_list))
    plt.plot(x, train_loss_list)
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.ylim(0, 9.0)
    plt.show()
