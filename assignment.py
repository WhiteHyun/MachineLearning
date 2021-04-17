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
        if a.shape() == b.shape():
            for x, y in zip(a, b):
                result.append(tensor())
                for i, j in zip(x, y):
                    result[idx].append(i + j)
                idx += 1
        # 행렬 내 스칼라 합인 경우
        elif a[0].shape() == b.shape():
            for row in a:
                result.append(tensor())
                result[idx] = tensor(map(lambda x, y: x + y, row, b))
                idx += 1
        return result

    def __rtruediv__(self, x):
        if type(x) is tensor:
            pass
        result = tensor()
        idx = 0
        for i in self:
            result.append(tensor())
            for value in i:
                result[idx].append(x/value)
            idx += 1

        return result

    def __mul__(self, x):
        if type(x) is int:
            return super().__mul__(x)
        return self.__dot(self, x)

    def __add__(self, x):
        if type(x) is not int:
            return self.__add(self, x)
        # x가 정수일 때
        result = tensor()
        idx = 0
        for row in self:
            result.append(tensor())
            for value in row:
                result[idx].append(value + x)
            idx += 1
        return result

    def __radd__(self, x):
        if type(x) is not int:
            return self.__add(self, x)

        # x가 정수일 때
        result = tensor()
        idx = 0
        for row in self:
            result.append(tensor())
            for value in row:
                result[idx].append(value + x)
            idx += 1
        return result

    def __sub__(self, x):
        """행렬간 뺄셈
        """
        result = tensor()
        for mtrx1, mtrx2 in zip(self, x):
            result.append(tensor(map(float.__sub__, mtrx1, mtrx2)))
        return result

    def T(self):
        """전치행렬
        """
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
        return 0.5 * math.sum((predicted-label)**2)

    def sigmoid(self, x):
        return 1/(1+tensor(map(lambda args: tensor(math.exp(v) for v in args), x)))

    def sigmoid_grad(self, x):
        return (1.0 - self.sigmoid(x)) * self.sigmoid(x)

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
        z1 = self.sigmoid(a1)
        a2 = z1*w2 + b2
        y = self.sigmoid(a2)

        # backward
        # print(f"y = {y}")
        # print(f"label = {label}")
        dy = y - label
        grads['w2'] = z1.T()*dy
        grads['b2'] = math.sum(dy)

        da1 = dy*w2.T()
        dz1 = self.sigmoid_grad(a1)*da1
        grads['w1'] = input_data.T()*dz1
        grads['b1'] = math.sum(dz1)

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
    perceptron.gradient(x_train, y_train)
