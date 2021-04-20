# 201601639 홍승현 인공지능 과제

## 내용

- 언어: `Python`
- 라이브러리: math, random, numpy(relu로 변환할 때만 사용)
- 시드: `1`로 사용
- 가중치 초기화: 0~1 사이의 값을 `random`을 이용
- 사용: `2-layer MLP`
- 입력층: 2개의 노드
- 은닉층: 2개의 노드
- 출력층: 2개의 노드
- Loss function: `MSE`
- Training dataset = Test
- 활성함수: `sigmoid`
  - $S(x)$ = $ 1 \over {1+e^{-x}}$ = $ e^x \over {e^x+1}$

## 파일 제출

1. Training을 위한 파일

   - 네트워크 구성 및 weight initialization 함수
   - weight summation 및 activation 함수
   - 순방향 전파 함수
   - 각 layer에서의 error 계산 및 저장 함수
   - weight update 함수
   - Epoch (시행횟수)를 입력 중 하나로 받는 전체 training 함수

2. Test를 위한 파일
   - Training의 결과를 사용 (학습된 weight값 사용)
   - train dataset을 이용한 예측 값 구하는 함수
   - 예측 값과 정답을 비교하여 출력하는 함수

## 평가 항목

1. 2개의 파일 및 파일 내 해당 함수 구현하였는가?
2. 에러 없이 동작하는가?
3. 정상적으로 동작하는가?

- 각 변수를 입력 dataset과 output class가 변함에 따라 적용될 수 있도록 구현하였는가?
- ReLU activation을 바꾸고 동작하는가?
- ReLU로 바꿨을 경우 성능이 차이난다면 그 이유는 무엇인가?

## dataset

- 2개의 class \[0, 1\] 을 label로 갖는 1x2크기의 data 10개

---

## 코드 리뷰

### 클래스

```python
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
   ...
```

- 2-layer 퍼셉트론 모델을 생성하는 클래스
- `ni`: input node
- `nh`: hidden node
- `no`: output node

#### layer

- hidden_layer: 은닉층
- output_layer: 출력층
- 각 layer은 [node](#node)를 `리스트` 형태로 가지고 있다.
- 형태
  <img width="811" alt="image" src="https://user-images.githubusercontent.com/57972338/115352368-3b101780-a1f2-11eb-8309-0dc06de1b611.png">

#### node

- 각 은닉층, 출력층의 요소
- node는 `dict`임
- 각 자신에게 들어오는 가중치를 리스트로 가지고 있음

  - 형태
    <img width="636" alt="image" src="https://user-images.githubusercontent.com/57972338/115351558-575f8480-a1f1-11eb-80e3-e6ede5d09714.png">

    ```python
    node["weights"]
    >>> [1.125125, -0.0018124, 0.0000523123]
    ```

- 각 가중치합을 활성함수로 적용시킨 노드는 `output`을 키값으로 하여 값을 저장함

  - 형태
    <img width="655" alt="image" src="https://user-images.githubusercontent.com/57972338/115351763-8d9d0400-a1f1-11eb-939f-df663dee2e25.png">

    ```python
    node["output"]
    >>> 0.008123125
    ```

- `backward`를 통해 델타값을 구한 것을 각 노드마다 갖고 있어야 하므로 `delta`를 키값으로 하여 저장
  - 형태
    <img width="803" alt="image" src="https://user-images.githubusercontent.com/57972338/115352118-f1273180-a1f1-11eb-9386-a0dc3bb7c4c5.png">
    ```python
    node["delta"]
    >>> 0.0013421
    ```

### 함수

- 난수 생성 함수

  ```python
  def randn(size):
     """난수 생성
     """
     return [random.random() for _ in range(size+1)]
  ```

  > _기존에는 가우시안 정규분포를 이용한 랜덤값인 `random.gauss()`을 사용하려 했으나 학습률이 좋지 않아 random()으로 바꿈_

- input값과 가중치를 계산하여 합하는 함수

  ```python
  def weight_sum(self, weights, inputs):
  """각각의 가중치계산 후 결과값을 리턴합니다.
  """
  sum = weights[-1]  # 바이어스 값은 노드가 1이므로 미리 할당합니다.
  for i in range(len(weights)-1):
     sum += weights[i]*inputs[i]
  return sum
  ```

- 활성함수

  ```python
  def activation_func(self, x):
       """활성함수입니다.
       """
       return max(0, x)    # ReLU
       # return 1.0/(1.0+math.exp(-x))  # sigmoid
  ```

- 활성함수 도함수

  ```python
  def activation_func_grad(self, x):
       """활성함수 도함수 값입니다.
       """
       return 1 if x > 0 else 0 # ReLU 도함수
       # return x*(1.0-x)  # sigmoid 도함수
  ```

- 순전파 함수

  ```python
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
  ```

  - `dataset`을 input으로 하여 각 layer들의 가중치를 곱하고 활성함수를 적용하여 출력값을 얻는다.
  - 각 노드들은 순전파할 때 가중치합을 활성함수로 적용시킨 output값을 저장해놓는다.
  - 최종적으로 해당 모델에 대한 최종 output을 리턴한다.
    - 이 때 각 노드들은 output에 대한 값들을 다 저장해놓고있다.

- 역전파 함수

  ```python
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
  ```

  - 각 노드들은 자신의 `output`을 가지고 있기 때문에 인자를 받지 않고 label만 인자로 받음
  - 출력층인 경우 해당 코드는

    ```python
    for j in range(len(layer)):
      node = layer[j]
      errors.append(label[j] - node['output'])
    ```

    <img width="306" alt="image" src="https://user-images.githubusercontent.com/57972338/115361398-83800300-a1fb-11eb-9b53-37720cd8f147.png">

    - 각 노드의 `-(y-o)`에 해당한다.

  - 그 외 레이어의 경우

    ```python
    for j in range(len(layer)):
      error = 0.0
      for node in self.model[i+1]:  # 다음 레이어에 대해
        error += (node['weights'][j]*node['delta'])
      errors.append(error)
    ```

    <img width="446" alt="image" src="https://user-images.githubusercontent.com/57972338/115364770-b5469900-a1fe-11eb-8fcb-a8dec23f8d4d.png">

    - `sigma` 에 해당한다.

  - 각 레이어에서 구한 오차 `error`는
    ```python
    for j in range(len(layer)):
      node = layer[j]
      node['delta'] = errors[j] * self.activation_func_grad(node['output'])
    ```
    - 위와 같이 노드에 `delta`를 key값으로 하여 저장한다.
      > 각 레이어마다 자신의 노드값을 곱하지 않고 저장한 이유는 그 이외에 다른 층에서 델타값을 사용하기 때문에 사용 용이성을 높이기 위해 제외하였다. 자신의 노드값을 곱하는 것은 `update`함수에서 진행된다.

- 가중치 갱신 함수

  ```python
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
  ```

  - 위 역전파 코드에서 곱해야할 노드를 생략하였는데, 위 코드에서 가중치를 갱신할 때 곱하여 갱신합니다.

- 학습 함수

  ```python
  def train(self, epochs, lr=0.5, verbose=False):
       """주어진 dataset을 가지고 학습합니다.

       Parameters
       ----------

       epochs : int
       lr : float
       verbose : bool
           epoch과 에러율을 보여줍니다.
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
  ```

  - 각 `dataset`을 가지고 순전파를통해 예측값을 저장하고 label과의 차이를 가지고 역전파하여 가중치값을 **갱신**한다.

## 결론

<!-- TODO: 작성해야함 -->
