from training import MultiLayerPerceptron


def predict(model, dataset=None):
    """model의 dataset을 가지고 계산하여 예측값과 정답값을 리턴합니다.


    Example
    -------

    >>> predict(model)
    ([[1, 0], [1, 0], ..., [0, 1]], [[1, 0], [1, 0], ..., [0, 1]])
    """
    if dataset is not None:
        model.dataset = dataset
    outputs = []
    labels = []
    for train_set in model.dataset:
        output = model.feed_foward(train_set)
        outputs.append(output)
        # one hot vector로 구성
        label = [0 for _ in range(
            len(set(map(lambda x: x[-1], model.dataset))))]
        label[train_set[-1]] = 1
        labels.append(label)
    return (outputs, labels)


def print_result(outputs, labels):
    """예측값과 레이블을 가지고 나온 결과를 정리하여 출력합니다

    Parameters
    ----------
    outputs : list
        - 예측값(hypothesis)

    labels : list
        - 정답(label)
    """
    hypothesis = list(map(list, list(map(lambda x: 1 if x > 0.95 else 0 if x <
                                         0.05 else x, output) for output in outputs)))
    print(f"""======================================predict======================================
{hypothesis}
===================================================================================

    """)
    print(f"""=======================================label=======================================
{labels}
===================================================================================

    """)
    print(
        f"accuracy: {sum([1 for x in zip(hypothesis, labels) if x[0] == x[1]])/len(model.dataset)*100}%")


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

    model = MultiLayerPerceptron(
        len_input_nodes, len_hidden_nodes, len_output_nodes, dataset, epochs)

    model.train(verbose=False)  # 출력: False

    outputs, labels = predict(model)  # dataset이 같기 때문에 추가적인 파라미터 입력 X

    print_result(outputs, labels)
