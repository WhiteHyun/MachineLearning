from training import MultiLayerPerceptron


def predict(model, dataset=None):
    """예측 값을 보여줍니다.
    """
    if dataset is not None:
        model.dataset = dataset
    outputs = []
    labels = []
    for train_set in model.dataset:
        output = model.feed_foward(train_set)
        outputs.append(
            list(map(lambda x: 1 if x > 0.95 else 0 if x < 0.05 else x, output)))
        # one hot vector로 구성
        label = [0 for _ in range(
            len(set(map(lambda x: x[-1], model.dataset))))]
        label[train_set[-1]] = 1
        labels.append(label)
    return (outputs, labels)


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
    model = MultiLayerPerceptron(2, 2, 2, dataset)
    model.train(5000)
    outputs, labels = predict(model)
    print(f"predict: \n{outputs}")
    print(f"label: \n{labels}")
    print(
        f"accuracy: {sum([1 for x in zip(outputs, labels) if x[0] == x[1]])/len(model.dataset)}")
