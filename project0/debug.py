def get_sum_metrics(predictions, metrics=[]):

    # check if metrics was not passed
    # if so, then reset this variable to prevent carry over
    metrics = metrics.copy()
#    if metrics is None:
#        metrics = []
    
    # use binding hack (i=i) to prevent late binding of i to 2
    for i in range(3):
        metrics.append(lambda x, i=i: x + i)

    sum_metrics = 0
    for metric in metrics:
        sum_metrics += metric(predictions)

    return sum_metrics


def main():
    print(get_sum_metrics(0))  # Should be (0 + 0) + (0 + 1) + (0 + 2) = 3
    print(get_sum_metrics(1))  # Should be (1 + 0) + (1 + 1) + (1 + 2) = 6
    print(get_sum_metrics(2))  # Should be (2 + 0) + (2 + 1) + (2 + 2) = 9
    print(get_sum_metrics(3, [lambda x: x]))  # Should be (3) + (3 + 0) + (3 + 1) + (3 + 2) = 15
    print(get_sum_metrics(0))  # Should be (0 + 0) + (0 + 1) + (0 + 2) = 3
    print(get_sum_metrics(1))  # Should be (1 + 0) + (1 + 1) + (1 + 2) = 6
    print(get_sum_metrics(2))  # Should be (2 + 0) + (2 + 1) + (2 + 2) = 9
    
    metrics = [lambda x: x]
    print(get_sum_metrics(1, metrics))
    print(get_sum_metrics(2, metrics))

if __name__ == "__main__":
    main()
