




def apk(actual, predicted, k=10):
    """
    Computes the precision at k.
    This function computes the precision at k between the actual element and a list
    of predicted elements.
    Parameters
    ----------
    actual : int
             The actual element that has to be predicted
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    precision : double
                The precision at k over the input
    """
    if len(predicted) > k:
        predicted = predicted[:k]
    correct_predictions = 0
    for i in range(len(predicted)):
        if predicted[i] == 1:
            correct_predictions += 1

    return correct_predictions / k



def mapk(actual, predicted, k=10):
    """
    Computes the mean precision at k.
    This function computes the mean precision at k between a list of query elements and a list
    of database retrieved elements.
    Parameters
    ----------
    actual : list
             The list of query elements that have to be predicted
    predicted : list
                A list of lists of predicted elements (order does matter) for each query element
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    mean_precision : double
                    The mean precision at k over the input
    """
    total_precision = 0
    for i in range(len(actual)):
        total_precision += apk(actual[i], predicted[i], k)
    
    return total_precision / len(actual)