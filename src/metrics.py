import pickle

def fidelity_plus(pred, pred_new):
    """Metric calculation to measure explainer
     robustness to graph noise.
        Arguments:
            pred: probabilities of original predictions
            pred_new: probabilites of new predictions

        Returns: metric int
        """
    total = 0

    for i in range(len(pred)):
        total += pred[i] - pred_new[i]

    fidelity = total / len(pred)

    return fidelity


def completeness(explainer_class, model, data):
    """Metric to measure completeness (classifier accuracy)
        Arguments:
            explainer_class
            model
            data

        Returns: metric int
        """
    explainer = explainer_class()

    explainer.learn_prototypes(model, data)
    
    completeness = explainer.get_completeness(model, data)

    return completeness


def concepts(explainer_class, model, data, output_location):
    """Metric to return concepts
        Arguments:
            explainer_class
            model
            data

        Returns: concepts vector
        """
    explainer = explainer_class()

    concepts = explainer.get_concepts(model, data)

    #pkl concepts because need to loop through
    #aggressive/conservative/noise combinations
    #and calculate difference of concept vectors

    with open(f'results/concepts/{output_location}.pkl', 'wb') as f:
        pickle.dump(concepts, f)


def prototype_probs(explainer_class, model, data, output_location):
    """Metric to return prototype probabilities
        Arguments:
            explainer_class
            model
            data

        Returns: concepts vector
        """
    explainer = explainer_class()

    prototype_probs = explainer.get_prototype_probs(model, data)

    #pkl concepts because need to loop through
    #aggressive/conservative/noise combinations
    #and calculate difference of prototype_probs

    with open(f'results/prototype_probs/{output_location}.pkl', 'wb') as f:
        pickle.dump(prototype_probs, f)