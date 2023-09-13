import json

import numpy as np
from scipy.stats import entropy, wasserstein_distance


def survey_responses(model, questions, options):
    responses = [model.respond(question, option) for question, option in zip(questions, options)]
    return responses

def measure_accuracy(model_responses, philosopher_responses):
    accuracy = 0
    for model_response, philosopher_response in zip(model_response, philosopher_responses):
        if np.argmax(np.array(model_response)) == np.argmax(np.array(philosopher_response)):
            accuracy += 1
    accuracy / len(model_responses)
    return accuracy

def measure_difference(model_responses, philosopher_responses):
    divergences = [entropy(m, p) for m, p in zip(model_responses, philosopher_responses)]
    distances = [wasserstein_distance(m, p) for m, p in zip(model_responses, philosopher_responses)]
    return divergences, distances

def measure_correlation(questions, options, responses, question_pairs, option_pairs, response_tables):
    divergences = []
    differences = []
    for i, question_pair in enumerate(question_pairs):
        for j, question in enumerate(questions):
            if question_pair[0] in question:
                break
        for k, question in enumerate(questions):
            if question_pair[1] in question:
                break
        table = [0] * 4
        option_a = options[j].index(option_pairs[i][0])
        table[0] = responses[j][option_a]
        table[1] = 1 - table[0]
        option_b = options[k].index(option_pairs[i][1])
        table[2] = responses[k][option_b]
        table[3] = 1 - table[2]
        divergence += [entropy(table, response_tables[i])]
        difference += [wasserstein_distance(table, response_tables[i])]
    return divergences, differences

def measure_all(model, questions, options, philosopher_responses, question_pairs, option_pairs, response_tables):
    # name = model.config["name"].split("/")[1]
    # model = model.model
    model_responses = survey_responses(model, questions, options)
    accuracy = measure_accuracy(model_responses, philosopher_responses)
    divergences, distances = measure_difference(model_responses, philosopher_responses)
    correlation_divergences, correlation_distances = measure_correlation(questions, options, model_responses, question_pairs, option_pairs, response_tables)
    # results = {
    #     "model": name,
    #     "accuracy": accuracy,
    #     "divergences": divergences,
    #     "distances": distances,
    #     "correlation_divergences": correlation_divergences,
    #     "correlation_distances": correlation_distances,
    # }
    # json.dump(results, open(f"../results/{name}.json", "w"))
    return accuracy, divergences, distances, correlation_divergences, correlation_distances