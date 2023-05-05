from train_wm import wang_mendel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from T1_set import T1_LeftShoulder, T1_RightShoulder, T1_Triangular
from T1_output import T1_Triangular_output, T1_RightShoulder_output, T1_LeftShoulder_output


def mackey_glass(tau, beta=0.2, gamma=0.1, n=10, max_time=1000):
    """Define the Mackey-Glass equation with given parameters and return the time series"""
    # First 17th records in the time-series
    y = [0.9697, 0.9699, 0.9794, 1.0003, 1.0319, 1.0703, 1.1076, 1.1352, 1.1485,
         1.1482, 1.1383, 1.1234, 1.1072, 1.0928, 1.0820, 1.0756, 1.0739, 1.0759]

    for n in range(17, max_time-1):
        y.append(y[n] - gamma*y[n] + beta*y[n-tau]/(1+y[n-tau]**10))
    return np.array(y)


def generate_outputs_object(pairs_of_strength_antecedent, train_obj):
    outputs = {}
    for index_of_ant, fs in pairs_of_strength_antecedent:
        if (isinstance(train_obj.antecedents[index_of_ant], T1_Triangular)):
            outputs[index_of_ant] = T1_Triangular_output(
                fs, train_obj.antecedents[index_of_ant].interval)
        if (isinstance(train_obj.antecedents[index_of_ant], T1_RightShoulder)):
            outputs[index_of_ant] = T1_RightShoulder_output(
                fs, train_obj.antecedents[index_of_ant].interval)
        if (isinstance(train_obj.antecedents[index_of_ant], T1_LeftShoulder)):
            outputs[index_of_ant] = T1_LeftShoulder_output(
                fs, train_obj.antecedents[index_of_ant].interval)

    if (len(outputs) == 0):
        return 0

    degree = []
    try:
        disc_of_all = np.linspace(list(train_obj.antecedents.values())[0].interval[0],
                                  train_obj.antecedents[list(
                                      train_obj.antecedents.keys())[-1]].interval[1],
                                  int((500 / 2.0) * (len(train_obj.antecedents) + 1)))
    except:
        print("error in generate outputs object")

    for x in disc_of_all:
        max_degree = 0.0
        for i in outputs:
            if max_degree < outputs[i].get_degree(x):
                max_degree = outputs[i].get_degree(x)
        degree.append(max_degree)

    numerator = np.dot(disc_of_all, degree)
    denominator = sum(degree)
    if not denominator == 0:
        return (numerator / float(denominator))
    else:
        return (0.0)


def get_MSE(real_values_list, predicted_value_list):
    return (np.square(np.subtract(real_values_list, predicted_value_list)).mean())


def individual_rule_output(firing_strengths, rule):
    firing_level_of_pairs = 1
    for i in range(0, len(inputs)):
        temp_firing = inputs[i][int(rule[i]) - 1]

        if (temp_firing == 0):
            firing_level_of_pairs = "nan"
            break

        # minimum is implemented
        if (temp_firing < firing_level_of_pairs):
            firing_level_of_pairs = temp_firing

    return firing_level_of_pairs


def union_strength_of_same_antecedents(list_of_antecedent_strength, output_antecedent_list):
    grouped_output_antecedent_strength = pd.DataFrame(
        index=range(0, len(output_antecedent_list)), columns=range(1, 3))

    grouped_output_antecedent_strength[1] = list_of_antecedent_strength
    grouped_output_antecedent_strength[2] = output_antecedent_list

    l1 = grouped_output_antecedent_strength.groupby([2]).max()

    l1 = pd.DataFrame.dropna(l1)

    return (zip(l1.index, l1[1]))


def apply_rules_to_inputs(all_firing_strengths, train_obj, inputs, outputs):
    output_results = []

    for firing_strengths in all_firing_strengths:

        rule_output_strength = np.empty([len(train_obj.reduced_rules), 1])
        rule_output_strength.fill(np.NaN)

        for rule_index, rule in enumerate(train_obj.reduced_rules):
            rule_output_strength[rule_index] = individual_rule_output(
                firing_strengths, rule)

        firing_level_for_each_output = union_strength_of_same_antecedents(
            rule_output_strength, train_obj.reduced_rules[:, train_obj.p])

        # calculate the centroid of the united outputs
        centroid = generate_outputs_object(
            firing_level_for_each_output, train_obj)
        output_results.append(centroid)

    MSE = get_MSE(X_test[train_obj.p:], output_results[:-train_obj.p])
    return (MSE, output_results)


def generate_test(train_obj, inputs, outputs):

    # Generate_firing_strengths
    firing_strengths = np.empty([len(inputs), train_obj.antecedent_number, 2])
    firing_strengths.fill(np.NaN)

    for index, (distance_input, angle_input) in enumerate(inputs):
        antecedent_firings = []

        for i in range(1, train_obj.antecedent_number+1):
            distance_fs = train_obj.distance_antecedents[i].get_degree(
                distance_input)

            angle_fs = train_obj.angle_antecedents[i].get_degree(angle_input)

            antecedent_firings.append((distance_fs, angle_fs))

        firing_strengths[index] = antecedent_firings

    # Apply_rules_to_inputs
    mse, output_results = apply_rules_to_inputs(
        firing_strengths, train_obj, inputs, outputs)

    print("MSE: ", mse)

    # Plot the results
    plt.figure(figsize=(10, 5))
    plt.rcParams.update({'font.size': 15})

    plt.gca().yaxis.grid(True)
    plt.plot(X_test[train_obj.p:], label='MG')
    plt.plot(output_results[:-train_obj.p], 'r-.', label='Pred')
    plt.title("Mackey-Glass Chaotic Time Series\nMSE:"+str(round(mse, 4)))
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.show()


def main():

    data_matrix = np.load("datos.npy")[100:1000]

    # First 300 records used for learning rules
    train_matrix = data_matrix[:300]
    test_matrix = data_matrix[-700:]  # Remaining 700 records used for testing

    # Generating rules from noise free set
    # 7 antecedents, 9 past points
    linear_train_obj = wang_mendel("linear", train_matrix, 7)
    angular_train_obj = wang_mendel("angular", train_matrix, 7)

    generate_test(linear_train_obj, test_matrix[:, 0:2], test_matrix[:, 2])
    generate_test(angular_train_obj, test_matrix[:, 0:2], test_matrix[:, 3])


if __name__ == "__main__":
    main()
