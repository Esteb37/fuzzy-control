from train_wm import wang_mendel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from T1_set import T1_LeftShoulder, T1_RightShoulder, T1_Triangular
from T1_output import T1_Triangular_output, T1_RightShoulder_output, T1_LeftShoulder_output
import multiprocessing as mp
import copy as cp
from time import time


def generate_outputs_object(pairs_of_strength_antecedent, antecedents):
    outputs = {}
    for index_of_ant, fs in pairs_of_strength_antecedent:
        if (isinstance(antecedents[index_of_ant], T1_Triangular)):
            outputs[index_of_ant] = T1_Triangular_output(
                fs, antecedents[index_of_ant].interval)
        if (isinstance(antecedents[index_of_ant], T1_RightShoulder)):
            outputs[index_of_ant] = T1_RightShoulder_output(
                fs, antecedents[index_of_ant].interval)
        if (isinstance(antecedents[index_of_ant], T1_LeftShoulder)):
            outputs[index_of_ant] = T1_LeftShoulder_output(
                fs, antecedents[index_of_ant].interval)

    if (len(outputs) == 0):
        return 0

    degree = []
    try:
        disc_of_all = np.linspace(list(antecedents.values())[0].interval[0],
                                  antecedents[list(
                                      antecedents.keys())[-1]].interval[1],
                                  int((500 / 2.0) * (len(antecedents) + 1)))
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


def individual_rule_output(inputs, rule):
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


def apply_rules_to_inputs(params):

    all_firing_strengths, reduced_rules, output_antecedents = params

    output_results = []
    for firing_strengths in all_firing_strengths:

        rule_output_strength = np.empty([len(reduced_rules), 1])
        rule_output_strength.fill(np.NaN)

        for rule_index, rule in enumerate(reduced_rules):
            rule_output_strength[rule_index] = individual_rule_output(
                firing_strengths, rule[0:2])

        firing_level_for_each_output = union_strength_of_same_antecedents(
            rule_output_strength, reduced_rules[:, 2])

        centroid = generate_outputs_object(
            firing_level_for_each_output, output_antecedents)
        output_results.append(centroid)

    return output_results


def apply_rules_to_inputs_parallel(all_firing_strengths, train_obj, outputs):
    num_processes = 3
    # create a pool of worker processes
    pool = mp.Pool(num_processes)
    # calculate chunk size for each process
    chunksize = int(len(all_firing_strengths) / num_processes)

    # apply the function to each chunk of input firing strengths in parallel

    results = pool.map_async(apply_rules_to_inputs,
                             [(chunk, np.copy(train_obj.reduced_rules), cp.deepcopy(train_obj.output_antecedents))
                              for chunk in chunks(all_firing_strengths, chunksize)])

    # combine the results from each worker process
    output_results = []
    for r in results.get():
        output_results.extend(r)

    # calculate the average MSE across all input firing strengths
    MSE = get_MSE(outputs, output_results)

    # close the pool of worker processes
    pool.close()
    pool.join()

    return (MSE, output_results)


def chunks(lst, chunk_size):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def generate_test(train_obj, inputs, outputs):
    # Generate_firing_strengths
    matrix_width = max(train_obj.antecedent_numbers)
    firing_strengths = np.empty(
        [len(inputs), 2, matrix_width])
    firing_strengths.fill(np.NaN)

    for index, (distance_input, angle_input) in enumerate(inputs):
        distance_antecedent_firings = np.empty(matrix_width)
        angle_antecedent_firings = np.empty(matrix_width)

        for i in range(1, train_obj.antecedent_numbers[0]+1):
            distance_fs = train_obj.distance_antecedents[i].get_degree(
                distance_input)

            distance_antecedent_firings[i-1] = distance_fs

        for i in range(1, train_obj.antecedent_numbers[1]+1):
            angle_fs = train_obj.angle_antecedents[i].get_degree(angle_input)
            angle_antecedent_firings[i-1] = angle_fs

        firing_strengths[index] = [distance_antecedent_firings,
                                   angle_antecedent_firings]

    results = apply_rules_to_inputs_parallel(
        firing_strengths, train_obj, outputs)
    return results


def generate_rules(ant_numbers, distance_range, angle_range,
                   linear_speed_range, angular_speed_range):

    print("Generating rules for antecedent numbers: ", ant_numbers)
    data_matrix = np.load("datos.npy")

    train_matrix = data_matrix[:2000]
    test_matrix = data_matrix[-3000:]

    linear_train_obj = wang_mendel(
        "linear", train_matrix, ant_numbers, distance_range, angle_range, linear_speed_range)
    angular_train_obj = wang_mendel(
        "angular", train_matrix, ant_numbers, distance_range, angle_range, angular_speed_range)

    (linear_mse, _) = generate_test(
        linear_train_obj, test_matrix[:, 0:2], test_matrix[:, 2])

    (angular_mse, _) = generate_test(angular_train_obj,
                                     test_matrix[:, 0:2], test_matrix[:, 3])

    del data_matrix

    numbers_id = int("".join(map(str, ant_numbers)))

    results = np.load("results.npy", allow_pickle=True)
    results = np.append(results, [(numbers_id, linear_mse, angular_mse)],
                        axis=0)
    np.save("results.npy", results, allow_pickle=True)


def plot_data(data_matrix):
    fig = plt.figure()
    ax1 = fig.add_subplot(411)
    ax1.plot(data_matrix[:, 0])
    ax1.set_title('Distance error')
    ax1.set_ylabel('m')
    ax1.set_xlabel('time')
    ax1.grid(True)

    ax2 = fig.add_subplot(412)
    ax2.plot(data_matrix[:, 1])
    ax2.set_title('Angle error')
    ax2.set_ylabel('rad')
    ax2.set_xlabel('time')
    ax2.grid(True)

    ax3 = fig.add_subplot(413)
    ax3.plot(data_matrix[:, 2])
    ax3.set_title('Linear velocity')
    ax3.set_ylabel('m/s')
    ax3.set_xlabel('time')
    ax3.grid(True)

    ax4 = fig.add_subplot(414)
    ax4.plot(data_matrix[:, 3])
    ax4.set_title('Angular velocity')
    ax4.set_ylabel('rad/s')
    ax4.set_xlabel('time')
    ax4.grid(True)

    plt.tight_layout()
    plt.show()


def train_model(distance_range, angle_range, linear_speed_range, angular_speed_range):

    try:
        results = np.load("results.npy", allow_pickle=True)
    except IOError:
        results = np.empty((0, 3))

    antecedents = [3, 5, 7, 9, 11, 13, 15, 17, 19]
    parents = 3

    antecedent_numbers = [
        (k, j, i) for i in antecedents for j in antecedents for k in antecedents]

    active = np.empty(parents, dtype=object)

    for i in range(parents):
        active[i] = mp.Process(target=generate_rules,
                               args=(antecedent_numbers[i], distance_range, angle_range, linear_speed_range, angular_speed_range))
        active[i].daemon = False
        active[i].start()

    pos = parents

    start_time = time()
    while active.any():
        for index, process in enumerate(active):
            if process is None:
                continue

            if not process.is_alive():
                if pos < len(antecedent_numbers):

                    num_id = int("".join(map(str, antecedent_numbers[pos])))
                    if num_id in results[:, 0]:
                        pos += 1
                        print("Skipping: ", antecedent_numbers[pos])
                        continue

                    active[index] = mp.Process(
                        target=generate_rules, args=(antecedent_numbers[pos], distance_range, angle_range, linear_speed_range, angular_speed_range))
                    active[index].daemon = False
                    active[index].start()
                    print(
                        f"{pos/len(antecedent_numbers)*100}% done in {time()-start_time} seconds")
                    pos += 1
                else:
                    active[index] = None


def test_model(linear_antecedents, angular_antecedents, distance_range,
               angle_range, max_linear_speed, max_angular_speed):

    data_matrix = np.load("datos.npy")

    train_matrix = data_matrix[:2000]
    test_matrix = data_matrix[-3000:]

    linear_train_obj = wang_mendel(
        "linear", train_matrix, linear_antecedents, distance_range, angle_range, max_linear_speed)
    angular_train_obj = wang_mendel(
        "angular", train_matrix, angular_antecedents, distance_range, angle_range, max_angular_speed)

    np.save("linear_rules", linear_train_obj.reduced_rules)
    np.save("angular_rules", angular_train_obj.reduced_rules)

    linear_train_obj.plot_antecedents()
    linear_train_obj.plot_output_antecedents()

    angular_train_obj.plot_antecedents()
    angular_train_obj.plot_output_antecedents()

    (linear_mse, linear_outputs) = generate_test(
        linear_train_obj, test_matrix[:, 0:2], test_matrix[:, 2])

    (angular_mse, angular_outputs) = generate_test(angular_train_obj,
                                                   test_matrix[:, 0:2], test_matrix[:, 3])

    plt.figure(figsize=(10, 5))
    plt.rcParams.update({'font.size': 15})

    plt.gca().yaxis.grid(True)
    plt.plot(test_matrix[:, 2], label='MG')
    plt.plot(linear_outputs, 'r-.', label='Pred')
    plt.title("Linear model\nMSE:"+str(round(linear_mse, 4)))
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.rcParams.update({'font.size': 15})

    plt.gca().yaxis.grid(True)
    plt.plot(test_matrix[:, 3], label='MG')
    plt.plot(angular_outputs, 'r-.', label='Pred')
    plt.title("Angular model\nMSE:"+str(round(angular_mse, 4)))
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    DISTANCE_RANGE = [0, 5]
    ANGLE_RANGE = [-np.pi, np.pi]
    LINEAR_SPEED_RANGE = (0, 0.75)
    ANGULAR_SPEED_RANGE = (-1, 1)

    # train_model(DISTANCE_RANGE, ANGLE_RANGE, LINEAR_SPEED_RANGE, ANGULAR_SPEED_RANGE)

    test_model([9, 17, 15], [3, 13, 13], DISTANCE_RANGE,
               ANGLE_RANGE, LINEAR_SPEED_RANGE, ANGULAR_SPEED_RANGE)
