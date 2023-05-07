from train_wm import wang_mendel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from T1_set import T1_LeftShoulder, T1_RightShoulder, T1_Triangular
from T1_output import T1_Triangular_output, T1_RightShoulder_output, T1_LeftShoulder_output
from time import time
import multiprocessing as mp
import pickle
import psutil


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

    all_firing_strengths, train_obj, parent_id, process_id = params

    output_results = []

    for firing_strengths in all_firing_strengths:

        rule_output_strength = np.empty([len(train_obj.reduced_rules), 1])
        rule_output_strength.fill(np.NaN)

        for rule_index, rule in enumerate(train_obj.reduced_rules):
            rule_output_strength[rule_index] = individual_rule_output(
                firing_strengths, rule[0:2])

        firing_level_for_each_output = union_strength_of_same_antecedents(
            rule_output_strength, train_obj.reduced_rules[:, 2])

        centroid = generate_outputs_object(
            firing_level_for_each_output, train_obj.output_antecedents)
        output_results.append(centroid)

    return output_results


def apply_rules_to_inputs_parallel(all_firing_strengths, train_obj, outputs, parent_id):
    num_processes = 8  # get number of available CPU cores
    # create a pool of worker processes
    pool = mp.Pool(num_processes)
    # calculate chunk size for each process
    chunksize = int(len(all_firing_strengths) / num_processes)

    # apply the function to each chunk of input firing strengths in parallel
    results = pool.map_async(apply_rules_to_inputs, [(chunk, train_obj, parent_id, index)
                                                     for index, chunk in enumerate(chunks(all_firing_strengths, chunksize))])
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


def generate_test(train_obj, inputs, outputs, process_id):
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

    mse, output_results = apply_rules_to_inputs_parallel(
        firing_strengths, train_obj, outputs, process_id)

    return (mse, output_results)


def generate_rules(ant_numbers):

    process_id = f"{ant_numbers[0]}{ant_numbers[1]}{ant_numbers[2]}"
    start = time()

    data_matrix = np.load("datos.npy")

    linear_train_obj = wang_mendel("linear", data_matrix[:2000], ant_numbers)
    angular_train_obj = wang_mendel("angular", data_matrix[:2000], ant_numbers)
    (linear_mse, linear_outputs) = generate_test(linear_train_obj,
                                                 data_matrix[-3000:][:, 0:2], data_matrix[-3000:][:, 2], process_id)
    (angular_mse, angular_outputs) = generate_test(angular_train_obj,
                                                   data_matrix[-3000:][:, 0:2], data_matrix[-3000:][:, 3], process_id)

    del data_matrix

    return (ant_numbers, linear_mse, angular_mse, linear_outputs, angular_outputs)


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


def main():

    antecedents = [3, 5, 7, 9, 11, 13, 15, 17, 19]

    proc_num = 8

    start = time()
    procs = np.empty(proc_num, dtype=object)

    pos = 0
    start = time()
    for i in antecedents:
        for j in antecedents:
            for k in antecedents:
                if pos < proc_num:
                    antecedent_numbers = (i, j, k)
                    process = mp.Process(target=generate_rules,
                                         args=(antecedent_numbers,))
                    process.daemon = False
                    procs[pos] = process
                    pos += 1

    for p in procs:
        p.start()

    for p in procs:
        p.join()

    print(f"Time taken: {time() - start}")

    """
    with open('results.pkl', 'wb') as fp:
        pickle.dump(results, fp)

    # Plotting the results
    plt.figure(figsize=(10, 5))
    plt.rcParams.update({'font.size': 15})

    plt.gca().yaxis.grid(True)
    plt.plot(list(results.keys()), [x[0] for x in results.values()],
             label='Linear MSE')
    plt.plot(list(results.keys()), [x[1] for x in results.values()],
             label='Angular MSE')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.xlabel("Antecedent number")
    plt.ylabel("MSE")
    plt.title("MSE vs Antecedent Number")
    plt.tight_layout()
    plt.show()

    best_ant_number = results[results.keys()[0]]

    print("\nBest antecedent number: ", best_ant_number)

    print("\n---------------- Antecedent number: ",
          best_ant_number, "----------------")

    linear_train_obj = wang_mendel("linear", train_matrix, best_ant_number)
    angular_train_obj = wang_mendel("angular", train_matrix, best_ant_number)

    linear_train_obj.plot_antecedents()

    linear_train_obj.plot_output_antecedents()
    angular_train_obj.plot_output_antecedents()

    best_results = results[best_ant_number]
    linear_test_outputs = test_matrix[:, 2]
    angular_test_outputs = test_matrix[:, 3]

    linear_mse = best_results[0]
    angular_mse = best_results[1]
    linear_pred_outputs = best_results[2]
    angular_pred_outputs = best_results[3]

    plt.figure(figsize=(10, 5))
    plt.rcParams.update({'font.size': 15})

    plt.gca().yaxis.grid(True)
    plt.plot(linear_test_outputs, label='MG')
    plt.plot(linear_pred_outputs, 'r-.', label='Pred')
    plt.title("Linear model\nMSE:"+str(round(linear_mse, 4)))
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.rcParams.update({'font.size': 15})

    plt.gca().yaxis.grid(True)
    plt.plot(angular_test_outputs, label='MG')
    plt.plot(angular_pred_outputs, 'r-.', label='Pred')
    plt.title("Angular model\nMSE:"+str(round(angular_mse, 4)))
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.show()
    """


if __name__ == "__main__":
    main()
