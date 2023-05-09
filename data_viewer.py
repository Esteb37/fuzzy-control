import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def id_to_array(id_):
    id_str = str(int(id_))

    i = 0
    array = []
    while i < len(id_str):
        if id_str[i] == '1':
            num = int(id_str[i:i+2])
            i += 2
        else:
            num = int(id_str[i])
            i += 1
        array.append(num)
    return array


def print_mins(data):

    linear_mse = data[:, 1]
    angular_mse = data[:, 2]
    ids = np.array([id_to_array(i) for i in data[:, 0]])

    min_linear_id = ids[np.argmin(linear_mse)]

    min_angular_id = ids[np.argmin(angular_mse)]
    print("Minimum linear MSE: ", np.min(
        linear_mse), " at ", min_linear_id)

    print("Minimum angular MSE: ", np.min(
        angular_mse), " at ", min_angular_id)


def plot_averaged_output(data):
    joint = np.column_stack(
        (np.array([id_to_array(i) for i in data[:, 0]]), data[:, 1], data[:, 2]))

    joint = joint[joint[:, 2].argsort()]
    joint = joint[joint[:, 1].argsort(kind='mergesort')]
    joint = joint[joint[:, 0].argsort(kind='mergesort')]

    avg_linear = 0
    avg_angular = 0

    distance_vs_angle = []

    last_distance = joint[0, 0]
    last_angle = joint[0, 1]

    for row in joint:

        distance = row[0]
        angle = row[1]

        if distance != last_distance or angle != last_angle:
            distance_vs_angle.append(
                (row[0], row[1], avg_linear/9, avg_angular/9))
            avg_linear = 0
            avg_angular = 0
            last_distance = distance
            last_angle = angle

        avg_linear += row[3]
        avg_angular += row[4]

    distance_vs_angle = np.array(distance_vs_angle)

    # take the first three columns
    dva_linear = distance_vs_angle[:, 2]
    dva_linear = np.column_stack(
        (distance_vs_angle[:, 0], distance_vs_angle[:, 1], dva_linear))

    dva_angular = distance_vs_angle[:, 3]
    dva_angular = np.column_stack(
        (distance_vs_angle[:, 0], distance_vs_angle[:, 1], dva_angular))

    # plot dva_linear as a 3D surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(dva_linear[:, 0], dva_linear[:, 1], dva_linear[:, 2])
    ax.set_xlabel("Distance")
    ax.set_ylabel("Angle")
    ax.set_zlabel("Linear MSE")
    plt.show()

    # plot dva_angular as a 3D surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(dva_angular[:, 0], dva_angular[:, 1], dva_angular[:, 2])
    ax.set_xlabel("Distance")
    ax.set_ylabel("Angle")
    ax.set_zlabel("Angular MSE")
    plt.show()


def plot_averaged_distance(data):
    joint = np.column_stack(
        (np.array([id_to_array(i) for i in data[:, 0]]), data[:, 1], data[:, 2]))

    joint = joint[joint[:, 0].argsort()]
    joint = joint[joint[:, 1].argsort(kind='mergesort')]
    joint = joint[joint[:, 2].argsort(kind='mergesort')]

    avg_linear = 0
    avg_angular = 0

    angle_vs_output = []

    last_output = joint[0, 2]
    last_angle = joint[0, 1]

    for row in joint:

        output = row[2]
        angle = row[1]

        if output != last_output or angle != last_angle:
            angle_vs_output.append(
                (row[1], row[2], avg_linear/9, avg_angular/9))
            avg_linear = 0
            avg_angular = 0
            last_output = output
            last_angle = angle

        avg_linear += row[3]
        avg_angular += row[4]

    angle_vs_output = np.array(angle_vs_output)

    avo_linear = angle_vs_output[:, 2]
    avo_linear = np.column_stack(
        (angle_vs_output[:, 0], angle_vs_output[:, 1], avo_linear))

    avo_angular = angle_vs_output[:, 3]
    avo_angular = np.column_stack(
        (angle_vs_output[:, 0], angle_vs_output[:, 1], avo_angular))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(avo_linear[:, 0], avo_linear[:, 1], avo_linear[:, 2])
    ax.set_xlabel("Output")
    ax.set_ylabel("Angle")
    ax.set_zlabel("Linear MSE")
    plt.show()

    # plot dva_angular as a 3D surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(avo_angular[:, 0], avo_angular[:, 1], avo_angular[:, 2])
    ax.set_xlabel("Output")
    ax.set_ylabel("Angle")
    ax.set_zlabel("Angular MSE")
    plt.show()


def plot_averaged_angle(data):
    joint = np.column_stack(
        (np.array([id_to_array(i) for i in data[:, 0]]), data[:, 1], data[:, 2]))

    joint = joint[joint[:, 1].argsort()]
    joint = joint[joint[:, 0].argsort(kind='mergesort')]
    joint = joint[joint[:, 2].argsort(kind='mergesort')]

    avg_linear = 0
    avg_angular = 0

    distance_vs_output = []

    last_output = joint[0, 2]
    last_distance = joint[0, 0]

    for row in joint:

        output = row[2]
        distance = row[0]

        if output != last_output or distance != last_distance:
            distance_vs_output.append(
                (row[0], row[2], avg_linear/9, avg_angular/9))
            avg_linear = 0
            avg_angular = 0
            last_output = output
            last_distance = distance

        avg_linear += row[3]
        avg_angular += row[4]

    distance_vs_output = np.array(distance_vs_output)

    dvo_linear = distance_vs_output[:, 2]
    dvo_linear = np.column_stack(
        (distance_vs_output[:, 0], distance_vs_output[:, 1], dvo_linear))

    dvo_angular = distance_vs_output[:, 3]
    dvo_angular = np.column_stack(
        (distance_vs_output[:, 0], distance_vs_output[:, 1], dvo_angular))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(dvo_linear[:, 0], dvo_linear[:, 1], dvo_linear[:, 2])
    ax.set_xlabel("Distance")
    ax.set_ylabel("Output")
    ax.set_zlabel("Linear MSE")
    plt.show()

    # plot dva_angular as a 3D surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(dvo_angular[:, 0], dvo_angular[:, 1], dvo_angular[:, 2])
    ax.set_xlabel("Distance")
    ax.set_ylabel("Output")
    ax.set_zlabel("Angular MSE")
    plt.show()


def plot_raw_data(data):

    linear_mse = data[:, 1]
    angular_mse = data[:, 2]
    ids = np.array([str(int(i)) for i in data[:, 0]])

    plt.plot(ids, linear_mse, "ro", markersize=1)
    plt.xticks(ids, ids, fontsize=1)
    plt.xlabel("Antecedents")
    plt.ylabel("Linear MSE")
    plt.show()

    plt.plot(ids, angular_mse, "ro", markersize=1)
    plt.xticks(ids, ids, fontsize=1)
    plt.xlabel("Antecedents")
    plt.ylabel("Angular MSE")
    plt.show()


def plot_4D(data):
    joint = np.column_stack(
        (np.array([id_to_array(i) for i in data[:, 0]]), data[:, 1], data[:, 2]))

    # plot first three dimensions as a 3D scatter and the fourth as size

    # make the size difference more visible with an exponential function
    values = np.exp(joint[:, 3])
    values = np.exp((4*np.max(values))/values)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(joint[:, 0], joint[:, 1], joint[:, 2], s=values, c=values)
    ax.set_xlabel("Distance")
    ax.set_ylabel("Angle")
    ax.set_zlabel("Output")
    fig.suptitle("Linear MSE")
    plt.show()

    values = np.exp(joint[:, 4])
    values = np.exp((4*np.max(values))/values)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(joint[:, 0], joint[:, 1], joint[:, 2], s=values, c=values)
    ax.set_xlabel("Distance")
    ax.set_ylabel("Angle")
    ax.set_zlabel("Output")
    fig.suptitle("Angular MSE")
    plt.show()


def main():
    data = np.load("results.npy")

    print_mins(data)
    # plot_averaged_output(data)
    # plot_data(data)
    # plot_averaged_distance(data)
    # plot_averaged_angle(data)
    plot_4D(data)


if __name__ == "__main__":
    main()
