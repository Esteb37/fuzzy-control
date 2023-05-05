import numpy as np
from T1_set import T1_LeftShoulder, T1_RightShoulder, T1_Triangular
from matplotlib import pyplot as plt


class wang_mendel(object):

    def __init__(self, output_type, train_data_matrix, antecedent_number):

        self.output_type = output_type

        if (output_type not in ['linear', 'angular']):
            raise ValueError(
                "The output type must be either 'linear' or 'angular'")

        # Break the input matrix into vectors
        self.train_distances = train_data_matrix[:, 0]
        self.train_angles = train_data_matrix[:, 1]

        train_linear_vels = train_data_matrix[:, 2]
        train_angular_vels = train_data_matrix[:, 3]

        self.train_outputs = train_linear_vels if output_type == 'linear' else train_angular_vels

        self.train_data_matrix = train_data_matrix
        self.training_size = len(train_data_matrix)

        self.distance_antecedents = self.generate_antecedents(
            self.train_distances, antecedent_number)

        self.angle_antecedents = self.generate_antecedents(
            self.train_angles, antecedent_number)

        self.output_antecedents = self.generate_antecedents(
            self.train_outputs, antecedent_number)

        self.antecedent_number = antecedent_number

        self.__reduced_rules = self.rule_matrix_generating()

    def plot_antecedents(self):

        for i in range(1, self.antecedent_number + 1):
            interval = self.distance_antecedents[i].interval

            mf_degrees = self.distance_antecedents[i].get_mf_degrees()

            x = np.linspace(interval[0], interval[1], len(mf_degrees))

            plt.subplot(311).plot(x, mf_degrees)
            plt.title("Membership Functions")
            plt.ylabel("Distance")

        for i in range(1, self.antecedent_number + 1):
            interval = self.angle_antecedents[i].interval

            mf_degrees = self.angle_antecedents[i].get_mf_degrees()

            x = np.linspace(interval[0], interval[1], len(mf_degrees))

            plt.subplot(312).plot(x, mf_degrees)
            plt.ylabel("Angle")

        for i in range(1, self.antecedent_number + 1):
            interval = self.output_antecedents[i].interval

            mf_degrees = self.output_antecedents[i].get_mf_degrees()

            x = np.linspace(interval[0], interval[1], len(mf_degrees))

            plt.subplot(313).plot(x, mf_degrees)
            plt.ylabel("Output (" + self.output_type + ")")

        plt.show()

    def generate_antecedents(self, training_data, antecedent_number):

        max_value = max(training_data)
        min_value = min(training_data)

        antecedents = {}

        step = ((max_value - min_value) /
                (antecedent_number - 1)) / 4.0

        for i in range(1, antecedent_number + 1):

            mean = min_value + (i - 1) * step * 4.0
            if i == 1:
                antecedents[i] = T1_LeftShoulder(
                    mean, step, 500)
            elif i == antecedent_number:
                antecedents[i] = T1_RightShoulder(
                    mean, step, 500)
            else:
                antecedents[i] = T1_Triangular(
                    mean, step, 500)

        return antecedents

    def rule_matrix_generating(self):

        # For each training record return the x value,
        # membership and number of antecedent with the highest membership
        distance_memberships = self.assign_points(
            self.distance_antecedents, self.train_distances)

        angle_memberships = self.assign_points(
            self.angle_antecedents, self.train_angles)

        output_memberships = self.assign_points(
            self.output_antecedents, self.train_outputs)

        # Each rule will have the following shape:
        # [distance_antecedent, angle_antecedent, output_antecedent, rule_degree]
        all_rule_matrix = np.zeros(
            [self.training_size, 4])

        for i in range(self.training_size):

            distance_membership = distance_memberships[i][2]
            angle_membership = angle_memberships[i][2]
            output_membership = output_memberships[i][2]

            # The rule degree is obtained by multiplying the membership degrees of the antecedents
            rule_degree = distance_memberships[i][1] * \
                angle_memberships[i][1] * \
                output_memberships[i][1]

            all_rule_matrix[i] = np.array(
                [distance_membership, angle_membership, output_membership, rule_degree])

        return self.rule_reduction(all_rule_matrix)

    def assign_points(self, antecedents, training_data):
        """
            This function returns an array that contains,
            for each value "x" in the training data, a touple,
            with the value itself, the membership degree and the
            antecedent with the highest membership degree.
        """

        memberships = np.empty(
            [len(training_data), 3])

        for index, x in enumerate(training_data):
            memberships[index][0] = x
            memberships[index][1:3] = self.get_antIndex_and_maxDegree(
                antecedents, x)

        return memberships

    def get_antIndex_and_maxDegree(self, antecedents, x):

        max_degree = 0.0

        for i in antecedents:
            degree = antecedents[i].get_degree(x)
            if degree > max_degree:
                max_degree = degree
                ant_index = i

        if max_degree == 0.0:
            raise ValueError("There is no max degree")

        return ((max_degree, ant_index))

    def rule_reduction(self, all_rule_matrix):

        for i, _ in enumerate(all_rule_matrix):
            temp_rule_1 = all_rule_matrix[i]
            if not np.isnan(temp_rule_1).any():
                for t in range(i + 1, len(all_rule_matrix)):
                    temp_rule_2 = all_rule_matrix[t]
                    # check antecedent equality
                    if np.array_equal(temp_rule_1[0:2], temp_rule_2[0:2]):
                        # check degree and keep the greatest
                        if temp_rule_2[-1] > temp_rule_1[-1]:
                            # the rule, with lower degree, is replaced by the one with higher degree
                            all_rule_matrix[i] = all_rule_matrix[t]
                        all_rule_matrix[t] = np.nan

        return all_rule_matrix[~np.isnan(all_rule_matrix).any(axis=1)]

    @ property
    def reduced_rules(self):
        return self.__reduced_rules


# np.savetxt(" .csv",self.reduced_rules,delimiter=",")
