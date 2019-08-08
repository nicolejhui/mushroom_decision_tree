import numpy as np
import pandas as pd

# reads the data and creates a numpy array containing all mushroom examples
all_examples = pd.read_csv('data/mush_train.data', header=None)
all_examples = all_examples.values

# a list of attribute names to use when printing the final output
attributes_list = ['cap-shape', 'cap-surface', 'cap-color', 'bruises?', 'odor', 'gill-attachment', 'gill-spacing',
                 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
                 'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type',
                 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat', 'root']

# creates a node class
class Node:
    def __init__(self, name, examples, depth=0):

        # name of the node
        self.name = name

        # dictionary where they key is the child node name and the value is the child node itself
        self.children = {}

        # list of the examples that the node holds
        self.examples = examples

        # used to determine number of tabs to print when visualizing the output
        self.depth = depth

        # initializes entropy
        self.entropy = 1

        # split feature holds the name of the feature that was split on to create this node
        self.split_feature = None

        # dictionary where the key is the attribute and the value is the information gain
        self.info_gain_dict = {}

        # predicted label of a terminal node
        self.prediction = None


    # calculates entropy
    def get_entropy(self):
        label, label_counts = np.unique(self.examples[:, 0], return_counts=True)
        # print(label, label_counts)
        label1_prob = label_counts[0] / self.examples.shape[0]
        label2_prob = label_counts[1] / self.examples.shape[0]
        # print(label1_prob, label2_prob)
        out = - (label1_prob * (np.log2(label1_prob)) + (label2_prob * (np.log2(label2_prob))))
        # print(out)
        return out


    # calculates conditional entropy
    # attribute_col is the index number for an attribute
    def con_entropy(self, attribute_col):
        conditional_array = {}
        for example in self.examples:
            if example[attribute_col] not in conditional_array:
                conditional_array[example[attribute_col]] = [0, 0]
            if example[0] == 'e':
                conditional_array[example[attribute_col]][0] += 1
            if example[0] == 'p':
                conditional_array[example[attribute_col]][1] += 1
        conditional_entropy = 0
        for unique_var, counts, in conditional_array.items():
            frequency = (counts[0] + counts[1]) / self.examples.shape[0]
            label1_joint_prob = counts[0] / (counts[0] + counts[1])
            label2_joint_prob = counts[1] / (counts[0] + counts[1])
            if label1_joint_prob == 0:
                label1_log_prob = 0
            else:
                label1_log_prob = label1_joint_prob * np.log2(label1_joint_prob)
            if label2_joint_prob == 0:
                label2_log_prob = 0
            else:
                label2_log_prob = label2_joint_prob * np.log2(label2_joint_prob)
            conditional_entropy += frequency * (label1_log_prob + label2_log_prob)
        conditional_entropy = -1 * conditional_entropy
        return conditional_entropy


    # determines the best attribute to split on by calculating information gain using entropy and conditional entropy from previous functions
    def find_best_attribute(self):
        ig_list = []
        for i in range(self.examples.shape[1]):
            if i > 0:
                conditional_entropy = self.con_entropy(i)
                information_gain = self.entropy - conditional_entropy
                ig_list.append(information_gain)
        info_gain_dict = dict(zip(attributes_list, ig_list))
        best_attribute_col = max(info_gain_dict, key=info_gain_dict.get)
        conditional_entropy = self.entropy - info_gain_dict[best_attribute_col]
        self.split_feature = best_attribute_col
        best_attribute_index = attributes_list.index(best_attribute_col) + 1
        best_attribute_col = self.examples[:, best_attribute_index]
        best_attribute_name = max(info_gain_dict, key=info_gain_dict.get)
        self.info_gain_dict = info_gain_dict
        return best_attribute_name, best_attribute_index, best_attribute_col, ig_list, info_gain_dict, conditional_entropy,


    # splits the node and creates children nodes
    def split_node(self):
        # print('-----------------------------------NEXT CALL OF split_node--------------------------------------')
        # print(str('STOP CONDITION CHECK. ENTROPY IS: ' + str(self.entropy)))
        if abs(self.entropy) < 0.001:
            # print('entropy is 0')
            return

        best_attribute_name, best_idx, best_attribute, ig_list, info_gain_dict, conditional_entropy = self.find_best_attribute()

        if self.split_feature is not None:
            # print('INFO GAIN: ' + str(self.info_gain_dict))
            if self.info_gain_dict[self.split_feature] < 0.001:
                # print('info gain is 0')
                return

        unique_val_of_best_attribute = []
        unique_val_key = []
        unique_val_key_number = 0
        self.depth += 1
        for unique_val in np.unique(best_attribute):
            unique_val_key_number += 1
            unique_val_key.append(unique_val_key_number)
            unique_val_of_best_attribute.append(unique_val)
            unique_val_dict = dict(zip(unique_val_key, unique_val_of_best_attribute))
            split = np.empty((0, all_examples.shape[1]), int)
            for example in self.examples:
                if example[best_idx] == unique_val:
                    split = np.append(split, np.array([example]), axis=0)
            child = Node(unique_val_dict[unique_val_key_number], split, depth=self.depth)
            child.entropy = child.con_entropy(best_idx)
            child.info_gain_dict = info_gain_dict
            child.split_feature = best_attribute_name
            self.children[unique_val_dict[unique_val_key_number]] = child


    # recursively builds decision tree using split_node()
    def build_tree(self):
        self.split_node()
        for yeet, child in self.children.items():
            child.build_tree()
            self.predict_label()
        return


    # predicts label for terminal nodes
    def predict_label(self):
        for node_name, node in self.children.items():
            if node.children == {}:
                example_label = [0, 0]
                for example in node.examples:
                    if example[0] == 'e':
                        example_label[0] += 1
                    if example[0] == 'p':
                        example_label[1] += 1
                if example_label[0] > example_label[1]:
                    node.prediction = 'edible'
                if example_label[1] > example_label[0]:
                    node.prediction = 'poisonous'
                if example_label[0] == example_label[1]:
                    node.prediction = 'poisonous'
                # print(self.split_feature, '==', node_name, ':', node.prediction)
        return node_name, node, node.prediction, self.split_feature


    # visualizes the final tree
    def print_tree(self):
        for node_name, node in self.children.items():
            # print(node, node_name, node.prediction)
            if node.children != {}:
                print('\t' * self.depth + f'{self.split_feature} == {node_name}')
                node.print_tree()
            else:
                print('\t' * self.depth + f'{self.split_feature} == {node_name} : {node.prediction}')








