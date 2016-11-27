#!/usr/bin/env python
# -*- coding: utf-8 -*-
# from __future__ import print_function
import collections
import copy
import csv
import itertools
import math
import random
import shelve
import sys
import time
import os
import re
import prettytable
# import traceback

# import future_builtins  as builtins
from datetime import datetime as dt
from datetime import timedelta
from subprocess import Popen
from reportlab.pdfgen import canvas

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

# Defining constants
"""
NeuroCharter
A software program programmed in python 2.7.12
the main role of the program is to perform neural networks for selected data, save the netwrok, and recall anytime.
The program is designed, and implemented by Dr. Mohammad N. Elnesr as one of the research activities by
Alamoudi Chair for Water Research - King Saud University.
This work, and all the research works of the chair i under supervission of Prof. A. A. Alazba
"""

# Defining constants
I_H = 0
H_O = 1

SIGMOID = 0
TANH = 1
SOFTMAX = 2
LINEAR = 3
BINARY = 4
ARCTAN = 5
SOFTSIGN = 6
SINUSOID = 7
BENT = 8
SOFTPLUS = 9
GAUSIAN = 10
SINC = 11

HIDDEN_LAYER = 1
OUTPUT_LAYER = 2

ACTIVATION_FUNCTIONS = {0: 'Sigmoid', 1: 'Tanh', 2: 'Softmax', 3: 'Linear', 4: 'Binary',
                        5: 'ArcTan', 6: 'SoftSign', 7: 'Sinusoid', 8: 'Bent',
                        9: 'SoftPlus', 10: 'Gausian', 11: 'Sinc'}


# definition of Auxiliary Math functions #


def rms_mean(elements):

    """
    A function to calculate the root mean squared mean value of a list of elements
    :param elements: a list of elements
    :return: root mean squared mean value
    """
    return math.sqrt(sum(x * x for x in elements) / len(elements))


def dominant_sign(elements):
    """
    A function to calculate the dominant sign of a set of numbers
    :param elements: a list of numbers
    :return: the dominant sign
    """
    return sum(elements) / abs(sum(elements))


def transpose_matrix(matrix, really_invert=True):
    """
    A function to transpose a matrix or to convert list of tuples to a list of lists
    :param matrix: a list of lists or list of tuples
    :param really_invert: if True, then it transposes the matrix, else it converts list of tuples to a list of lists
    :return: transposed matrix, or the same matrix but converting tuples to lists
    """
    m0 = matrix[0]
    if not (isinstance(m0, list) or isinstance(m0, tuple)):
        return matrix

    if really_invert:
        return map(list, zip(*matrix))
    else:  # to convert tuples to lists
        return map(list, matrix)


def print_matrix(title, mat):
    """
    prints a matrix (list of lists, or of tuples) in a readable form {to the console}
    :param title: the name of the matrix, that will be printed before it
    :param mat: the matrix
    """
    print '~' * len(title)
    print title
    print '~' * len(title)
    for line in mat:
        print line
    print


def elapsed_time(start, end):
    """
    A function that calculate time difference between two times
    :param start: the starting time
    :param end: the ending time
    :return: the time difference in readable format.
    """
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)


def float_range(x, y, jump=1.0):
    """
    A generator quoted from http://j.mp/1V7BE5g by Akseli Palén
    @param x:Starting float value
    @param y:Ending float value
    @param jump:Step
    @return: yields a float function similar to range function
    """
    '''Range for floats. float range'''
    i = 0.0
    x = float(x)  # Prevent yielding integers.
    x0 = x
    epsilon = jump / 2.0
    yield x  # yield always first value
    while x + epsilon < y:
        i += 1.0
        x = x0 + i * jump
        yield x


class Logger(object):
    """
    Lumberjack class - duplicates sys.stdout to a log file and it's okay
    #source: http://stackoverflow.com/q/616645
    """
    def __init__(self, filename="NrChLog.txt", mode="a", buff=0):
        #clearing the log file
        open(filename, "w").close()
        self.stdout = sys.stdout
        self.file = open(filename, mode, buff)
        sys.stdout = self

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)

    def flush(self):
        self.stdout.flush()
        self.file.flush()
        os.fsync(self.file.fileno())

    def close(self):
        if self.stdout != None:
            sys.stdout = self.stdout
            self.stdout = None

        if self.file != None:
            self.file.close()
            self.file = None


class NeuralNetwork:
    """
    A class for creating artificial neural networks
    """
    def __init__(self, num_inputs, num_hidden, num_outputs,
                 activation_functions=(SIGMOID, SIGMOID),
                 hidden_layer_weights=None, hidden_layer_bias=None,
                 output_layer_weights=None, output_layer_bias=None,
                 learning_rate=0.35, parent_study=None, categorical_extra_divisor=1, out_var_bool=None):

        """
        Initialize a neural network
        @param num_inputs: number of input neurons
        @param num_hidden: number of hidden neurons
        @param num_outputs: number of output neurons
        @param activation_functions:
                a list of possible activation functions.
                Default is (SIGMOID, SIGMOID) , for hidden and output layers
                For output layer, it can be a list of activation functions, one for each output neuron.
                 hence, if there are 3 output neurons, the function, for example,
                 may be in the form (SIGMOID, (TANH, SIGMOID, BINARY))
        @param hidden_layer_weights: the hidden_layer_weights,
                Default is None, so the computer will generate random weights
                Or it can be a list of weights to consider (all the weights must be in one list of dimensions 1 * n
                where n = num_inputs * num_hidden)
        @param hidden_layer_bias: same like above but the weights of bias ar only 1 * num_hidden
        @param output_layer_weights: same like above but the weights list is  1 * n
                where n = num_outputs * num_hidden)
        @param output_layer_bias: same like above but the weights of bias ar only 1 * num_output
        @param learning_rate: the learning rate (float, for stability, use 0.1-0.45)
        @param parent_study: the study instance that called this NN
        @param categorical_extra_divisor: a value that is used for categoric neurons,
                when calculating the error through the cost function, we devide that error by this value.
                The default is 1 (no devison), it can be >1 for smaller error, but less reliable networks,
                or can be < 1 for larger error (value), but less overfitted networks.
        @param out_var_bool: for query-type studies, this value should include a list of type of neurons (True for
                Numeric, and False for Categoric). For other studies, this should be left to default (None),
                so that the program fetches the ouput variables types from the Data Class.
        """
        self.num_inputs = num_inputs
        self._num_hidden = num_hidden
        self._num_outputs = num_outputs
        self.learning_rate = learning_rate
        self.parent_study = parent_study
        # var_map = self.parent_study.get_variables_info('loc')
        query = False
        if out_var_bool is None:
            var_bool = self.parent_study.get_variables_info('bool')[1]
        else:
            var_bool = out_var_bool
            query = True
        self.categorical_extra_divisor = categorical_extra_divisor
        self.hidden_layer = self.NeuronLayer(num_hidden, hidden_layer_bias,
                                             HIDDEN_LAYER, self, activation_functions[0])
        self.output_layer = self.NeuronLayer(num_outputs, output_layer_bias,
                                             OUTPUT_LAYER, self, activation_functions[1], variables_types=var_bool)

        self.initialize_weights("IH", hidden_layer_weights, query)
        self.initialize_weights("HO", output_layer_weights, query)

        self.activation_functions = activation_functions

        self._inputs = []
        self._output_targets = []

    def __str__(self):
        """
        Printing Information about the NN
        @return: A string representing the basic information about the network.
        """
        l = int(len('Multi Layered ANN configuration') * 1.2)
        _output = '\n' + '@' * l + '\n'
        _output += '   Multi Layered ANN configuration\n'
        _output += 'Input : Hidden : Output = '
        _output += str(self.num_inputs) + ' : ' + str(self._num_hidden) + ' : ' + str(self._num_outputs)
        _output += '\n' + '@' * l
        return _output

    def initialize_weights(self, layer, given_weights, query=False):
        """
            Initializing weights as random values if not given
            @param layer: the target layer
            @param given_weights: if None, it will generates weights, otherwise, it will assign the given weights.
            @param query: Boolean, if the study_type is query, (True), then the network will read from the stored
                    NN file, otherwise for other studies, the program will act depending on the given_weights flag
        """
        if layer == "IH":
            # Inputs to hidden layer
            from_count = self.num_inputs
            to_count = self._num_hidden
            to_layer = self.hidden_layer
        else:  # layer == "HO":
            # Hidden to Outputs Layer
            from_count = self._num_hidden  # len(self.hidden_layer.neurons)
            to_count = self._num_outputs
            to_layer = self.output_layer

        if not given_weights:
            # creating normalized random numbers by numpy
            norm_weights = np.random.normal(0.0, pow(from_count, -0.5), (to_count, from_count))
            lnw = norm_weights.tolist()
            for h in range(to_count):
                to_layer.neurons[h].weights = lnw[h]

        else:
            if query:
                if layer == "IH":
                    for h in range(to_count):
                        for i in range(from_count):
                            to_layer.neurons[h].weights.append(self.parent_study.temporary['weights_of_i_h'][i][h])
                else:
                    for h in range(to_count):
                        for i in range(from_count):
                            to_layer.neurons[h].weights.append(self.parent_study.temporary['weights_of_h_o'][h][i])

                pass
            else:
                weight_num = 0
                for h in range(to_count):
                    for i in range(from_count):
                        to_layer.neurons[h].weights.append(given_weights[weight_num])
                        weight_num += 1
        pass

    def feed_forward(self, inputs):
        """
        Perform Feed Forward algorithm
        @param inputs: a list of values of the input layer neurons
        @return: list of outputs of the output layer neurons
        """
        hidden_layer_outputs = self.hidden_layer.feed_forward(inputs)
        return self.output_layer.feed_forward(hidden_layer_outputs)

    def calculate_total_error(self, training_sets):
        """
        calculates the errors/costs of the given data set
        @param training_sets: a list of the lines of the training dataset
        @return:  It returns three outputs
                    1- the total cost of the given data set (float)
                    2- The specific error of each output neuron (list)
                    3- The specific MSE of each output neuron (list)
        """
        n = len(training_sets)
        m = len(training_sets[0][1])  # Number of normalized outputs
        total_error = 0
        specific_error = [0] * m  # len(training_outputs) = len(training_sets[1])
        mse = [0] * m  # Mean Square Error

        for t in range(n):
            training_inputs, training_outputs = training_sets[t]
            self.feed_forward(training_inputs)
            for o in range(m):
                error = self.output_layer.neurons[o].calculate_error(training_outputs[o])[
                    0]  # Nesr, see the function
                total_error += error
                specific_error[o] += error
                # the calculated error here is 0.5*(o-t)^2, but MSE is calculated from the term (o-t)^2 without the 0.5
                mse[o] += error * 2

        total_error /= n
        for o in range(m):
            specific_error[o] /= n
            mse[o] /= n
        return total_error, specific_error, mse

    def get_activation_functions(self):
        """
        querying the used activation functions list
        @return: a list of the used activation functions
        """
        return self.activation_functions

    def get_ann_outputs(self):
        """
        querying the current values of output neurons
        @return: A list of values of outputs of each output neuron
        """
        outputs = []
        for neuron in self.output_layer.neurons:
            outputs.append(neuron.output)
        return outputs

    def get_inputs(self):
        """
        returning the initial inputs giving a list of each variable
        @return: transposed inputs matrix
        """
        return transpose_matrix(self._inputs)

    def get_output_targets(self):
        """
        returning the initial outputs
        @return: the initial outputs
        """
        return self._output_targets

    def get_predictions(self, inputs):
        """
        querying the predictions of the current ANN of a given input list
        @param inputs: a list of values of input neurons
        @return: a list of values of output neurons
        """
        outputs = []
        self.feed_forward(inputs)
        for i, neuron in enumerate(self.output_layer.neurons):
            output = self.output_layer.neurons[i].output
            outputs.append(output)
        return outputs

    def get_structure(self):
        """
        Returns the structure of the NN
        @return: A tuple of (# of input neurons, # of hidden neurons, # of output neurons)
        """
        return self.num_inputs, self._num_hidden, self._num_outputs

    def get_weights(self, layer):
        # Getting Weights without bias, and bias also
        """
        Getting Weights and biases of an ANN
        @param layer: the requested layer
        @return: A tuple of (the in-weights targeting the layer, and the biases of the layer)
        """
        _weights_without_bias = []
        _bias = []
        if layer == I_H:
            _weights = self.hidden_layer.nesr_collect_weights()
        else:
            _weights = self.output_layer.nesr_collect_weights()
        for i in _weights:
            _weights_without_bias.append(i[0])
            _bias.append(i[1])
        _b = []
        for bias in _bias:
            _b.append(bias[0])
        return _weights_without_bias, _b

    def clone(self):
        """
        Clones the ANN
        @return: a copy of the cloned ANN
        """
        cloned = NeuralNetwork(self.num_inputs, self._num_hidden, self._num_outputs,
                               activation_functions=self.activation_functions,
                               learning_rate=self.learning_rate, parent_study=self.parent_study,
                               categorical_extra_divisor=self.categorical_extra_divisor, out_var_bool=None)
        for h in range(self._num_hidden):
            cloned.hidden_layer.neurons[h].bias = self.hidden_layer.neurons[h].bias
            for i in range(self.num_inputs):
                cloned.hidden_layer.neurons[h].weights[i] = self.hidden_layer.neurons[h].weights[i]
        for h in range(self._num_outputs):
            cloned.output_layer.neurons[h].bias = self.output_layer.neurons[h].bias
            for i in range(self._num_hidden):
                cloned.output_layer.neurons[h].weights[i] = self.output_layer.neurons[h].weights[i]
        return cloned

    def revert_to(self, ann_instance):
        """
        Copies weights from a cloned ANN to the current ANN
        @param ann_instance: A previous clone of the current ANN
        @return: NONE, just copy the previous clone to the current

        """

        for h in range(self._num_hidden):
            self.hidden_layer.neurons[h].bias = ann_instance.hidden_layer.neurons[h].bias
            for i in range(self.num_inputs):
                self.hidden_layer.neurons[h].weights[i] = ann_instance.hidden_layer.neurons[h].weights[i]
        for h in range(self._num_outputs):
            self.output_layer.neurons[h].bias = ann_instance.output_layer.neurons[h].bias
            for i in range(self._num_hidden):
                self.output_layer.neurons[h].weights[i] = ann_instance.output_layer.neurons[h].weights[i]

    class NeuronLayer:
        """
        Adding a Neuron Layer
        """

        def __init__(self, num_neurons, bias, layer_type, parent_network,
                     activation_function=SIGMOID, variables_types=None):
            # Nesr From now on, each neuron has its own bias So, provided bias should be a list not a value
            # self.bias has been cancelled
            """
            Creates a NN layer
            @param num_neurons: number of neurons in the layer
            @param bias: weights of bias of this layer
            @param layer_type: either 0, 1, or 2 for input hidden, or output
            @param parent_network: the NeuralNetwork Class instance of the paren layer
            @param activation_function: the activation_function associated with the layer
            @param variables_types: only for input and output layers, a list of types (Numeric or Categoric)
                    of the neurons of the layer
            """
            self.neurons = []
            self.parent_network = parent_network
            self.activation_function = activation_function  # each layer has its activation function
            self.layer_type = layer_type  # 1= hidden, 2=output
            self.variables_types = variables_types
            if bias is None:
                # generate a bias for each neuron
                for i in range(num_neurons):
                    self.add_neuron(0.5 - random.random(), activation_function, layer_type, i, variables_types)
                    # # If it is the output layer, then each neuron may have its own activation function
                    # # otherwise, all neurons of the layer must have the same function
                    # if layer_type == 2 and isinstance(activation_function, tuple):
                    #     self.neurons.append(Neuron(random.random(), self,activation_function[i]))
                    # else:
                    #     self.neurons.append(Neuron(random.random(), self, activation_function))
            elif isinstance(bias, tuple) or isinstance(bias, list):
                # copy biasses from input
                # only if the count of provided biasses = the desired neurons
                if len(bias) == num_neurons:
                    for i, b in enumerate(bias):
                        self.add_neuron(b, activation_function, layer_type, i, variables_types)

            else:
                # it is a single value for bias, then it should be equal for all neurons
                for i in range(num_neurons):
                    self.add_neuron(bias, activation_function, layer_type, i, variables_types)

        def add_neuron(self, bias, activation_function, layer_type, neuron_number, variables_types):
            """
            Adding a neuron to the layer
            # If it is the output layer, then each neuron may have its own activation function
            # otherwise, all neurons of the layer must have the same function
            @param bias: the bias wieght associated with the current neuron
            @param activation_function: of this neuron
            @param layer_type: of the parent layer
            @param neuron_number: from 0 to number of neurons for this layer
            @param variables_types: only for input and output layers, the variable type
                    (Numeric or Categoric) for this neuron
            """

            variable_type = None
            if variables_types is not None:
                variable_type = variables_types[neuron_number]
            if layer_type == 2 and isinstance(activation_function, tuple):
                self.neurons.append(self.Neuron(bias, self,
                                                activation_function[neuron_number], variable_type))
            else:
                self.neurons.append(self.Neuron(bias, self, activation_function))
            pass

        def feed_forward(self, inputs):
            """
            takes inputs from synapses and outputs the values after getting out of the neuron
            @param inputs: a list of input values to the layer
            @return: list of outputs from the layer
            """
            outputs = []
            func = 0

            def softmax(x):
                """
                Returns the softmax function
                Input: x as array of list
                Output: the softmax value
                """
                e_x = np.exp(x - np.max(x))
                return e_x / e_x.sum()

            for neuron in self.neurons:
                func = neuron.activation_function  # was = neuron.parent_layer.get_activation_function()
                outputs.append(neuron.calculate_output(inputs))
            # NESR added function

            if func == SOFTMAX:
                outputs = softmax(outputs)
                for i, neuron in enumerate(self.neurons):
                    neuron.output = outputs[i]
            return outputs

        def nesr_collect_weights(self):
            """
            Returns a list of tuples contains weights of neurons of this layer
            """
            _layer_weights = []
            n_weights = range(len(self.neurons[0].weights))
            for n in range(len(self.neurons)):
                _neuron_weights = []
                _neuron_biases = []
                for w in n_weights:
                    _neuron_weights.append(self.neurons[n].weights[w])
                    _neuron_biases.append(self.neurons[n].bias)
                temp = [tuple(_neuron_weights), tuple(_neuron_biases)]
                _layer_weights.append(tuple(temp))
                # Each layer will have the nurons weights on the form
                # [weights then bias of each neuron]
                # for example if the input layer contains 3 neurons, and the hidden contains 4 neurons
                # then the output will be in the form
                # [((w00, w01, w02), b0), ((w10, w11, w12), b1),((w20, w21, w22), b2), ((w30, w31, w32), b3)]
                # Where the first number is the hidden layer's index, and the second number is the input's index
            return _layer_weights

        def get_outputs(self):
            """
            returns a list of the outputs of the current layer
            @return: a list of the outputs of the current layer
            """
            outputs = []
            for neuron in self.neurons:
                outputs.append(neuron.output)
            return outputs

        class Neuron:
            """
            A class of neurons
            """
            def __init__(self, bias, parent, activation_function, variable_type=None):
                """

                @param bias: the bias associated with current neuron
                @param parent: the parent layer of the neuron
                @param activation_function: the function associated with it
                @param variable_type: the type of variable represented by the current neuron
                """
                self.bias = bias
                self.weights = []
                self.parent_layer = parent
                self.inputs = []
                self.output = 0
                self.activation_function = activation_function
                self.is_categoric = not variable_type

            def calculate_output(self, inputs):
                """
                returns the output of this neuron after squashing the input value
                @param inputs: the input value of this neuron
                @return: the output value of this neuron
                """
                self.inputs = inputs
                func = self.activation_function
                self.output = self.apply_activation_function(self.calculate_total_net_input(), func)
                return self.output

            def calculate_total_net_input(self):
                """
                collects the total values that enters into the neuron
                @return: the sum of inputs * weights + bias
                """
                total = 0
                for i in range(len(self.inputs)):
                    total += self.inputs[i] * self.weights[i]
                return total + self.bias

            @staticmethod
            def apply_activation_function(total_net_input, func):
                """
                Apply the activation function to get the output of the neuron
                Nesr Added 2 more functions
                @param total_net_input: the input of the neuron
                @param func: the activation function that will be used
                @return: the squashed value of the input
                """
                # func = self.parent_layer.get_activation_function()
                x = total_net_input
                if func == SIGMOID:
                    return 1 / (1 + math.exp(-x))
                elif func == TANH:

                    if x < -20.0:
                        return -1.0
                    elif x > 20.0:
                        return 1.0
                    else:
                        return math.tanh(x)
                elif func == SOFTMAX:
                    # The SOFTMAX function requires a list of inputs, so I will pass the values as is
                    # Then will be manipulated later at the feed_forward function
                    return x
                elif func == LINEAR:
                    return x
                elif func == BINARY:
                    if x >= 0.5:
                        return 1
                    else:
                        return 0
                elif func == ARCTAN:
                    return math.atan(x)
                elif func == SOFTSIGN:
                    return x / (1 + abs(x))
                elif func == SINUSOID:
                    temp = 0
                    try:
                        temp = math.sin(x)
                    except:
                        if x == float('Inf'):
                            temp = 1
                        elif x == -float('Inf'):
                            temp = 0
                            # print 'Sinusoid error', x
                    finally:
                        return temp  # math.sin(x)
                elif func == BENT:
                    return x + 0.5 * ((x * x + 1) ** 0.5 - 1)
                elif func == SOFTPLUS:
                    temp = 0
                    try:
                        temp = math.log(1 + math.exp(x))
                    except:
                        if x == float('Inf'):
                            temp = 300
                        elif x == -float('Inf'):
                            temp = 0
                        else:
                            temp = 300
                            # print 'Soft plus error', x
                    finally:
                        return temp  # math.sin(x)

                elif func == GAUSIAN:
                    return math.exp(-x * x)
                elif func == SINC:
                    if x == 0.:
                        return 1
                    else:
                        return math.sin(x) / x
                pass

            def calc_delta(self, target_output):
                """
                Determine how much the neuron's total input has to change to move closer to the expected output
                @param target_output: the target outputs of the output layer
                @return: the value of delta as shown below
                """
                return self.derive_cost(target_output) * self.derive_func(self.activation_function)

            def calculate_error(self, target_output):
                """
                Returns the cost function
                The error for each neuron is calculated by the Mean Square Error method:
                NESR changed the equation to yield the difference between output and expected
                in addition to its main role i.e. 0.5(o-t)^2
                @param target_output: the target outputs of the output layer
                @return: a tuple contains (the cost value, target output - calculated output)
                """
                difference = target_output - self.output
                ann_error = 0.5 * difference * difference
                # is_categoric = True if self.parent_layer == 2 else False
                # if self.is_categoric:
                #     #logistic cost function
                #     ann_error = -target_output * math.log(self.output) - (1- target_output) * math.log(1- self.output)
                # else:  # Numeric
                #     # Linear cost function
                #     ann_error = 0.5 * difference * difference
                if self.is_categoric:
                    categorical_extra_divisor = self.parent_layer.parent_network.categorical_extra_divisor
                    ann_error /= categorical_extra_divisor
                return ann_error, difference

            def derive_cost(self, target_output):
                """
                Returns the derivative of the cost function
                @param target_output:
                @return:
                """
                # if self.is_categoric:
                #     # logistic cost function
                #     if target_output == 0:
                #         return 1. / (1.- self.output)
                #     else:
                #         return -1. / self.output
                # else:  # Numeric
                #     # Linear cost function
                #     return self.output - target_output  # this form is a bit faster than -(target_output - self.output)
                if self.is_categoric:
                    categorical_extra_divisor = self.parent_layer.parent_network.categorical_extra_divisor
                    return (self.output - target_output) / categorical_extra_divisor
                return self.output - target_output

            def derive_func(self, func):
                """
                returns the derivative of the activation function for the value of the output of the neuron
                @param func: the function type
                @return: the derivative, depending on the function
                """
                # func = self.parent_layer.get_activation_function()
                x = self.output
                if func == SIGMOID:
                    return x * (1 - x)
                elif func == TANH:
                    return (math.cosh(x)) ** -2
                elif func == SOFTMAX:
                    # since the SOFTMAX function's derevative is similar to the sigmoid when i = m
                    # and is different otherwise. But in ANN, we only deal with the first case
                    # Then we will apply the sigmoid derivative.
                    return x * (1. - x)
                elif func == LINEAR:
                    return 1.
                elif func == BINARY:
                    return 0.
                elif func == ARCTAN:
                    return 1. / (1. + x * x)
                elif func == SOFTSIGN:
                    return 1. / (1. + abs(x)) ** 2
                elif func == SINUSOID:
                    return math.cos(x)
                elif func == BENT:
                    return 1. + 0.5 * x * (x * x + 1) ** -0.5
                elif func == SOFTPLUS:
                    return 1. / (1 + math.exp(-x))
                elif func == GAUSIAN:
                    return -2 * x * math.exp(-x * x)
                elif func == SINC:
                    if x == 0.:
                        return 0
                    else:
                        return math.cos(x) / x - math.sin(x) / (x * x)
                pass

            def neuron_net_input(self, index):
                """
                Returns the net input of a specific neuron
                @param index: the index of the neuron
                @return: the net input of a specific neuron
                """
                return self.inputs[index]


class PlotNeuralNetwork:
    """
    Plot a neural network
    basic code quoted from the following stack exchange article
    http://stackoverflow.com/questions/29888233/how-to-visualize-a-neural-network
    """
    def __init__(self, labels, horizontal__distance_between_layers=10., vertical__distance_between_neurons=2.,
                 neuron_radius=0.5, number_of_neurons_in_widest_layer=9, numeric_categoric_list=None,
                 categoric_is_lighter=True):
        """
        Plots a neural network with varying synapsys widths according to weights
        @param labels: A list contains 2 elements, the fist is a list of all labels, the second is NumInputs
        @param horizontal__distance_between_layers: as written
        @param vertical__distance_between_neurons: as written
        @param neuron_radius: the radius of the circle representing the neuron
        @param number_of_neurons_in_widest_layer: as written
        @param numeric_categoric_list: adding a list of two lists, the first list shows boolean representation
                    of variables types; for example [True, True, False, False], means the first two neurons represent
                    Numeric values, while the last two are categoric.
                    The first sub-list is for inputs, the other for outputs.
                    The gross list is on the form [[True, True, False, False], [False, False, False, True]]
                    This list is optional, the default is None.
                    If left blank, all the neuron will be colored the same color, other wise, Categoric neurons
                    will be lighter or darker depending on the following parameter
        @param categoric_is_lighter: if True, the categoric neurons color will be lighter than numeric ones,
                    and vise-versa
        """
        self.layers = []
        self.biases = []
        self.vertical__distance = vertical__distance_between_neurons
        self.horizontal__distance = horizontal__distance_between_layers
        self.neuron_radius = neuron_radius
        self.widest_layer = number_of_neurons_in_widest_layer
        self.labels = labels  # A list contains 2 elements, the fist is a list of all labels, the second is NumInputs
        self.highest_neuron = 0
        self.numeric_categoric_list = numeric_categoric_list
        self.categoric_is_lighter = categoric_is_lighter

    def add_layer(self, number_of_neurons, layer_type='any', weights=None):
        """
        Adds a layer to be drawn
        @param number_of_neurons: of the desred layer
        @param layer_type: either input, hidden, output, or 'any' for None
        @param weights: weights of synapses associated with this layer (input to it)
        """
        layer = self.PlotLayer(self, number_of_neurons, weights, layer_type)
        self.layers.append(layer)

    def add_bias(self, layer1, layer2, weights=None):
        """

        @param layer1: the bias will be drawn between which layers (this is the from)
        @param layer2: this is the important target layer
        @param weights: the weight of bias
        """
        from_layer = self.layers[layer1]
        to_layer = self.layers[layer2]
        bias = to_layer.PlotBias(self, from_layer, to_layer, weights, layer_type='bias')
        self.biases.append(bias)

    def draw(self, inputs_label="Inputs Layer", outputs_label="Outputs Layer"):
        """
        Draws the whole network depending on its components
        It will recall similar method from the subclasses
        @param inputs_label: The label that will appear to the left of the diagram corresponding to the inputs layer
        @param outputs_label: The label that will appear to the right of the diagram corresponding to the outputs layer
        """
        for layer in self.layers:
            layer.draw()

        for bias in self.biases:
            bias.draw()
        # plt.axis('scaled')
        # for layer in self.layers:
        #     layer.draw_only_neuron()
        # plt.axis('auto')
        # plt.axis('tight')

        xx = (self.layers[0].xz + self.layers[len(self.layers) - 1].xz) / 2 - 2.5
        yy = 0.5
        # label = str(self.layers[1].number_of_neurons) + ' hidden neurons'
        label = "Network Structure is ( " + str(self.layers[0].number_of_neurons) + ' : ' + \
                str(self.layers[1].number_of_neurons) + ' : ' + \
                str(self.layers[2].number_of_neurons) + " )"
        plt.text(xx, yy, label, color='r', zorder=8)

        # max_yz = (self.vertical__distance + self.neuron_radius) * max(n.number_of_neurons for n in self.layers)
        max_yz = self.highest_neuron
        plt.axis([-3, 33, -1, max_yz])  # plt.axis([-1, max_x, -1, 31])
        plt.ylabel(inputs_label)

        frame1 = plt.gca()
        frame1.set_xticklabels([])  # frame1.axes.get_xaxis().set_visible(False)
        frame1.set_yticklabels([])

        ax2 = plt.twinx()
        ax2.set_ylabel(outputs_label)  # ax2.set_xlabel(r"Modified x-axis: $1/(1+X)$")
        ax2.set_yticklabels([])

        # plt.savefig('nesr.png')
        # plt.show()

    class PlotLayer:
        """

        """

        def __init__(self, parent_network, number_of_neurons, weights, layer_type):
            """
            Draws a layer in the current network
            @param parent_network: the network that contains the current layer
            @param number_of_neurons: as written
            @param weights: as written
            @param layer_type: either input, hidden, or output layer
            """
            self.parent_net = parent_network
            self.previous_layer = self.__get_previous_layer()
            self.number_of_neurons = number_of_neurons
            self.xz = self.__calculate_layer_xz_position()
            self.weights = weights
            self.neurons = self.__initialize_neurons(number_of_neurons)
            self.layer_type = layer_type
            self.neuron_labels = [''] * number_of_neurons
            self.neron_types = [True] * number_of_neurons  # True means Numeric
            if layer_type == 'inputs':
                self.neuron_labels = self.parent_net.labels[0][:self.parent_net.labels[1]]
                if self.parent_net.numeric_categoric_list is not None:
                    self.neron_types = self.parent_net.numeric_categoric_list[0]
            elif layer_type == 'outputs':
                self.neuron_labels = self.parent_net.labels[0][self.parent_net.labels[1]:]
                if self.parent_net.numeric_categoric_list is not None:
                    self.neron_types = self.parent_net.numeric_categoric_list[1]

        def __initialize_neurons(self, number_of_neurons):
            """
            initializes the neurons of the layer
            @param number_of_neurons: of this layer
            @return: a list of Neuron Class objects
            """
            neurons = []
            yz = self.left_margin(number_of_neurons)
            for iteration in range(number_of_neurons):
                neuron = self.PlotNeuron(yz, self.xz, self)
                neurons.append(neuron)
                yz += self.parent_net.vertical__distance
                if self.parent_net.highest_neuron < yz:
                    self.parent_net.highest_neuron = yz
            return neurons

        def left_margin(self, number_of_neurons):
            """
            calculate left margin_so_layer_is_centered
            (previously it was bottom to top drawing, so the left was bottom)
            @param number_of_neurons: of this layer
            @return: the margin to be left to the left
            """
            return self.parent_net.vertical__distance * (self.parent_net.widest_layer - number_of_neurons) / 2

        def __calculate_layer_xz_position(self):
            """
            calculates the starting position of the layer
            @return: the horizontal coordinate
            """
            if self.previous_layer:
                return self.previous_layer.xz + self.parent_net.horizontal__distance
            else:
                return 0

        def __get_previous_layer(self):
            """
            specifies the previous layer to the current layer if any
            @return: the layer if exists, or None
            """
            if len(self.parent_net.layers) > 0:
                return self.parent_net.layers[-1]
            else:
                return None

        def __line_between_two_neurons(self, neuron1, neuron2, line_width):
            """

            @param neuron1: the first neuron to join the synappsis from
            @param neuron2: the second neuron to join the synapsis to
            @param line_width: the width of the line
            """
            angle = math.atan((neuron2.yz - neuron1.yz) / float(neuron2.xz - neuron1.xz))
            yz_adjustment = self.parent_net.neuron_radius * math.sin(angle)
            xz_adjustment = self.parent_net.neuron_radius * math.cos(angle)
            line_yz_data = (neuron1.yz - yz_adjustment, neuron2.yz + yz_adjustment)
            line_xz_data = (neuron1.xz - xz_adjustment, neuron2.xz + xz_adjustment)
            col = 'r' if line_width < 0 else 'b'

            line = plt.Line2D(line_xz_data, line_yz_data, linewidth=abs(line_width), color=col, alpha=0.7, zorder=1)
            plt.gca().add_line(line)

        def draw(self):

            """
            A procedure to dtaw the current layer and put labels if any
            """
            for this_layer_neuron_index in range(len(self.neurons)):
                neuron = self.neurons[this_layer_neuron_index]
                # neuron.draw()
                if self.previous_layer:
                    for previous_layer_neuron_index in range(len(self.previous_layer.neurons)):
                        previous_layer_neuron = self.previous_layer.neurons[previous_layer_neuron_index]
                        weight = self.previous_layer.weights[this_layer_neuron_index, previous_layer_neuron_index]
                        self.__line_between_two_neurons(neuron, previous_layer_neuron, weight)
                neuron.draw(self.neuron_labels[this_layer_neuron_index], self.neron_types[this_layer_neuron_index])

        class PlotBias:
            """

            """

            def __init__(self, parent_network, layer1, layer2, weights, layer_type='bias'):
                """
                Initialises bias
                @param parent_network: the parent network
                @param layer1: the layer before layer2
                @param layer2: the layer it goes to
                @param weights: the weights associated with this bias
                @param layer_type: the target layer type
                """
                self.parent_net = parent_network
                self.previous_layer = layer1
                self.target_layer = layer2
                self.xz = (layer1.xz + layer2.xz) / 2
                self.weights = weights
                self.layer_type = layer_type
                self.neuron = layer2.PlotNeuron(0, self.xz, self)

            def draw(self):
                """
                Draws the bias circle
                """
                if self.previous_layer:
                    for neuron_index in range(len(self.target_layer.neurons)):
                        target_layer_neuron = self.target_layer.neurons[neuron_index]
                        self.__line_between_two_neurons(self.neuron, target_layer_neuron, self.weights[neuron_index])
                    self.neuron.draw('bias')

            def __line_between_two_neurons(self, neuron1, neuron2, line_width):
                """
                Draws the  synapses associated with this bias
                @param neuron1: the bias neuron
                @param neuron2: the neuron in the next layer
                @param line_width: width of the synapses
                """
                angle = math.atan((neuron2.yz - neuron1.yz) / float(neuron2.xz - neuron1.xz))
                xz_adjustment = self.parent_net.neuron_radius * math.cos(angle)
                yz_adjustment = self.parent_net.neuron_radius * math.sin(angle)
                line_yz_data = (neuron1.yz - yz_adjustment, neuron2.yz + yz_adjustment)
                line_xz_data = (neuron1.xz - xz_adjustment, neuron2.xz + xz_adjustment)
                col = 'r' if line_width < 0 else 'b'
                line = plt.Line2D(line_xz_data, line_yz_data, linewidth=abs(line_width), color=col, alpha=0.7, zorder=1)
                plt.gca().add_line(line)

        class PlotNeuron:
            """

            """

            def __init__(self, yz, xz, mother_layer):
                """
                Initializes a circle for the neuron
                @param yz: the y coordinate
                @param xz: the x coordinate
                @param mother_layer: the layer it belongs to
                """
                self.yz = yz
                self.xz = xz
                self.mother_layer = mother_layer

            def draw(self, name, n_type=True):
                """
                Draws a circle for the neuron
                @param n_type: the type of the neuron, True if Numeric, False if Categoric
                @param name: the name of the variable associated to the neuron
                """
                layer = self.mother_layer.layer_type
                col = 'w'
                edg = 'b'
                if layer == "inputs":
                    col = 'gold' if n_type else "moccasin"
                    if not n_type and not self.mother_layer.parent_net.categoric_is_lighter:
                        col = 'goldenrod'
                    edg = 'navy'
                    xx = self.xz - 2
                    yy = self.yz - 0.25
                    plt.text(xx, yy, name, color=edg, zorder=8)  # name.replace('I', ' ')
                elif layer == 'hidden':
                    col = 'grey'
                    edg = 'black'
                elif layer == 'outputs':
                    col = 'lime' if n_type else "aquamarine"
                    if not n_type and not self.mother_layer.parent_net.categoric_is_lighter:
                        col = 'limegreen'
                    edg = 'green'
                    xx = self.xz + 1
                    yy = self.yz - 0.25
                    plt.text(xx, yy, name, color=edg, zorder=8)
                elif layer == 'bias':
                    col = 'honeydew'
                    edg = 'blueviolet'
                    xx = self.xz - 1.5
                    yy = self.yz
                    plt.text(xx, yy, name, color=edg, zorder=8)
                # I used the color property, WITH the edgecolor property, but this error appear,
                # so I used forecolor and edgecolor instead
                # UserWarning: Setting the 'color' property will overridethe edgecolor or facecolor properties.
                # warnings.warn("Setting the 'color' property will override"
                circle = plt.Circle((self.xz, self.yz), radius=self.mother_layer.parent_net.neuron_radius,
                                    fill=True, facecolor=col, edgecolor=edg, linewidth=3, zorder=5)
                plt.gca().add_patch(circle)

                # plt.text(self.y, self.x, "INPUT")


class Data:
    """
    Import, read, and Normalize data
    """

    def __init__(self, source_file, num_outputs=1, data_style=None,
                 has_titles=False, has_briefs=False, querying_data=False, parent_study=None,
                 variable_selection_string=''):
        """

        @param source_file: the source file of the data in csv format
                must be placed in the same path of the py file
        @param num_outputs: number of output features (the default is 1)
        @param data_style: a list of the data types of each variable in the form:
                [nI0, cI1, nI2, nI3, cO0, nO1, nO2]
                where n for numeric, c for categoric
                I for input, O for output
                numbering starts from 0 for either inputs or outputs
        @param has_titles: Boolean, if True, the first dataline will be considered titles,
                otherwise Titles will be generated
        @param has_briefs: Boolean, if True, the second dataline will be considered brief titles,
                otherwise they will be generated
        @param querying_data: Boolean, If True, then the datafile is for querying through a saved network
        @param parent_study: the study in which the data is called
        """
        self.source_data_file = source_file
        self.num_outputs = num_outputs
        self.data_style = data_style
        self.basic_data_style = []
        self.has_titles = has_titles
        self.has_briefs = has_briefs
        self.titles = []
        self.briefs = []
        self.parent_study = parent_study
        self.variable_selection_string = variable_selection_string

        if querying_data:
            self.num_inputs = parent_study.num_inputs
            self.classified_titles = [self.titles[:self.num_inputs], self.titles[self.num_inputs:]]
            self.classified_briefs = [self.briefs[:self.num_inputs], self.briefs[self.num_inputs:]]
            temp_data_numeric = range(10)
            temp_data_categoric = ['a', 'b', 'c']
            self.input_variables = []
            self.output_variables = []

            for i, info in enumerate(parent_study.temporary['var_info_input']):
                temp_data = temp_data_numeric if info[3] == 'Numeric' else temp_data_categoric
                temp_variable = self.Variable(temp_data, info[1],info[3], info[2])
                if info[3] == 'Numeric':
                    temp_variable.min = info[4]
                    temp_variable.max = info[5]
                    temp_variable.avg = info[6]
                    temp_variable.stdev = info[7]
                else:
                    temp_variable.num_categories = info[4]
                    temp_variable.unique_values = info[5]
                    temp_variable.normalized_lists = info[6]
                    temp_variable.members_indices = info[7]
                    temp_variable.members = temp_variable.members_indices.keys()
                    temp_variable.values = temp_variable.members
                    temp_variable.frequency = temp_variable.members_indices
                self.input_variables.append(temp_variable)

            for i, info in enumerate(parent_study.temporary['var_info_output']):
                temp_data = temp_data_numeric if info[3] == 'Numeric' else temp_data_categoric
                temp_variable = self.Variable(temp_data, info[1], info[3], info[2])
                if info[3] == 'Numeric':
                    temp_variable.min = info[4]
                    temp_variable.max = info[5]
                    temp_variable.avg = info[6]
                    temp_variable.stdev = info[7]
                else:
                    temp_variable.num_categories = info[4]
                    temp_variable.unique_values = info[5]
                    temp_variable.normalized_lists = info[6]
                    temp_variable.members_indices = info[7]
                    temp_variable.members = temp_variable.members_indices.keys()
                    temp_variable.values = temp_variable.members
                    temp_variable.frequency = temp_variable.members_indices
                self.output_variables.append(temp_variable)

            pass
        else:

            # reading file
            train = self.read_file()
            # specifying inputs/outputs
            len_train = len(train[0])
            if self.variable_selection_string == '':
                self.variable_selection_string = '1' * len_train
            if len(self.variable_selection_string) < len_train:
                temp_str = '{:1>'+str(len_train)+'}'
                self.variable_selection_string = temp_str.format(self.variable_selection_string)
            if len(self.variable_selection_string) > len_train:
                self.variable_selection_string = self.variable_selection_string[:len_train]
            var_selection_list = [int(x) for x in list(self.variable_selection_string)]

            def clean_line(line, criteria):
                if len(criteria) == sum(criteria):  # The criteria is all True
                    return line
                return [x for i, x in enumerate(line) if criteria[i] == 1]

            all_variables_included = True

            if len(var_selection_list) != sum(var_selection_list):
                all_variables_included = False
                self.parent_study.all_variables_included = False
                len_train = sum(var_selection_list)
                self.titles=clean_line(self.titles, var_selection_list)
                self.briefs = clean_line(self.briefs, var_selection_list)


            self.num_inputs = len_train - self.num_outputs

            self.classified_titles = [self.titles[:self.num_inputs], self.titles[self.num_inputs:]]
            self.classified_briefs = [self.briefs[:self.num_inputs], self.briefs[self.num_inputs:]]
            training_sets = []
            for case in train:
                case_list = list(case) if all_variables_included else clean_line(list(case), var_selection_list)
                training_inputs = case_list[:self.num_inputs]
                training_outputs = case_list[self.num_inputs:]
                temp = [training_inputs, training_outputs]
                training_sets.append(temp)

            self.source_data_set = training_sets

            # separate variables
            input_variables_data = [[] for i in range(self.num_inputs)]
            output_variables_data = [[] for i in range(self.num_outputs)]
            record = 0
            while record < len(training_sets):
                for i in range(self.num_inputs):
                    input_variables_data[i].append(training_sets[record][0][i])
                for i in range(self.num_outputs):
                    output_variables_data[i].append(training_sets[record][1][i])
                record += 1

            # Identifying data types
            input_types = ['Numeric'] * self.num_inputs
            output_types = ['Numeric'] * self.num_outputs
            bool_input_types = [True] * self.num_inputs
            bool_output_types = [True] * self.num_outputs
            for i in range(self.num_inputs):
                for cell in input_variables_data[i]:
                    if isinstance(cell, str):
                        input_types[i] = 'Categorical'
                        bool_input_types[i] = False
                        break
            for i in range(self.num_outputs):
                for cell in output_variables_data[i]:
                    if isinstance(cell, str):
                        output_types[i] = 'Categorical'
                        bool_output_types[i] = False
                        break

            self.basic_data_style =[bool_input_types, bool_output_types]
            # var1 = Variable(input_variables_data[i], 'Input' + str(0))
            self.input_variables = [self.Variable(input_variables_data[i], self.classified_titles[0][i],
                                                  input_types[i], str(self.classified_briefs[0][i]))
                                    for i in range(self.num_inputs)]
            self.output_variables = [self.Variable(output_variables_data[i], self.classified_titles[1][i],
                                                   output_types[i], str(self.classified_briefs[1][i]))
                                     for i in range(self.num_outputs)]

            # Check if all variables are valid.
            invalid_in = []
            invalid_out = []

            for num, var in enumerate(self.input_variables):
                if not var.is_valid:
                    invalid_in.append(num)
            for num, var in enumerate(self.output_variables):
                if not var.is_valid:
                    invalid_out.append(num)

            # Now We should modify all previously assigned variables accoding to the new variable list if any.
            if len(invalid_in) > 0:
                invalid_in.sort(reverse=True)
                for invalid_variable in invalid_in:
                    del self.input_variables[invalid_variable]
                    del self.classified_titles[0][invalid_variable]
                    del self.classified_briefs[0][invalid_variable]
                    for record in self.source_data_set:
                        del record[0][invalid_variable]
                    del input_types[invalid_variable]
                    del bool_input_types[invalid_variable]
                    # del self.basic_data_style[0][invalid_variable]
                self.num_inputs = len(self.input_variables)

            if len(invalid_out) > 0:
                invalid_out.sort(reverse=True)
                for invalid_variable in invalid_out:
                    del self.output_variables[invalid_variable]
                    del self.classified_titles[1][invalid_variable]
                    del self.classified_briefs[1][invalid_variable]
                    for record in self.source_data_set:
                        del record[1][invalid_variable]
                    del output_types[invalid_variable]
                    del bool_output_types[invalid_variable]
                    # del self.basic_data_style[1][invalid_variable]
                self.num_outputs = len(self.output_variables)
                if self.num_outputs == 0:
                    print "ALL OUTPUT VARIABLES ARE INVALID\n\n Please check the values and try again." \
                          "\n\nProgram Terminated."
                    exit()

            normalized_variables = []

            print  "\nShowing normalization analysis of the data"

            print '\nInput variables: (' + str(self.num_inputs) + ')\n'
            temp_var_count = '/' + str(self.num_inputs)
            for num, variable in enumerate(self.input_variables):
                print 'Input variable #' + str(num + 1) + temp_var_count
                print str(variable)
                normalized_variables.append(variable.normalize())

            print 'Output variables: (' + str(self.num_outputs) + ')\n'
            temp_var_count = '/' + str(self.num_outputs)
            for num, variable in enumerate(self.output_variables):
                print 'Output variable #' + str(num + 1) + temp_var_count
                print str(variable)
                normalized_variables.append(variable.normalize())

            normalized_data = []
            record = 0
            while record < len(training_sets):
                temp_input = []
                temp_output = []
                for i in range(self.num_inputs):
                    if isinstance(normalized_variables[i][record], list):
                        for num in normalized_variables[i][record]:
                            temp_input.append(float(num))
                    else:
                        temp_input.append(normalized_variables[i][record])
                for i in range(self.num_outputs):
                    if isinstance(normalized_variables[i + self.num_inputs][record], list):
                        for num in normalized_variables[i + self.num_inputs][record]:
                            temp_output.append(float(num))
                    else:
                        temp_output.append(normalized_variables[i + self.num_inputs][record])
                normalized_data.append([temp_input, temp_output])
                record += 1
            self.normalized_data = normalized_data
            self.save_normalized_data_to_file()
            # to make the order of data n a random sequence
            # random.shuffle(self.normalized_data)
            # print self.get_mean_row()

    def read_file(self):
        # train = np.array(list(csv.reader(open(self.source_data_file, "rb"), delimiter=',')))  #  .astype('float')
        """
        reading the data file
        @return: list of lists, each sub list is a data line
        """
        tmp = []
        with open(self.source_data_file, 'rb') as csvfile:
            spam_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in spam_reader:
                # tmp.append(', '.join(row))
                tmp.append(row)

        def read_cell(cel):
            """
            Read data and specify if string or numeric
            @param cel: data cell
            @return: float of string value
            """
            try:  # if is is a number
                return float(cel)
            except:  # otherwise return a trimmed string (with no spaces on either directions
                return cel.strip()
            # return x

        # creating titles and separating data from them

        var_count = len(tmp[0])
        self.num_inputs = var_count - self.num_outputs
        if self.has_titles and self.has_briefs:
            # remove white spaces if any (trim)
            tmp[0] = map(lambda x: x.strip(), tmp[0])
            tmp[1] = map(lambda x: x.strip(), tmp[1])
            self.titles = tmp[0]
            self.briefs = tmp[1]
            tmp = tmp[2:]
        elif self.has_titles:
            # if it only has full titles, we will initiate a brief title
            tmp[0] = map(lambda x: x.strip(), tmp[0])
            self.titles = tmp[0]
            self.briefs = ['In' + str(x) if x < self.num_inputs
                           else 'Ot' + str(x - self.num_inputs) for x in range(var_count)]
            tmp = tmp[1:]
        elif self.has_briefs:
            # if it only has briefs we will consider them as full titles as well
            tmp[0] = map(lambda x: x.strip(), tmp[0])
            self.briefs = tmp[0]
            self.titles = tmp[0]
            tmp = tmp[1:]
        else:  # no titles provided
            self.titles = ['Input variable {' + str(x + 1) + '}' if x < self.num_inputs
                           else 'Output variable {' + str(x - self.num_inputs + 1) + '}' for x in range(var_count)]
            self.briefs = ['In' + str(x + 1) if x < self.num_inputs
                           else 'Ot' + str(x - self.num_inputs + 1) for x in range(var_count)]

        data_ok = []
        for line in tmp:
            lll = []
            for cell in line:
                lll.append(read_cell(cell))
            data_ok.append(lll)
        return data_ok

    def save_normalized_data_to_file(self, clear_file=True, file_name='NormalizedData.csv'):
        """
        Save normalized data to a text file
        @param clear_file:If True, then the file will be cleaned before appending current data,
                otherwise, it will append current data to previous data
        @param file_name: the saving file name, Default value is 'NormalizedData.csv'
        """
        file_name = self.parent_study.new_folder_path + '\\' + file_name
        if clear_file:
            open(file_name, "w").close()
        file_ann = open(file_name, "a")
        for line in self.normalized_data:
            clean_line = str(line)
            clean_line = clean_line.replace('[', '')
            clean_line = clean_line.replace(']', '')
            clean_line = clean_line.replace("'", "")
            file_ann.writelines(clean_line + '\n')
        file_ann.close()

    def get_normalized_structure(self):

        """
        returns the normalized structure of the ANN
        @return: a tuple of (# of inputs, # of hidden, # of outputs)
        """
        inputs = self.num_inputs
        outputs = self.num_outputs
        self.data_style = []
        for i, var in enumerate(self.input_variables):
            if var.data_type != 'Numeric':
                unique_values = len(var.unique_values)
                inputs += unique_values - 1
                for j in range(unique_values):
                    self.data_style.append('cI' + str(i) + '-' + str(j))
            else:
                self.data_style.append('nI' + str(i))

        for i, var in enumerate(self.output_variables):
            if var.data_type != 'Numeric':
                unique_values = len(var.unique_values)
                outputs += unique_values - 1
                for j in range(unique_values):
                    self.data_style.append('cO' + str(i) + '-' + str(j))
            else:
                self.data_style.append('nO' + str(i))

        # Consider hidden size = 2/3 the sum of inputs and outputs
        hidden = int(math.ceil((inputs + outputs) * 2 / 3))
        return inputs, hidden, outputs

    def get_titles(self, type='titles', source='inputs'):
        """
        returns titles of data depending on requested parameters
        @param type: either 'titles', or 'briefs'. The default is 'titles'
        @param source: either 'inputs' to return input variables' titles or briefs,
                        or 'outputs' to return output variables' titles or briefs
        @return: required feature as described above
        """
        variables = self.input_variables if source == 'inputs' else self.output_variables
        tmp = []
        for var in variables:
            tmp.append(var.name if type == 'titles' else var.brief)
        return tmp

    def get_mean_row(self, expanded=True, location='average', encrypted_result=True):
        """

        @param expanded: if True (Default), it will returns number of 'c' equivalent to
                            the number of members of the categoric variable.
                            otherwise, returns one 'c' per categoric variable
        @param location: Default is 'average' to return the line in the middle of the data
                            Other values are '1/4' and '3/4', but not yet implemented
        @param encrypted_result:NOT yet completed, leave defaults please
                                Default is True, to return 'c's or 'n's
        @return:
        """
        mean_row = []
        if location == 'average':
            if encrypted_result:
                for var in self.input_variables:
                    if var.data_type != 'Numeric':  # categoric, it will return only the number of categories
                        if not expanded:
                            mean_row.append('c' + str(len(var.get_basic_stats())))
                        else:
                            for i in range(len(var.get_basic_stats())):
                                mean_row.append('c')
                    else:  # Numeric
                        mean_row.append(var.get_basic_stats()[0])
            else:
                for var in self.input_variables:
                    if var.data_type != 'Numeric':
                        mean_row.extend([0 for i in range(len(var.get_basic_stats()))])

        elif location == '1/4':

            pass
        elif location == '3/4':

            pass

        return mean_row

    def get_data_style(self, required_style='binary'):
        """

        :return:
        @param required_style: Default is 'binary',
                                returns a list of boolean values
                                True = Numeric
                                False = Categoric@return:
                                Otherwise, 'vars' returns just number of variables.
        """
        temp = None
        if required_style == 'binary':
            temp = []
            for var in self.data_style:
                if var[0] == 'c':
                    temp.append(False)
                else:
                    temp.append(True)

        elif required_style == 'vars':
            temp = [[], []]
            for var in self.data_style:
                if var[1] == 'I':
                    temp[0].append(int(var[2]))
                else:
                    temp[1].append(int(var[2]))
        return temp

    class Variable:
        """
        A new variable, to define data_type, min, max, etc...
        """

        def __init__(self, value_list, caption, data_type='Numeric', brief='Var'):
            """

            @param value_list: the list of values of this variable
            @param caption: the caption/ title of the variable
            @param data_type: its data type (Numeric or Categoric)
            @param brief: Its brief title (for small plots)
            """

            def average(s):
                """
                Calculates the average of a list
                @param s: A list of values
                @return: its average
                """
                return sum(s) * 1.0 / len(s)

            self.name = caption
            self.brief = brief
            self.data_type = data_type
            self.values = value_list
            if self.data_type == 'Numeric':
                self.min = min(value_list)
                self.max = max(value_list)
                self.count = len(value_list)
                self.avg = average(value_list)
                self.var = average(map(lambda x: (x - self.avg) ** 2, self.values))
                self.stdev = math.sqrt(self.var)
                if self.min == self.max:  # The variable is single-valued variable and should be removed
                    self.is_valid = False
                    print "The variable of name '" + caption +"' is a single valued variable! it will not be " \
                                                              "considered in the analysis."
                else:
                    self.is_valid = True
            else:
                # self.unique_values = sorted(list(set(value_list)))
                # collections.Counter([i-i%3+3 for i in self.values])
                value_list_ok = [i for i in value_list]
                self.frequency = collections.Counter(value_list_ok)
                # collections.Counter([i-i%3+3 for i in self.values])
                self.members = []  # will be filled after normalization(similar to unique values but sorte descending)
                self.normalized_lists = []  # will be filled after normalization
                self.members_indices = {}
                self.do_one_of_many_normalization()
                # change the unique_values list to be like the members list
                self.unique_values = self.members
                self.num_categories = len(self.unique_values)
                # print self.get_de_normalized_value([.4, .8, .1, .0, .7, .2])
                if self.num_categories == 1:
                    self.is_valid = False
                    print "The variable of name '" + caption + "' is a single valued variable! it will not be " \
                                                               "considered in the analysis."
                else:
                    self.is_valid = True
            pass

        def __str__(self):
            # print 'Variable: ', self.name
            """
            Prints the basic information about the variable to the console
            @return: ... And returns what is printed!
            """
            string = ''
            if self.data_type != 'Numeric':  # Categoric data types
                labels = ['Variable', 'Brief name', 'Data type', 'Values', 'Num. of categories', 'Frequencies']
                l = max(map(lambda x: len(x), labels)) + 1
                values = [self.name, self.brief,  self.data_type, self.unique_values, self.num_categories, dict(self.frequency)]
                for i, label in enumerate(labels):
                    string += '{:<{}s}'.format(label, l) + ': ' + str(values[i]) + '\n'
            else:
                labels = ['Variable', 'Brief name', 'Data type', 'Mean value', 'Standard deviation',
                          'Minimum value', 'Maximum value', 'Count']
                l = max(map(lambda x: len(x), labels)) + 1
                values = [self.name, self.brief, self.data_type, self.avg, self.stdev, self.min, self.max, self.count]
                for i, label in enumerate(labels):
                    string += '{:<{}s}'.format(label, l) + ': ' + str(values[i]) + '\n'

            return string

        def normalize(self):
            """
            Normalizes a variable depending on its type
            @return: the normalized list
            """
            if self.data_type == 'Numeric':
                return self.mini_max()
            else:
                return self.one_of_many()

        def get_normalized_value(self, v):
            """
            returns the normalized value of a value of the variable depending on its type
            @param v: the original value
            @return: the normalized value
            """
            if self.data_type == 'Numeric':
                return self.single_mini_max(v)
            else:
                return self.single_one_to_many(v)

        def get_de_normalized_value(self, v):
            """
            Inverts normalized values to norma values
            @param v: the normalized value
            @return: the original value
            """
            if self.data_type == 'Numeric':
                return self.inverted_mini_max(v)
            else:
                return self.inverted_one_of_many(v)

        def mini_max(self):
            """
            Apply mini_max normalization
            @return: the normalized list of current variable according to minimax procedure
            """
            rng = self.max - self.min
            return map(lambda x: (x - self.min) / rng, self.values)

        def single_mini_max(self, x):
            """
            Apply mini_max normalization to Numeric variables
            @param x: single value to be normalized
            @return: normalized value
            """
            return (x - self.min) / (self.max - self.min)

        def inverted_mini_max(self, n):
            """
            Revert mini_max normalization
            :return:
            @param n: the normalized list of categoric variables
            @return: denormalized value of the input
            """
            return n * (self.max - self.min) + self.min

        def do_one_of_many_normalization(self):
            """
            Normalizes the categoric variable lists
            """
            elements = dict(self.frequency).keys()
            members = {}

            for i, member in enumerate(elements):
                members[member] = i
            self.members_indices = members
            sorted_members = sorted(members.items(), key=lambda value: value[1])
            self.members = [sorted_members[g][0] for g in range(len(sorted_members))]
            self.normalized_lists = []
            for val, member in enumerate(self.members):
                tmp = [0] * len(elements)
                tmp[val] = 1
                self.normalized_lists.append(tmp)
                # return self.normalized_lists

        def one_of_many(self):
            """
            Apply one_of_many normalization to the whole variable values
            @return: normalized categoric list of the variable
            """

            return map(lambda x: self.single_one_to_many(x), self.values)

        def single_one_to_many(self, cat):
            # cell.strip().upper() not in map(lambda x: x.upper(),var.members)
            """
            Normalizes only one category of the variable
            @param cat: category of the variable
            @return: NOrmalized value
            """
            for i, ctg in enumerate(self.members):
                if cat.strip().upper() == ctg.strip().upper():
                    return self.normalized_lists[i]
            return [0] * len(self.normalized_lists[0])

            pass

        def inverted_one_of_many(self, var_norm_list):
            """
            A function that denormalises categoric variables
            @param var_norm_list: a list of normalized categoric variable
            @return: the corresponding category list
            """
            # convverting to a normalized binary list
            tmp = map(lambda x: 1 if x >= 0.5 else 0, var_norm_list)

            # if it is all zeros or it contains mor than two ones, then consider no match
            if sum(tmp) == 0 or sum(tmp) > 2:
                return "*No match*"

            # if the list is one of the caegories in the variable, then return its original name
            if tmp in self.normalized_lists:
                return self.members[self.normalized_lists.index(tmp)]

            get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
            ones_indices = get_indexes(1, tmp)
            ones_values = map(lambda x: var_norm_list[x], ones_indices)
            max_val = max(ones_values)
            new_ind = map(lambda x: 1 if x == max_val else 0, ones_values)

            new_dict = {}
            for i in range(len(ones_indices)):
                new_dict[ones_indices[i]] = new_ind[i]

            new_list = []
            for i, valist in enumerate(tmp):
                if i not in ones_indices:
                    new_list.append(valist)
                else:
                    new_list.append(new_dict[i])

            return self.inverted_one_of_many(new_list)
            # =========================================================
            # ======== THe following has been cancelled ===============
            # =========================================================
            # the category is not recognized
            # then it will be defined as between Cat1, Cat2, ...
            # intermediate_category = "("
            # for i, idx in enumerate(var_norm_list):
            #     if idx == 1:
            #         intermediate_category += self.members[i] + ' ~ '
            # intermediate_category = intermediate_category[:-3] + ')'
            # # if intermediate_category == "May be one of:(":
            # #     intermediate_category = "*No match*"
            # # else:
            # #     intermediate_category = intermediate_category[:-3] + ')'
            #
            # return intermediate_category
            pass

        def is_numeric(self):
            """
            Checks if the variable is NUmeric or not
            @return: True if Numeric, False if Categoric
            """
            return True if self.data_type == 'Numeric' else False

        def get_basic_stats(self):
            """
            Returns the basic statistics depending on the variable type.
            @return: as described above
            """
            if self.data_type == 'Numeric':
                return self.avg, self.min, self.max, self.stdev
            else:
                return self.unique_values


class Study:
    """

    """

    def __init__(self, data_file='DrMohammadEl.nesr', purpose='cross validation', num_outputs=1,
                 data_file_has_titles=False, data_file_has_brief_titles=False,
                 activation_functions=(SIGMOID, SIGMOID),
                 tolerance=0.001, maximum_epochs=1000000, learning_rate=0.4, data_partition=(70, 20),
                 refresh_weights_after_determining_structure=False, find_activation_function=False,
                 validation_epochs=20, layer_size_range=(0.5, 2, 1), start_time=0, adapt_learning_rate=True,
                 try_different_structures=False, annealing_value=0,
                 display_graph_pdf=True, display_graph_windows=False, categorical_extra_divisor=1,
                 previous_study_data_file='NeuroCharterNet.nsr',
                 minimum_slope_to_consider_overfitting=3.0, number_of_epochs_for_overfit_check=10,
                 input_parameters_suggested_values=(), relative_importance_method="ac", folder_prefix="",
                 variable_selection_key='', weights_trials=10):


        """

        @param data_file: the input data file in csv format
        @param purpose: purpose of the study,
                    'query', or 'q': to query about data for a saved ANN (prediction)
                    'cross validation', or 'cv': to train some of the data while checking the errors of the validation dataset
                                        if the error of the latter starts to increase instead of decrease,
                                        the training will stop as it will be considered a n evidence of overfitting.
                                        This is the Default purpose
                    'full run', or 'fr': to run all the data as training set, no validation, and no testing
                    'sequential validation', or 'sv': to run some of the data as training, some as validation,
                                      and some as testing. the validation starts, then traing for maximum of double
                                      the convergence epochs of the training set, the the testing set.
                    'optimization', or 'op': to do similar to the 'validation run', but before that it searched the
                                      best structure and best activation functions. This is the slowest one.
                    'advanced query', or 'aq': to query about data for a saved ANN (prediction) with specific values
                                      of the input parameters.

        @param num_outputs: number of output variables of the datafile, Default = 1
        @param data_file_has_titles: Default is False
        @param data_file_has_brief_titles: Default is False
        @param activation_functions: Default is (SIGMOID, SIGMOID) , for hidden and output layers
                    For output layer, it can be a list of activation functions, one for each output neuron.
                    hence, if there are 3 output neurons, the function, for example,
                    may be in the form (SIGMOID, (TANH, SIGMOID, BINARY))
        @param tolerance: the minimum value to stop if the cost difference is less than it, Default = 0.001
        @param maximum_epochs: The maximum trials (epocs) of training till stopping the training.
                                Normally Training stops when reaching minimum tollerence, but if not it will stop here
                                The Default value is 1000000
        @param learning_rate: the value of the learning rate, The Default value is 0.4
        @param data_partition: A tuple of % of data used for training and  validation.
                                The Default value is (75, 20), Then the remaining (10%) for be for testing
        @param refresh_weights_after_determining_structure: when running on 'optimization' mode
                                If true, the a new random weights will be tried after selecting structure.
                                The Default value is False
        @param find_activation_function: when running on 'optimization' mode,
                                if true, the program will try to optimize the activation function. The Default is False
        @param validation_epochs: number of epochs when seeking for optimized structure.
                                    Only usable for optimization'  and 'Cross Validation' modes, Default = 20
        @param layer_size_range: In optimization' mode, if we seeks for optimum structure,
                                   this is a tuple of range of the minimum and maximum allowed number of hidden neurons
        @param start_time: the time the study starts, if 0 or not provided, it will take the current time
        @param adapt_learning_rate: if True, then the LR will be changed according to the results.
                                    Otherwise, it will be fixed. THe default value is True
        @param try_different_structures: In optimization' mode, if True, the program will seeks the optimum structure
                                    The Default is False
        @param annealing_value: see http://j.mp/1V7eP1C for info.
        @param display_graph_pdf: Displays the outputs as pdf file (Better for publication quality.
                                    The Default is True
        @param display_graph_windows: Displays the outputs in windows (better for speed).
                                    The Default is False
        @param categorical_extra_divisor: a value that is used for categoric neurons,
                when calculating the error through the cost function, we devide that error by this value.
                The default is 1 (no devison), it can be >1 for smaller error, but less reliable networks,
                or can be < 1 for larger error (value), but less overfitted networks.
        @param previous_study_data_file: only in 'query' mode, this is the saved ANN file name,
                                    the Default is 'NeuroCharterNet.nsr'
        @param number_of_epochs_for_overfit_check: The number of points to check if the error increases,
                each epoch is represented by one point. (default number = 10)
        @param minimum_slope_to_consider_overfitting:  the minimum slope of the line formed from the number of
                error points (above) from which, if exceeded it will be considered overfitting.
        @param input_parameters_suggested_values: the suggested values of input parameters, this is used only in
                'advanced query' mode where the ANN will run for the given inputs, and gets the values of outputs.

        @param relative_importance_method: The method which the program will calculate relative importance
                #: the method of calculation
                'Garson' or 'g',
                'Milne' or 'm',
                'Nesr' or 'n',
                'Advance' or 'a',
                'Advance Corrected' or 'AdvCor' or 'ac'
                Default is Advance Corrected method
        @Param weights_trials: The number of trials to identify best set of weights to get least error.
                                Default = 10 trials

        """

        def create_a_folder_for_the_study():
            """ Create a folder and put all files in:"""

            def shorten_time_stamp(time_stamp_text):
                short_stamp = ""
                split_stamp = map(lambda x: int(x), re.findall('.{1,2}', time_stamp_text))
                # print split_stamp
                short_stamp += chr(split_stamp[0] + 50)
                short_stamp += str(split_stamp[1]) if split_stamp[1] < 10 else chr(split_stamp[1] + 55)
                short_stamp += str(split_stamp[2]) if split_stamp[2] < 10 else chr(split_stamp[2] + 55)
                short_stamp += str(split_stamp[3]) if split_stamp[3] < 10 else chr(split_stamp[3] + 55)
                short_stamp += '{0:02d}'.format(split_stamp[4])
                tmp = split_stamp[5] / 2
                short_stamp += str(tmp) if tmp < 10 else chr(tmp + 55)
                return short_stamp

            time_stamp = shorten_time_stamp(dt.now().strftime("%Y%m%d%H%M%S")[2:])  # was [2, -1]
            current_folder = os.getcwd()

            # create a "Results" Folder
            if not os.path.exists(current_folder + "\\Results"):
                os.makedirs(current_folder + "\\Results")

            # The filename of datafile if it is for simulation, otherwise if for retrival take the previous study name
            data_or_ann_name = data_file if data_file !='DrMohammadEl.nesr' else previous_study_data_file
            data_or_ann_name = data_or_ann_name[:-4]

            directory_name = self.folder_prefix + 'NrCh_' + data_or_ann_name + '_' + time_stamp
            new_folder_path = current_folder + "\\Results\\" + directory_name

            if not os.path.exists(new_folder_path):
                os.makedirs(new_folder_path)
            else:
                print "Error, Cannot create folder with name %s\n A folder with the same name already exists\n" \
                      "Please, wait for few seconds then try again." % directory_name
                exit()

            return time_stamp, directory_name, new_folder_path, current_folder

        # Determinig the study's folder and path
        self.folder_prefix = folder_prefix
        self.time_stamp, self.directory_name, self.new_folder_path, self.current_folder = \
            create_a_folder_for_the_study()

        # logging Procedure (Refer to class Logger)
        self.log_file = Logger(self.new_folder_path + '\\' + 'NrChLog.txt')

        # print 'Console log...'

        self.variable_selection_key = variable_selection_key
        temp_key = [int(x) for x in list(variable_selection_key)]
        self.all_variables_included = False
        if len(temp_key) == sum(temp_key):
            self.all_variables_included = True

        print '#####################################################################'
        print '############        NeuroCharter 1.0.B16-52         #################'
        print '#####################################################################'
        print '########      Dr. Mohammad Elnesr & Dr. A.A. Alazba        ##########'
        print '#####################################################################'
        print '\nAnalysis started at           : ' + time.strftime('%Y-%m-%d %H:%M:%S')

        self.source_file_name = data_file if data_file != 'DrMohammadEl.nesr' else previous_study_data_file

        if data_file != 'DrMohammadEl.nesr':
            print 'Input data file               : '+ data_file
        else:
            print 'Input network file            : '+ previous_study_data_file

        print 'Output directory name         : ' + self.directory_name
        print 'Current path                  : ' + self.new_folder_path +"\n"
        print 'All the variables are included: ' + str(self.all_variables_included)
        if not self.all_variables_included:
            print 'Included variables key        : ' + self.variable_selection_key


        self.data_file = data_file

        self.num_inputs = 0
        self.num_outputs = 0

        if data_file == 'DrMohammadEl.nesr' and purpose not in ['advanced query', 'aq']:
            print "A valid input data filename must be provided\n You provided none!\n\nProgram terminated."
            exit()



        if purpose.lower() in ['query', 'q', 'advanced query', 'aq']:
            self.previous_study_data_file = previous_study_data_file
            self.temporary = {}
            self.data_style = None
            query_mode = False if purpose.lower() in ['query', 'q'] else True
            self.input_parameters_suggested_values = input_parameters_suggested_values
            self.perform_query(query_mode)
            print "\nElapsed time throughout the study: ", elapsed_time(start_time, time.time())
            print "\n\n***Done***"
            pass
        else:
            # self.data_file = data_file
            self.data_file_has_titles = data_file_has_titles
            self.data_file_has_brief_titles = data_file_has_brief_titles
            self.data_partition = data_partition
            self.activation_functions = activation_functions
            self.find_activation_function = find_activation_function
            self.refresh_weights_after_determining_structure = refresh_weights_after_determining_structure
            self.validation_epochs = validation_epochs
            self.weights_trials = weights_trials
            self.layer_size_range = layer_size_range
            self.relative_importance_method = relative_importance_method

            self.start_time = start_time if start_time != 0 else time.time()

            self.purpose = purpose.lower()
            self.tolerance = tolerance
            self.maximum_epochs = maximum_epochs
            self.basic_learning_rate = learning_rate
            self.learning_rate = learning_rate
            self.annealing_value = annealing_value
            self.categorical_extra_divisor = categorical_extra_divisor

            self.master_error_list = []
            self.adapt_learning_rate = adapt_learning_rate

            # Start to manipulate data
            self.source_data = Data(data_file, num_outputs,
                                    has_titles=data_file_has_titles, has_briefs=data_file_has_brief_titles,
                                    parent_study=self, variable_selection_string=variable_selection_key)
            self.main_normalized_data = self.source_data.normalized_data
            # the amount of data to be used in training
            self.normalized_data = []  # self.main_normalized_data[:]
            self.structure = self.source_data.get_normalized_structure()
            self.num_inputs_normalized = self.structure[0]
            self.num_outputs_normalized = self.structure[2]
            self.try_different_structures = try_different_structures
            self.display_graph_pdf = display_graph_pdf
            self.display_graph_windows = display_graph_windows
            self.minimum_slope_to_consider_overfitting = minimum_slope_to_consider_overfitting
            self.number_of_epochs_for_overfit_check = number_of_epochs_for_overfit_check
            # Initialize an ANN
            self.ann = None
            # Start running the study
            self.perform_study()

    def perform_query(self, advanced=False):
        """
        Applies the query mode to predict outputs from inputs
        @param advanced: True if advanced mode, False if normal mode
        @return:
        @return: pass
        """
        self.start_time = time.time()

        def read_cell(cel):
            """
            Read data and specify if string or numeric
            @param cel: data cell
            @return: float of string value
            """
            try:
                return float(cel)
            except:
                return cel

        def steps_of_var(lst):
            """ Determine the number of steps that will be executed in a range
            @rtype: tuple of two elements, the first is the number of runs of current range,
                    the second is a boolean value True if the values are float, and False  elsewhere.
            """
            i = 0
            is_float = False
            if len(lst) == 1:
                lst = [0, lst[0], 1]
            if len(lst) == 2:
                lst.append(1)
            if len (lst) > 3:
                print "Warning:\n ========\n\n A list you input contains more than 3 elements!\n If you want to " \
                      "input multiple values, please use a tuple instead of a list.\ni.e. use () instead of []\n"
                print "The list you entered is: ", str(lst)
                print "Now, exiting..."
                exit()
            if isinstance(lst[0], float) or isinstance(lst[1], float) or isinstance(lst[2], float):
                is_float = True
                for dummy_j in float_range(lst[0], lst[1], lst[2]):
                    i += 1
            else:

                for dummy_j in range(lst[0], lst[1] + lst[2], lst[2]):
                    i += 1
            return i, is_float

        self.network_load(self.previous_study_data_file)
        num_norm_inputs, num_hidden, num_norm_outputs = self.structure



        # if the mode is advanced query, so, we must create the input data file
        if advanced:
            # create data file from suggested values:
            given_variables = self.input_parameters_suggested_values
            num_given_variables =  len(given_variables)

            if num_given_variables != self.num_inputs:
                print "The number of given input parameters are different from the ANN's number of inputs\n Please" \
                      "correct it and try again. \n Note: the number of ANN's inputs=", str(self.num_inputs)," while" \
                      "you give only ", str(num_given_variables),"parameters."
                exit()

            # intiating a dictionary to store number of instances of each variable
            var_dict = {}
            # var_dict is a dictionary, contains number of lists, one for each variable.
            # Each list contains the following fields:
            # [0. Group length (Number of values of the variable),
            # 1. the type of variable (0 for number, 1 for list, 2 for tuple, 3 for string),
            # 2. type of the repeated variable (True if float, False elsewhere),
            # 3. Cells occupied by each group (will be described below),
            # 4. Member Repetition (will be described below),
            # 5. Group Repetition (will be described below)]

            # determining the number of cases needed to be evaluated
            number_of_cases = 1
            for v, variable in enumerate(given_variables):
                if isinstance(variable, (float, int)):
                    var_dict[v] = [1, 0, False]
                    pass  # number_of_lines *= 1
                elif isinstance(variable, str):
                    var_dict[v] = [1, 3, False]
                elif isinstance(variable, list):
                    var_steps = steps_of_var(variable)
                    var_dict[v] = [var_steps[0], 1, var_steps[1]]
                    number_of_cases *= var_dict[v][0]
                elif isinstance(variable, tuple):
                    var_dict[v] = [len(variable), 2, False]
                    number_of_cases *= var_dict[v][0]

            # Determining the Group properties to make the distribution table
            # cells_occupied by each group is the product of Group lengths of all the groups next to the current group
            for v in range(num_given_variables):
                cells_occupied = 1
                for c in range(v, num_given_variables):
                    cells_occupied *= var_dict[c][0]
                var_dict[v].append(cells_occupied)
                # Member repetition is the number of repetitions each value of each group occurs
                # before taking the next value = cells_occupied / group_length
                var_dict[v].append(cells_occupied / var_dict[v][0])
                # Group repetition is the number of repititions each group is repeated until the end of file
                # Group repetition = number_of_cases /  cells_occupied
                var_dict[v].append(number_of_cases / cells_occupied)

            # Getting the values of each variable in a list
            variables_values = [[] for x in xrange(num_given_variables)]

            for v, variable in enumerate(given_variables):
                if var_dict[v][1] in [0, 3]:  # it is just one value either number or text
                    variables_values[v] = [variable]
                else:
                    if var_dict[v][1] == 1:  # it is a list
                        if var_dict[v][2]:  # The desired range is float not integer
                            for var_value in float_range(variable[0], variable[1], variable[2]):
                                variables_values[v].append(var_value)
                        else:  # The desired range is integer
                            for var_value in range(variable[0], variable[1] + variable[2], variable[2]):
                                variables_values[v].append(var_value)
                    elif var_dict[v][1] == 2:  # it is a tuple
                        for var_value in variable:
                            variables_values[v].append(var_value)

            # Filling the cases
            cases_list = [[] for x in xrange(number_of_cases)]


            for v in range(num_given_variables):
                case = 0
                for group_repetition in range(var_dict[v][5]):
                    for member in variables_values[v]:
                        for member_repetition in range(var_dict[v][4]):
                            cases_list[case].append(member)
                            case += 1
            # Storing the cases in a csv file
            file_name = self.new_folder_path + '\\' + 'NrCH_QueryData' + self.time_stamp[-4:] + '.csv'
            # cleaning previous file:
            open(file_name, "w").close()
            data_file_aq = open(file_name, "a")
            for line in cases_list:
                clean_line = str(line)
                clean_line = clean_line.replace('[', '')
                clean_line = clean_line.replace(']', '')
                clean_line = clean_line.replace("'", "")
                data_file_aq.writelines(clean_line + '\n')
            data_file_aq.close()

            pass
            self.data_file = file_name

        # creating a data object with all variables as before
        self.source_data = Data(self.data_file, num_outputs=self.num_inputs,
                                has_titles=self.data_file_has_titles,
                                has_briefs=self.data_file_has_brief_titles,
                                querying_data=True, parent_study=self,
                                data_style=self.data_style)
        # creating an ANN
        self.ann = NeuralNetwork(num_norm_inputs, num_hidden, num_norm_outputs,
                                 activation_functions=self.activation_functions, parent_study=self,
                                 categorical_extra_divisor=self.categorical_extra_divisor,
                                 hidden_layer_weights=self.temporary['weights_of_i_h'],
                                 hidden_layer_bias=self.temporary['bias_of_i_h'],
                                 output_layer_weights=self.temporary['weights_of_h_o'],
                                 output_layer_bias=self.temporary['bias_of_h_o'],
                                 learning_rate=self.learning_rate,
                                 out_var_bool=self.temporary['out_var_bool'])
        # reading the query inputs
        tmp = []
        with open(self.data_file, 'rb') as csvfile:
            spam_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in spam_reader:
                # tmp.append(', '.join(row))
                tmp.append(row)

        var_count = len(tmp[0])
        if var_count < self.num_inputs:
            print "ERROR, insufficient number of input variables\n ... At least " + str(self.num_inputs) \
                  + ' variables should be provided\n(If you provided more, only the first ' + str(self.num_inputs) \
                  + ' variables will be considered.)'
            exit()

        data_ok = []
        for line in tmp:
            lll = []
            for cell in line:
                lll.append(read_cell(cell))
            data_ok.append(lll[:self.num_inputs])
        # finding the type of each variable of the input data.
        input_types = ['Numeric'] * self.num_inputs
        for i in range(self.num_inputs):
            for d_row in data_ok:
                if isinstance(d_row[i], str):
                    input_types[i] = 'Categorical'
                    break



        # comparing original types vs the given query types
        for var_num in range(self.num_inputs):
            if input_types[var_num] != self.source_data.input_variables[var_num].data_type:
                print "ERROR, data type mismatch.\n The given input # " + str(var_num) + " is " + \
                      input_types[var_num] + " while it should be of " + \
                      self.source_data.input_variables[var_num].data_type + \
                      " type.\n This error may occur if you added data titles to the query data, which is wrong.\n" \
                      + "If you did so, please REMOVE ALL TITLES, & LEAVE DATA ONLY IN THE FILE."
                exit()
        results = [[[in_title for in_title in self.source_data.get_titles(source='inputs')],
                    [in_title for in_title in self.source_data.get_titles(source='outputs')]]]
        # reading data, line by line, and finding results
        # defining a list of error messages
        warning_values = []
        has_warnings = False
        for j, line in enumerate(data_ok):
            for i, cell in enumerate(line):
                var = self.source_data.input_variables[i]
                # check data limits and raise warning if data violates limits
                if var.data_type == 'Numeric':
                    if cell > var.max:
                        warning_values.append((j, i, cell, var.min, var.max, 'Exceeds maximum'))
                        has_warnings = True
                        # print "Warning! Value in line " + str(j) + " of variable " + str(i) + \
                        #       " is more than the maximum trained value of the data.\n This may result in errors."
                    if cell < var.min:
                        warning_values.append((j, i, cell, var.min, var.max, 'Below minimum'))
                        has_warnings = True
                        # print "Warning! Value in line " + str(j) + " of variable " + str(i) + \
                        #       " is less than the minimum trained value of the data.\n This may result in errors."
                else:  # Categoric
                    if cell.strip().upper() not in map(lambda x: x.strip().upper(),var.members):
                        print "Error! Value in line " + str(j) + " of variable " + str(i) + \
                              " is not a member of the trained data.\n Normalization is not possible"
                        print 'The provided value is  : ', str(cell)
                        print 'The trained values were: ', str(var.members)
                        exit()

                # now the line is OK,

            # normalizing query inputs
            norm_line = self.get_normalized_input_line(line)
            # run the ANN on each data line, getting normalized output
            outputs_temp = self.ann.get_predictions(norm_line)
            # de-normalize the outputs
            de_norm_out = self.get_de_normalized_output_line(outputs_temp)
            de_norm_out = map(lambda x: "{:.3f}".format(x) if not isinstance(x, str) else x, de_norm_out)
            results.append([line, de_norm_out])
        # logging errors if any
        if has_warnings:
            header_list = ['Line #', 'Variable #', 'Value', 'Minimum', 'Maximum', 'Error']
            formatted_table = prettytable.PrettyTable(header_list)
            for warning in warning_values:
                formatted_table.add_row(warning)

            head_0 = 'Found ' + str(len(warning_values)) + ' data lines containing values that are ' \
                                                       'outside the training boundaries.\n'
            head_1 = 'This may result in unreliable results'
            print "~" * len(head_0)
            print head_0
            print head_1
            print "~" * len(head_0)
            print formatted_table

        # print all to a file.
        out_file = self.data_file[:-4] + "_Results.csv"
        self.save_to_file(4, results, out_file)
        print "Number of processed data lines: " + str(len(data_ok))
        print "Outputs saved in file         : " + out_file
        print 'Analysis ended at             : '+ time.strftime('%Y-%m-%d %H:%M:%S')

        pass

    def perform_study(self):

        """
        Runs one of the Study modes
            'cross validation': to train some of the data while checking the errors of the validation dataset
                                if the error of the latter starts to increase instead of decrese,
                                the training will stop as it will be considered a n evidence of over fitting.
                                This is the Default purpose
            'full run': to run all the data as training set, no validation, and no testing
            'sequential validation': to run some of the data as training, some as validation,
                                and some as testing. the validation starts, then traing for maximum of double
                                the convergence epochs of the training set, the the testing set.
            'optimization': to do similar to the 'validation run', but before that it searched the
                                best structure and best activation functions. This is the slowest one.
        """

        def study_full_run(study):

            """
            to run all the data as training set, no validation, and no testing
            @param study: is the Study Class instance
            """
            study.ann = study.create_net()
            study.normalized_data = study.perform_data_partition(100)[0]

            # calculations
            error_list, stopping_epoch, correlation_coefficients, outputs_to_graph, time_elapsed = study.train_net()
            errors_collection = study.ann.calculate_total_error(study.normalized_data)
            relative_importance_100, relative_importance_negative = study.separate_relative_importance()
            ecl = errors_collection  # To rename the variable for easier writing
            matrix_of_sum_of_errors = [ecl[0], sum(ecl[2]), sum(map((lambda x: x ** 0.5), ecl[2]))]

            # printing to console
            study.print_to_console(correlation_coefficients, errors_collection,
                                   matrix_of_sum_of_errors, stopping_epoch, study.maximum_epochs)

            # Storing weights and biases
            study.save_to_file(0)  # self.store_network_weights(ann)

            # # Retraining
            # self.perform_retraining(matrix_of_sum_of_errors, training_sets)

            # output to file
            study.prepare_output_file(error_list, stopping_epoch, study.tolerance, correlation_coefficients,
                                      matrix_of_sum_of_errors, errors_collection,
                                      relative_importance_100, relative_importance_negative, clear_file_state=False)
            # GRAPHING
            graphing_data_collection = [outputs_to_graph]
            study.graph_results(error_list, graphing_data_collection)
            pass

        def study_validation_run(study):
            # preparing all
            """
            to run some of the data as training, some as validation,
                                and some as testing. This is the Default
            @param study: is the Study Class instance
            """
            study.ann = study.create_net()
            optional_errors = []
            partitioned_data = study.perform_data_partition(study.data_partition[0], study.data_partition[1])
            graphing_data_collection = []

            # validation
            # for first look, we will use the validation data before the training data
            study.normalized_data = partitioned_data[1]
            error_list, stopping_epoch, correlation_coefficients, outputs_to_graph, time_elapsed = \
                study.train_net(training_title='Validating selected network')
            # ecl = errors_collection
            ecl = study.ann.calculate_total_error(study.normalized_data)
            matrix_of_sum_of_errors = [ecl[0], sum(ecl[2]), sum(map((lambda x: x ** 0.5), ecl[2]))]
            graphing_data_collection.append(outputs_to_graph)
            # printing to console
            print "Validation data results (", str(len(partitioned_data[1])), ") data points"
            study.print_to_console(correlation_coefficients, ecl,
                                   matrix_of_sum_of_errors, stopping_epoch, study.maximum_epochs)

            optional_errors.append(error_list)

            # training
            study.normalized_data = partitioned_data[0]
            study.maximum_epochs = stopping_epoch * 2  # Added *2 to improve error
            error_list, stopping_epoch, correlation_coefficients, outputs_to_graph, time_elapsed = \
                study.train_net(training_title='Running Training Stage, for maximum epochs of ' + str(stopping_epoch))
            ecl = study.ann.calculate_total_error(study.normalized_data)
            matrix_of_sum_of_errors = [ecl[0], sum(ecl[2]), sum(map((lambda x: x ** 0.5), ecl[2]))]
            # Now find the relative importance
            relative_importance_100, relative_importance_negative = study.separate_relative_importance()
            # printing to console
            print "Training data results (", str(len(partitioned_data[0])), ") data points"
            study.print_to_console(correlation_coefficients, ecl,
                                   matrix_of_sum_of_errors, stopping_epoch, study.maximum_epochs)
            graphing_data_collection.append(outputs_to_graph)
            optional_errors.append(error_list)

            # testing
            study.normalized_data = partitioned_data[2]
            error_list, stopping_epoch, correlation_coefficients, outputs_to_graph, time_elapsed = \
                study.train_net(training_title='Running Testing Stage...')
            ecl = study.ann.calculate_total_error(study.normalized_data)
            # # Now find the relative importance
            # relative_importance_100, relative_importance_negative = self.separate_relative_importance()
            # matrix_of_sum_of_errors = [ecl[0], sum(ecl[2]), sum(map((lambda x: x ** 0.5), ecl[2]))]
            # printing to console
            print "Testing data results (", str(len(partitioned_data[2])), ") data points"
            study.print_to_console(correlation_coefficients, ecl,
                                   matrix_of_sum_of_errors, stopping_epoch, study.maximum_epochs)
            # optional_errors.append(copy.deepcopy(error_list))
            optional_errors.append(error_list)
            graphing_data_collection.append(outputs_to_graph)

            # Storing weights and biases
            study.save_to_file(0)  # self.store_network_weights(ann)

            # output to file
            study.prepare_output_file(error_list, stopping_epoch, study.tolerance, correlation_coefficients,
                                      matrix_of_sum_of_errors, ecl,
                                      relative_importance_100, relative_importance_negative, clear_file_state=False)
            # GRAPHING
            # self.graph_results(error_list, outputs_to_graph, optional_errors)
            # to copy only the outputs of the partitioned data to separate vld, tst, and trn points
            partitioned_data_outs = [[], [], []]
            for i, dta in enumerate(partitioned_data):
                for lne in dta:
                    partitioned_data_outs[i].append(lne[1])
            study.network_save()
            study.graph_results(error_list, graphing_data_collection, optional_errors, partitioned_data_outs,
                                study.start_time)

            pass

        def study_cross_validation(study):
            # preparing all
            """
            to run some of the data as training, some as validation,
                                and some as testing. This is the Default
            @param study: is the Study Class instance
            """
            study.ann = study.create_net()
            optional_errors = []
            partitioned_data = study.perform_data_partition(study.data_partition[0], study.data_partition[1])
            graphing_data_collection = []

            # Selecting the best random numbers from 12 alternatives
            validation_error = float('inf')
            study.normalized_data = partitioned_data[1]
            header_list = ['run #', 'Error', 'time(s)']
            for cor in range(self.num_outputs_normalized):
                header_list.append('r Out#' + str(cor))
            header_list.append('Selected')
            formatted_table = prettytable.PrettyTable(header_list)

            for alter in range(self.weights_trials):
                results = []
                temp_ann = study.create_net()
                title = 'Finding The best random weights of the network, trial %d of %d' % (alter, self.weights_trials)
                error_list, stopping_epoch, correlation_coefficients, outputs_to_graph, time_elapsed = \
                    study.train_net(training_title=title, temp_maximum_epochs=self.validation_epochs,
                                    other_ann=temp_ann)
                formatted = "{:.6f}"
                results.append(alter)
                results.append(formatted.format((error_list[-1])))
                # results.append(time.strftime("%b %d %Y %H:%M:%S", (datetime(1970,1,1) +
                #                                                    timedelta(seconds=time_elapsed)).time()))
                results.append(str(timedelta(seconds=time_elapsed))[:-3])
                for cor in correlation_coefficients:
                    results.append(formatted.format((cor)))
                if error_list[-1] < validation_error:
                    validation_error = error_list[-1]
                    study.ann = temp_ann
                    results.append('***')
                else:
                    results.append('.')

                formatted_table.add_row(results)
            print formatted_table

            # training
            # for the cross validation, we will start by the training data
            study.normalized_data = partitioned_data[0]
            error_list, stopping_epoch, correlation_coefficients, outputs_to_graph, time_elapsed, cv_error = \
                study.train_net(training_title='Training and validating selected network',
                                cross_validation=True, validation_data_set=partitioned_data[1])

            # ecl = errors_collection
            ecl = study.ann.calculate_total_error(study.normalized_data)
            matrix_of_sum_of_errors = [ecl[0], sum(ecl[2]), sum(map((lambda x: x ** 0.5), ecl[2]))]
            graphing_data_collection.append(outputs_to_graph)
            # printing to console
            print "Training data results (", str(len(partitioned_data[0])), ") data points"
            study.print_to_console(correlation_coefficients, ecl,
                                   matrix_of_sum_of_errors, stopping_epoch, study.maximum_epochs)

            optional_errors.append(cv_error[1])
            optional_errors.append(error_list)

            # Now find the relative importance
            relative_importance_100, relative_importance_negative = study.separate_relative_importance()

            # testing
            study.normalized_data = partitioned_data[2]
            error_list, stopping_epoch, correlation_coefficients, outputs_to_graph, time_elapsed = \
                study.train_net(training_title='Running Testing Stage...')
            ecl = study.ann.calculate_total_error(study.normalized_data)

            # printing to console
            print "Testing data results (", str(len(partitioned_data[2])), ") data points"
            study.print_to_console(correlation_coefficients, ecl,
                                   matrix_of_sum_of_errors, stopping_epoch, study.maximum_epochs)
            # optional_errors.append(copy.deepcopy(error_list))
            optional_errors.append(error_list)
            graphing_data_collection.append(outputs_to_graph)

            # Storing weights and biases
            study.save_to_file(0)  # self.store_network_weights(ann)

            # output to file
            special_error_list = []
            for x in range(3):
                special_error_list.extend(optional_errors[x])

            study.prepare_output_file(special_error_list, stopping_epoch, study.tolerance, correlation_coefficients,
                                      matrix_of_sum_of_errors, ecl,
                                      relative_importance_100, relative_importance_negative, clear_file_state=False)
            # GRAPHING
            # self.graph_results(error_list, outputs_to_graph, optional_errors)
            # to copy only the outputs of the partitioned data to separate vld, tst, and trn points
            partitioned_data_outs = [[], [], []]
            for i, dta in enumerate(partitioned_data):
                for lne in dta:
                    partitioned_data_outs[i].append(lne[1])
            study.network_save()
            study.graph_results(error_list, graphing_data_collection, optional_errors, partitioned_data_outs,
                                study.start_time)

            pass

        def study_optimization_run(study):

            """
            to run some of the data as training, some as validation, and some as testing. This is the Default
            but before that it searched the best structure and best activation functions. This is the slowest one.
            @param study: is the Study Class instance
            """

            def select_best_alternative(training_results, evaluation_weights=(10., 5., 70., 15.)):
                """
                select_best_alternative of the structure
                @param training_results: the cost of each alternative
                @param evaluation_weights: the evaluation weights as tuple of percentages:
                        Deafults (10% for epochs, 5% for number of neurons, 70% for cost, 15% for time)
                @return: tuple (the optimum ANN Class instance, and its score)
                """
                alternatives = []
                for res in training_results:
                    alternatives.append(res[:4])
                scores = []

                for alter in alternatives:
                    print alter

                alternatives_tr = transpose_matrix(alternatives)
                var_stats = {}
                maped_matrix = []
                # Normalizing through minimax
                for i, v in enumerate(alternatives_tr):
                    if not (isinstance(v[0], tuple) or isinstance(v[0], list)):
                        min_v = float(min(v))
                        var_stats[i] = (min_v, float(max(v) - min_v))
                        if var_stats[i][1] == 0:
                            maped_matrix.append(map(lambda x: evaluation_weights[i], v))
                        else:
                            maped_matrix.append(
                                map(lambda x: evaluation_weights[i] * (1 - (x - var_stats[i][0]) / var_stats[i][1]),
                                    v))
                    else:  # if it is not a number, so put it as only the weight
                        maped_matrix.append([evaluation_weights[i] for x in range(len(v))])
                alternatives = transpose_matrix(maped_matrix)
                for i, alt in enumerate(alternatives):
                    scores.append((sum(alt), i))
                scores.sort(key=lambda x: x[0], reverse=True)
                # for scr in scores: print scr
                return training_results[scores[0][1]][4], scores[0][1]

            optional_errors = []
            graphing_data_collection = []
            partitioned_data = study.perform_data_partition(study.data_partition[0], study.data_partition[1])
            default_structure = study.structure
            num_hidden_range = range(int(default_structure[1] * study.layer_size_range[0]),
                                     int(default_structure[1] * study.layer_size_range[1]),
                                     int(study.layer_size_range[2]))  # [3, 5, 8, 12, 20] #
            # validation
            # for first look, we will use the validation data before the training data
            study.normalized_data = partitioned_data[1]
            training_results = []
            for hidden_layer in num_hidden_range:
                temp_ann = study.create_net(structure=(default_structure[0], hidden_layer, default_structure[2]))

                error_list, stopping_epoch, correlation_coefficients, outputs_to_graph, time_elapsed = \
                    study.train_net(other_ann=temp_ann, temp_maximum_epochs=study.validation_epochs,
                                    training_title='Finding hidden neurons, testing for # ' + str(hidden_layer))
                # ecl = errors_collection
                ecl = temp_ann.calculate_total_error(study.normalized_data)
                # sometimes the last value in the error list is not the least valued error
                min_error = min(ecl[0], min(error_list))
                training_results.append([hidden_layer, stopping_epoch, min_error, time_elapsed, temp_ann,
                                         error_list])  # ,
                # correlation_coefficients, outputs_to_graph, error_list, ecl])
            # Now sorting the tested structures
            # training_results.sort(key=lambda x: x[2])
            # the best structure is the first in the list
            best_structure_net = select_best_alternative(training_results)  # training_results[0]
            hidden_final = training_results[best_structure_net[1]][0]
            validation_error_list = training_results[best_structure_net[1]][5]
            # setting the network to the good structure
            training_results = []
            functions_final = (0, 0)
            if study.find_activation_function:
                for func_hidden in ACTIVATION_FUNCTIONS.keys():
                    for func_out in ACTIVATION_FUNCTIONS.keys():
                        temp_ann = study.create_net(structure=(default_structure[0],
                                                               hidden_final,
                                                               default_structure[2]),
                                                    activation_functions=(func_hidden, func_out))
                        error_list, stopping_epoch, correlation_coefficients, outputs_to_graph, time_elapsed = \
                            study.train_net(other_ann=temp_ann, temp_maximum_epochs=study.validation_epochs,
                                            training_title='Finding activation functions, searching for: ' +
                                                          ACTIVATION_FUNCTIONS[func_hidden] + " - " +
                                                          ACTIVATION_FUNCTIONS[func_out])
                        # ecl = errors_collection
                        ecl = temp_ann.calculate_total_error(study.normalized_data)
                        # sometimes the last value in the error list is not the least valued error
                        min_error = min(ecl[0], min(error_list))
                        training_results.append([(func_hidden, func_out),
                                                 stopping_epoch, min_error, time_elapsed, temp_ann])
                for res in training_results:
                    print ACTIVATION_FUNCTIONS[res[0][0]], ACTIVATION_FUNCTIONS[res[0][1]], res[1], res[2], \
                        res[3]
                best_structure_net = select_best_alternative(training_results)
                functions_final = training_results[best_structure_net[1]][0]
                print 'Selected functions: ', \
                    ACTIVATION_FUNCTIONS[functions_final[0]], \
                    ACTIVATION_FUNCTIONS[functions_final[0]]

            if not study.refresh_weights_after_determining_structure:
                study.ann = best_structure_net[0]
            else:
                if study.find_activation_function:
                    study.ann = study.create_net(structure=(default_structure[0], hidden_final, default_structure[2]),
                                                 activation_functions=functions_final)
                else:
                    study.ann = study.create_net(structure=(default_structure[0], hidden_final, default_structure[2]))

                    # best_structure = training_results [best_structure_net[1]]

            error_list, stopping_epoch, correlation_coefficients, outputs_to_graph, time_elapsed = \
                study.train_net(training_title='Validating selected network with hidden neurons= ' + str(hidden_final))

            ecl = study.ann.calculate_total_error(study.normalized_data)
            matrix_of_sum_of_errors = [ecl[0], sum(ecl[2]), sum(map((lambda x: x ** 0.5), ecl[2]))]
            graphing_data_collection.append(outputs_to_graph)
            # printing to console
            print "Validation data results (", str(len(partitioned_data[1])), ") data points"
            study.print_to_console(correlation_coefficients, ecl,
                                   matrix_of_sum_of_errors, stopping_epoch, study.maximum_epochs)
            validation_error_list.extend(error_list)
            optional_errors.append(validation_error_list)

            # training
            study.normalized_data = partitioned_data[0]
            study.maximum_epochs = stopping_epoch * 3
            error_list, stopping_epoch, correlation_coefficients, outputs_to_graph, time_elapsed = \
                study.train_net(training_title='Running Training Stage, for maximum epochs of ' + str(stopping_epoch))

            ecl = study.ann.calculate_total_error(study.normalized_data)
            matrix_of_sum_of_errors = [ecl[0], sum(ecl[2]), sum(map((lambda x: x ** 0.5), ecl[2]))]

            # Now find the relative importance
            relative_importance_100, relative_importance_negative = study.separate_relative_importance()

            # printing to console
            print "Training data results (", str(len(partitioned_data[0])), ") data points"
            study.print_to_console(correlation_coefficients, ecl,
                                   matrix_of_sum_of_errors, stopping_epoch, study.maximum_epochs)
            graphing_data_collection.append(outputs_to_graph)
            optional_errors.append(error_list)

            # testing
            study.normalized_data = partitioned_data[2]
            error_list, stopping_epoch, correlation_coefficients, outputs_to_graph, time_elapsed = \
                study.train_net(training_title='Running Testing Stage...')
            ecl = study.ann.calculate_total_error(study.normalized_data)
            matrix_of_sum_of_errors = [ecl[0], sum(ecl[2]), sum(map((lambda x: x ** 0.5), ecl[2]))]
            # printing to console
            print "Testing data results (", str(len(partitioned_data[2])), ") data points"
            study.print_to_console(correlation_coefficients, ecl,
                                   matrix_of_sum_of_errors, stopping_epoch, study.maximum_epochs)
            # optional_errors.append(copy.deepcopy(error_list))
            optional_errors.append(error_list)
            graphing_data_collection.append(outputs_to_graph)

            # Storing weights and biases
            study.save_to_file(0)  # self.store_network_weights(ann)

            # output to file
            study.prepare_output_file(error_list, stopping_epoch, study.tolerance, correlation_coefficients,
                                      matrix_of_sum_of_errors, ecl,
                                      relative_importance_100, relative_importance_negative, clear_file_state=False)
            # GRAPHING
            # self.graph_results(error_list, outputs_to_graph, optional_errors)
            # to copy only the outputs of the partitioned data to separate vld, tst, and trn points
            partitioned_data_outs = [[], [], []]
            for i, dta in enumerate(partitioned_data):
                for lne in dta:
                    partitioned_data_outs[i].append(lne[1])

            study.network_save()
            study.graph_results(error_list, graphing_data_collection, optional_errors, partitioned_data_outs,
                                study.start_time)
            pass

        if self.purpose in ['fr', 'full run']:
            study_full_run(self)
        elif self.purpose in ['sv', 'sequential validation']:
            study_validation_run(self)
        elif self.purpose in ['op', 'optimization']:
            study_optimization_run(self)
        elif self.purpose in ['cv', 'cross validation']:
            study_cross_validation(self)
        else:
            print 'Error\nStudy purpose not recognized'
            exit()

        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'The study  from source: (' + self.source_file_name + ') has been finished successfully,\n' \
                                                                   'all the outputs were stored in the folder: (' \
              + self.directory_name + ').'
        print "\nElapsed time throughout the study: ", elapsed_time(self.start_time, time.time())

        pass

    def perform_data_partition(self, amount_training=60, amount_validation=25):  # , amount_testing=15):
        """
        Partition the data to %Training, %Validatain, And %Testing
        @param amount_training: the percent of training data
        @param amount_validation:the percent of validation data
        @return: a tuple (trn, vld, tst)  of random data lines
        """
        n = len(self.main_normalized_data)
        data = copy.deepcopy(self.main_normalized_data)
        random.shuffle(data)
        trn = data[:int(n * amount_training / 100)]
        vld = data[int(n * amount_training / 100):int(n * amount_training / 100) + int(n * amount_validation / 100)]
        tst = data[int(n * amount_training / 100) + int(n * amount_validation / 100):]

        # In 22/11/16 I commented the sorting block to speedup processin, as I realised it improves nothing
        # # sorting all sub lists for better results
        # sorting_vars = range(len(data[0][0]))
        # trn = sorted(trn, key=lambda x: [x[0][f] for f in sorting_vars])
        # # sorted(trn, key=itemgetter(sorting_tuple))
        # vld = sorted(vld, key=lambda x: [x[0][f] for f in sorting_vars])
        # tst = sorted(tst, key=lambda x: [x[0][f] for f in sorting_vars])

        return trn, vld, tst

    def create_net(self, structure=None, activation_functions=None):  # , network_weights=None):
        """
        Nesr network creation
        @param structure: the suggested structure to create the ANN, if None, then Clone the current ANN of the study
        @param activation_functions: suggested activation functions of the ANN, if None, then Clone from the current
                                        study's ANN functions
        @return: An ANN Class object.
        """
        if structure is None:
            structure = self.structure
        if activation_functions is None:
            activation_functions = self.activation_functions
        num_inputs, num_hidden, num_outputs = structure
        nn = NeuralNetwork(num_inputs, num_hidden, num_outputs,
                           activation_functions=activation_functions, parent_study=self,
                           categorical_extra_divisor=self.categorical_extra_divisor)
        return nn  # train

    def separate_relative_importance(self):  # , method='Advance Corrected'):
        # ann = self.ann
        """
        separates results of relative importance to Only Positive, and Positve&Negative
        @param method: the relative importance calculation method, Default is 'Advance Corrected'
        @return: tuple of (Positive RI, and Positve&Negative RI) lists
        """
        # method = self.relative_importance_method
        ri = self.get_relative_importance()
        re100 = []
        re_negative = []
        for out in ri:
            re100.append(out[0])
            re_negative.append(out[1])
        return re100, re_negative

    def get_variables_info(self, style='bool'):
        """
        returns information about the variables depending on the style
        @param style: one of 4 styles:
                    'str': Text information of variables which is a list of the data types of each variable in the form:
                            [nI0, cI1, nI2, nI3, cO0, nO1, nO2] where n for numeric, c for categoric
                            I for input, O for output;  numbering starts from 0 for either inputs or outputs
                    'bool': a list of True / False, depending on the variable type NUmeric or Categoric
                    'loc': a list of 2 lists (input and output), contains numbers of original variables
                            (before normalization) on a normalized length:
                            example [[0, 1, 1, 2, 3, 3, 3],[0, 0, 0, 1, 2, 2]] if the input vars 1, 3 are categoric
                            with 2 and 3 members respectively, and the output variables 0, 2 are categoric with 3, 2
                            members respectively. The other variables are Numeric
                    'avg': returns the mean row of the data in its normalized form

        @return: a list as described above.
        """
        num_inputs = self.num_inputs_normalized
        # means = self.source_data.get_mean_row()
        if style == 'str':
            return self.source_data.data_style
        elif style == 'bool':
            return [self.source_data.get_data_style(required_style='binary')[:num_inputs],
                    self.source_data.get_data_style(required_style='binary')[num_inputs:]]
        elif style == 'loc':
            return self.source_data.get_data_style(required_style='vars')[:num_inputs]
        elif style == 'avg':
            return self.source_data.get_mean_row()
        elif style == 'original boolean':
            return self.source_data.basic_data_style

    def get_relative_importance(self):  # , method='Advance Corrected'):
        """
        Calculates the relative importance of each input on each output.
        @return: final_relative_importance list
        """
        ann = self.ann
        # param method: the metod of calculation 'Garson', 'Milne', 'Nesr', 'Advance', 'Advance Corrected'
        method = self.relative_importance_method

        inputs_weights = transpose_matrix(ann.get_weights(I_H)[0])
        outputs_weights = transpose_matrix(ann.get_weights(H_O)[0], False)
        bias_i_h = transpose_matrix(ann.get_weights(I_H)[1])
        bias_h_o = transpose_matrix(ann.get_weights(H_O)[1], False)

        final_relative_importance = []
        n_hidden = len(inputs_weights[0])
        n_inputs = len(inputs_weights)
        var_types_bool = self.get_variables_info('bool')
        var_locations = self.get_variables_info('loc')

        # auxiliary function
        def nesr_norm(input_x):
            """
            Calculates a normalized list
            @param input_x: a numeric list
            @return:normalized equivalent list
            """
            total = sum(input_x)
            out = []
            for num in input_x:
                out.append(num / total)
            return out

        def nesr_min(input_list):
            """
            Calculates the special minimum value which is the normal minimum if positive, otherwise
            it returns double the absolute value. THis is useful for converting all Relative Contribution
            values to positive
            @param input_list: a list of values
            @return: the modified minimum
            """
            n_min = min(input_list)
            return n_min if n_min > 0 else 2 * abs(n_min)

        def consolidate_categorical_matrix(matrix, var_map, var_bool):
            """
            to consolidate the categoric results into single values as the numeric ones.
            If an input cat variable has 4 members, then it has 4 values per output variable, we want all
            these values to be cobined in one equivalent value.

            @param matrix: the Relative Contribution matrix in its normalized form
            @param var_map: Map of variables, two tuples, the first for inputs and the second for outputs
            @param var_bool: Boolean list, True or False form Numeric and Categoric variables
            @return: consolidated Relative Contribution matrix in original data form.
            """
            mat = []
            # first, finish all the inputs if categorical
            for i, out in enumerate(matrix):
                tmp = []
                finished_var = []
                for j, cell in enumerate(out):
                    if var_map[0][j] not in finished_var:
                        if var_bool[0][j]:  # if true then it is numeric
                            tmp.append(cell)
                            finished_var.append(var_map[0][j])
                        else:
                            # find how many elments should be consolidated by the category
                            rep = var_map[0].count(var_map[0][j])
                            elements = []
                            for r in range(j, j + rep):
                                elements.append(matrix[i][r])

                            tmp.append(rms_mean(elements) * dominant_sign(elements))
                            finished_var.append(var_map[0][j])
                mat.append(tmp)
            # second, finish all the outputs if categorical
            fin = []
            finished_var = []
            for i, out in enumerate(mat):
                if var_map[1][i] not in finished_var:
                    if var_bool[1][i]:  # if true then it is numeric
                        tmp = []
                        for j, cell in enumerate(out):
                            tmp.append(cell)
                        finished_var.append(var_map[1][i])
                    else:
                        # find how many elments should be consolidated by the category
                        rep = var_map[1].count(var_map[1][i])
                        tmp = []
                        for j, cell in enumerate(out):
                            elements = []
                            for r in range(i, i + rep):
                                elements.append(mat[r][j])
                            tmp.append(rms_mean(elements) * dominant_sign(elements))
                        finished_var.append(var_map[1][i])
                    fin.append(tmp)

            return fin

        if method.lower() in ['garson', 'g']:
            # Garson (1991) Method

            for k, output in enumerate(outputs_weights):
                c = [[0 * j for j in range(len(inputs_weights[0]))] for dummy_i in range(len(inputs_weights))]
                s = [0] * len(inputs_weights[0])
                f = [0] * len(inputs_weights)
                r = [[0 * j for j in range(len(inputs_weights[0]))] for dummy_i in range(len(inputs_weights))]

                for i, neuron in enumerate(inputs_weights):
                    for j, weight in enumerate(neuron):
                        c[i][j] = weight * output[j]
                # print_matrix(c)
                for i, neuron in enumerate(inputs_weights):
                    for j, weight in enumerate(neuron):
                        s[j] += abs(c[i][j])  # abs(c[i][j]) / temp_sum
                # print_matrix(s)
                for i, neuron in enumerate(inputs_weights):
                    for j, weight in enumerate(neuron):
                        r[i][j] = abs(c[i][j]) / s[j]
                # print_matrix(r)
                for i, neuron in enumerate(inputs_weights):
                    for j, weight in enumerate(neuron):
                        f[i] += abs(r[i][j])  # abs(c[i][j]) / temp_sum
                # print_matrix(f)
                temp_sum = sum(f)
                for i, neuron in enumerate(inputs_weights):
                    f[i] = f[i] / temp_sum * 100
                # print_matrix(f)
                final_relative_importance.append((f, f))

        elif method.lower() in ['milne', 'm']:  # or method is None:
            # Milne (1995) Method
            effects_matrix = []
            positive_matrix = []
            for k, output in enumerate(outputs_weights):
                a = [[0 * j for j in range(len(inputs_weights[0]))] for dummy_i in range(len(inputs_weights))]
                s = [0] * len(inputs_weights[0])
                o = [0] * len(inputs_weights[0])
                n = [[0 * j for j in range(len(inputs_weights[0]))] for dummy_i in range(len(inputs_weights))]
                ns = [0] * len(inputs_weights)
                f = [0] * len(inputs_weights)
                for i, neuron in enumerate(inputs_weights):
                    for j, weight in enumerate(neuron):
                        a[i][j] = abs(weight)  # * output[j]
                for j in range(len((inputs_weights[0]))):
                    o[j] = abs(output[j])
                # print_matrix(c)
                for i, neuron in enumerate(inputs_weights):
                    for j, weight in enumerate(neuron):
                        s[j] += a[i][j]
                for i, neuron in enumerate(inputs_weights):
                    for j, weight in enumerate(neuron):
                        n[i][j] = weight * output[j] / s[j]
                        # no need for matrix 'a' now, so assign denominators to it
                        a[i][j] = weight * weight / s[j]
                for i, neuron in enumerate(inputs_weights):
                    for j, weight in enumerate(neuron):
                        ns[i] += n[i][j]
                den_sum = sum(sum(x) for x in a)
                # Now use 'ns' for final summation
                for i in range(len(inputs_weights)):
                    ns[i] /= den_sum
                # Nesr addition to make all results positive
                min_milna = min(ns)
                nesr_add = 2 * abs(min_milna) if min_milna < 0 else 0
                for i in range(len(inputs_weights)):
                    f[i] = ns[i] + nesr_add
                den_sum = sum(f)
                for i in range(len(inputs_weights)):
                    f[i] /= den_sum

                # final_relative_importance.append((f, ns))
                effects_matrix.append(ns)
                positive_matrix.append(f)
            final_matrix_e = consolidate_categorical_matrix(effects_matrix, var_locations, var_types_bool)
            final_matrix_e = [[100 * x / sum((n if n > 0 else (abs(n))) for n in r) for x in r] for r in final_matrix_e]
            # positive_matrix = consolidate_categorical_matrix(positive_matrix, var_locations, var_types_bool)
            # pos_matrix = [[100 * x / sum((n if n > 0 else (abs(n))) for n in r) for x in r] for r in positive_matrix]
            # Copy the same matrix but as possitive values
            positive_matrix = copy.deepcopy(final_matrix_e)
            # smart way to convert to absolute values.
            pos_matrix = [map(abs, mynum) for mynum in positive_matrix]
            for i, outs in enumerate(pos_matrix):
                final_relative_importance.append((outs, final_matrix_e[i]))
            # return final_relative_importance

        elif method.lower() in ['nesr', 'elnesr', 'el-nesr', 'n']:  # or method is None:
            n_inputs += 1  # This method takes the bias into account, we consider it as another input
            # Modified Milne (1995) Method by Dr. Mohammad Elnesr
            effects_matrix = []
            positive_matrix = []
            for k, output in enumerate(outputs_weights):
                a = [[0 * j for j in range(n_hidden)] for dummy_i in range(n_inputs)]
                s = [0] * n_hidden
                o = [0] * n_hidden
                n = [[0 * j for j in range(n_hidden)] for dummy_i in range(n_inputs)]
                ns = [0] * (n_inputs - 1)
                f = [0] * (n_inputs - 1)
                for i, neuron in enumerate(inputs_weights):
                    for j, weight in enumerate(neuron):
                        a[i][j] = abs(weight)  # * output[j]
                for j, weight in enumerate(bias_i_h):
                    a[n_inputs - 1][j] = abs(weight)
                for j in range(n_hidden):
                    o[j] = abs(output[j])
                # print_matrix(c)
                for i in range(len(a)):
                    for j in range(len(a[0])):
                        s[j] += a[i][j]
                for i, neuron in enumerate(inputs_weights):
                    for j, weight in enumerate(neuron):
                        n[i][j] = weight * output[j] / s[j]
                        # no need for matrix 'a' now, so assign denominators to it
                        a[i][j] = abs(n[i][j])
                for j, weight in enumerate(bias_i_h):
                    n[n_inputs - 1][j] = weight * output[j] / s[j]
                    a[n_inputs - 1][j] = abs(n[n_inputs - 1][j])

                # Finding the sum of denominators
                den_sum = sum(sum(x) for x in a)

                # Finding the sum of numerators
                for i in range(len(n) - 1):
                    # initialize each sum by the output bias value
                    ns[i] = bias_h_o[k]
                    for j in range(len(n[0])):
                        ns[i] += n[i][j]

                # Now use 'ns' for final summation
                # sum_milne =0
                for i in range(n_inputs - 1):
                    ns[i] /= den_sum
                    # sum_milne += ns[i]

                # Nesr addition to make all results positive
                min_milna = min(ns)
                nesr_add = 2 * abs(min_milna) if min_milna < 0 else 0
                for i in range(n_inputs - 1):
                    f[i] = ns[i] + nesr_add
                # to let all values of a list summing to 1.0
                f = nesr_norm(f)
                ns = nesr_norm(ns)
                effects_matrix.append(ns)
                positive_matrix.append(f)
            final_matrix_e = consolidate_categorical_matrix(effects_matrix, var_locations, var_types_bool)
            final_matrix_e = [[100 * x / sum((n if n > 0 else (abs(n))) for n in r) for x in r] for r in final_matrix_e]
            # positive_matrix = consolidate_categorical_matrix(positive_matrix, var_locations, var_types_bool)
            # pos_matrix = [[100 * x / sum((n if n > 0 else (abs(n))) for n in r) for x in r] for r in positive_matrix]
            # Copy the same matrix but as possitive values
            positive_matrix = copy.deepcopy(final_matrix_e)
            # smart way to convert to absolute values.
            pos_matrix = [map(abs, mynum) for mynum in positive_matrix]
            for i, outs in enumerate(pos_matrix):
                final_relative_importance.append((outs, final_matrix_e[i]))

                # final_relative_importance.append((f, ns))

        elif method == 'Advance':

            means = self.source_data.get_mean_row()
            # var_types_str = self.source_data.data_style
            var_types_bool = self.get_variables_info('bool')
            var_locations = self.get_variables_info('loc')

            normalized_mean = []
            for i, mean in enumerate(means):
                if isinstance(mean, str):
                    normalized_mean.append(0)  # for categoric, it is either 0 or 1
                else:
                    normalized_mean.append(self.source_data.input_variables[var_locations[0][i]].
                                           single_mini_max(mean))
            # Base case
            outputs_base = self.ann.get_predictions(normalized_mean)
            effects_matrix = []
            for i, var in enumerate(normalized_mean):
                changed_input = copy.deepcopy(normalized_mean)  # to copy a list
                if var_types_bool[0][i] is True:  # it is numeric
                    changed_input[i] = 1.1 * var
                else:  # the variable is categorized
                    changed_input[i] = 1
                outputs_temp = self.ann.get_predictions(changed_input)
                effects_matrix.append(outputs_temp)

            for i, input_line in enumerate(effects_matrix):
                for o, out in enumerate(input_line):
                    effects_matrix[i][o] = (effects_matrix[i][o] - outputs_base[o]) / outputs_base[o]
                    if var_types_bool[0][i] is True:  # it is numeric, then divide by 10%
                        effects_matrix[i][o] /= 0.1

            nesr_minimum = [nesr_min(lll) for lll in transpose_matrix(effects_matrix)]
            # print nesr_minimum

            positive_matrix = copy.deepcopy(effects_matrix)
            for i, input_line in enumerate(positive_matrix):
                for o, out in enumerate(input_line):
                    positive_matrix[i][o] += nesr_minimum[o]

            effects_matrix = transpose_matrix(effects_matrix)
            neg_sums = [sum(ooo) for ooo in effects_matrix]
            effects_matrix = [[x / neg_sums[i] for x in r] for i, r in enumerate(effects_matrix)]

            positive_matrix = transpose_matrix(positive_matrix)
            positive_matrix = consolidate_categorical_matrix(positive_matrix, var_locations, var_types_bool)

            # positive_matrix = swap_matrix(positive_matrix)
            var_sums = [sum(lll) for lll in positive_matrix]
            for i, input_line in enumerate(positive_matrix):
                for o, out in enumerate(input_line):
                    positive_matrix[i][o] = positive_matrix[i][o] / var_sums[i] * 100

            final_relative_importance = []

            final_matrix_e = consolidate_categorical_matrix(effects_matrix, var_locations, var_types_bool)
            # convert to percent
            # 1st trial, ... fail
            # effects_matrix = [[100 * x / sum(abs(n) for n in r) for x in r] for r in effects_matrix]
            # 1st trial, ... success
            final_matrix_e = [[100 * x / sum((n if n > 0 else (abs(n))) for n in r) for x in r] for r in
                              final_matrix_e]

            for i, outs in enumerate(positive_matrix):
                final_relative_importance.append((outs, final_matrix_e[i]))

        elif method.lower() in ['advance corrected', 'AdvCorr', 'ac']:

            means = self.source_data.get_mean_row()
            # var_types_str = self.source_data.data_style

            normalized_mean = []
            for i, mean in enumerate(means):
                if isinstance(mean, str):
                    normalized_mean.append(0)  # for categoric, it is either 0 or 1
                else:
                    normalized_mean.append(self.source_data.input_variables[var_locations[0][i]].
                                           single_mini_max(mean))
            # Base case
            outputs_base = self.ann.get_predictions(normalized_mean)
            effects_matrix = []
            for i, var in enumerate(normalized_mean):
                changed_input = copy.deepcopy(normalized_mean)  # to copy a list
                if var_types_bool[0][i] is True:  # it is numeric
                    changed_input[i] = 1.1 * var
                else:  # the variable is categorized
                    changed_input[i] = 1
                outputs_temp = self.ann.get_predictions(changed_input)
                effects_matrix.append(outputs_temp)

            for i, input_line in enumerate(effects_matrix):
                for o, out in enumerate(input_line):
                    # if outputs_base[o] != 0:
                    effects_matrix[i][o] = (effects_matrix[i][o] - outputs_base[o]) / outputs_base[o]
                    if var_types_bool[0][i] is True:  # it is numeric, then divide by 10%
                        effects_matrix[i][o] /= 0.1

            effects_matrix = transpose_matrix(effects_matrix)
            neg_sums = [sum(ooo) for ooo in effects_matrix]
            effects_matrix = [[x / neg_sums[i] for x in r] for i, r in enumerate(effects_matrix)]

            final_relative_importance = []

            final_matrix_e = consolidate_categorical_matrix(effects_matrix, var_locations, var_types_bool)

            final_matrix_e = [[100 * x / sum((n if n > 0 else (abs(n))) for n in r) for x in r] for r in
                              final_matrix_e]
            # Copy the same matrix but as possitive values
            positive_matrix = copy.deepcopy(final_matrix_e)
            # smart way to convert to absolute values.
            positive_matrix = [map(abs, mynum) for mynum in positive_matrix]

            for i, outs in enumerate(positive_matrix):
                final_relative_importance.append((outs, final_matrix_e[i]))

        return final_relative_importance

    @staticmethod
    def adjusted_line_thickness(matrix, max_absolute_value, max_thickness=6):
        """
        A function to make proportional line thicknesses of the ANN diagram,
        so that the maximum thickness is about 6 points, and all other lines of weights and biases
        are reduced accordingly
        @param matrix: of all weights or biases of a layer
        @param max_absolute_value: of weights and biases of the network
        @param max_thickness: is the maximum allowed line thickness in the diagram
        @return: a matrix normalized line thicknesses
        """
        mat_np = np.asarray(matrix)

        return mat_np * max_thickness / max_absolute_value

    @staticmethod
    def max_absolute_value(matrices):
        """
        Returns the maximum absolute value of a matrix
        @param matrices: an iterable
        @return: the maximum absolute value of the matrix
        """
        mxv = 0
        for mat in matrices:
            mat_np = np.asarray(mat)
            mxv = max(mxv, abs(max(mat_np.min(), mat_np.max(), key=abs)))
        return mxv

    def train_net(self, other_ann=None, temp_maximum_epochs=None,
                  training_title='Training the ANN', cross_validation=False, validation_data_set=None):
        """
        A procedure to train an ANN
        @param other_ann: if we need to train an ANN other than the main network of the study, we mention its name here
                            The main ANN of the study is referenced as self.ann
        @param temp_maximum_epochs: if we need to change  the default max epochs during training.
        @param cross_validation: Boolean, True if the mode of the study requires cross validation, False by default
        @param training_title: the specific trainig title (sometimes it is "Training Structure #???"
        @param validation_data_set: the name of the validation data set if the mode requires validation

        @return: A tuple of the following:
                    (>error_list or costs list, one value per epoch,
                    >reached_epochs during this training,
                    >a list of coefficient of correlation for each variable,
                    >outputs: a list of lists contains comparison between expected values and calculated ones
                            in the form:[[training_outputs],[ann_outputs]],
                    >the elapsed training time)
        """

        def train_line(study, training_inputs_l, training_outputs_l, other_ann_l=None):
            """
            # Uses online learning, i.e. updating the weights after each training case
            @param study: the parent study
            @param training_inputs_l: a list of normalized values to train as inputs
            @param training_outputs_l: a list of normalized values to train as outputs
            @param other_ann_l: if we want to train lines for an ANN other than the default, then tipe its name here
            """
            # 0. Perform Feed Farward
            # ann._inputs.append(training_inputs)
            # ann._output_targets.append(training_outputs)
            if other_ann_l is None:
                ann_l = study.ann
            else:
                ann_l = other_ann_l

            ann_l.feed_forward(training_inputs_l)

            # 1. Output neuron deltas
            # partial derivatives errors with respect to output neuron total net input
            output_neurons_error = [0] * len(ann_l.output_layer.neurons)
            # speedup variable
            output_layer_neurons = ann_l.output_layer.neurons
            for o in range(len(ann_l.output_layer.neurons)):
                # ∂E/∂zⱼ
                output_neurons_error[o] = output_layer_neurons[o].calc_delta(training_outputs_l[o])

            # 2. Hidden neuron deltas
            hidden_neurons_delta = [0] * len(ann_l.hidden_layer.neurons)

            hidden_layer_neurons = ann_l.hidden_layer.neurons
            rng_output_layer_neurons = range(len(output_layer_neurons))
            for h, n in enumerate(hidden_layer_neurons):

                # We need to calculate the derivative of the error with respect to the output of each hidden layer neuron
                # dE/dyⱼ = Σ ∂E/∂zⱼ * ∂z/∂yⱼ = Σ ∂E/∂zⱼ * wᵢⱼ

                hidden_neuron_outputs = 0
                for o in rng_output_layer_neurons:
                    hidden_neuron_outputs += output_neurons_error[o] * output_layer_neurons[o].weights[h]

                # ∂E/∂zⱼ = dE/dyⱼ * ∂zⱼ/∂
                hidden_neurons_delta[h] = hidden_neuron_outputs \
                                          * hidden_layer_neurons[h].derive_func(n.activation_function)

            # 3. Update output neuron weights
            for o in rng_output_layer_neurons:
                # update bias NESR
                updated_weight = output_neurons_error[o]
                output_layer_neurons[o].bias -= ann_l.learning_rate * updated_weight
                # Update weights here
                for w_ho in range(len(ann_l.output_layer.neurons[o].weights)):
                    # ∂Eⱼ/∂wᵢⱼ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢⱼ
                    updated_weight = output_neurons_error[o] * output_layer_neurons[o].neuron_net_input(w_ho)

                    # Δw = α * ∂Eⱼ/∂wᵢ
                    output_layer_neurons[o].weights[w_ho] -= ann_l.learning_rate * updated_weight

            # 4. Update hidden neuron weights
            for h in range(len(ann_l.hidden_layer.neurons)):
                # update bias NESR
                updated_weight = hidden_neurons_delta[h]
                hidden_layer_neurons[h].bias -= ann_l.learning_rate * updated_weight
                # Update weights here
                for w_ih in range(len(ann_l.hidden_layer.neurons[h].weights)):
                    # ∂Eⱼ/∂wᵢ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢ
                    updated_weight = hidden_neurons_delta[h] * hidden_layer_neurons[h].neuron_net_input(w_ih)

                    # Δw = α * ∂Eⱼ/∂wᵢ
                    hidden_layer_neurons[h].weights[w_ih] -= ann_l.learning_rate * updated_weight

        # ====================================================================================
        cv_error = [[], []]

        print "\n", training_title, ", \n ...showing number of epochs till now"
        t1 = time.time()
        ann = self.ann if other_ann is None else other_ann
        data = self.normalized_data
        tolerance = self.tolerance
        if temp_maximum_epochs is None:
            maximum_epochs = self.maximum_epochs
        else:
            maximum_epochs = temp_maximum_epochs
        error_list = []
        epoch = 0  # to avoid errors !

        def special_print(value):
            """
            http://stackoverflow.com/questions/39123549/how-to-print-tight-dots-horizontally-in-python
            @param value:
            """
            sys.stdout.write(value)
            sys.stdout.flush()

        fit_epochs = self.number_of_epochs_for_overfit_check

        # putting a huge number for stored error so that the system starts to store ANN with minimal error
        stored_error = float('inf')
        current_error = 0
        there_is_a_stored_ann = False
        minimal_error_is_at = 0
        stored_ann = 0  # to initialize the variable to avoid 'variable may be referenced before assignment' error
        for epoch in range(maximum_epochs):
            # Print current epoch's number if divisable by 10 of a dot instead
            if epoch % 10 == 0:
                # print epoch,
                special_print(str(epoch))
            else:
                # print ".",
                special_print('.')

            for case in data:
                case_list = list(case)
                training_inputs = case_list[0]
                training_outputs = case_list[1]
                # print training_inputs, training_outputs
                # was ann.train(training_inputs, training_outputs)
                train_line(self, training_inputs, training_outputs, other_ann)

            current_error = ann.calculate_total_error(data)[0]

            # ===========================================================================================

            # a sub-function for local normalizing
            def normalized_list (a_list):
                """
                                Normalizing any list through minimax
                                @param a_list: the list to be normalized
                                @return: normalized list
                                """
                list_min = min(a_list)
                list_range = max(a_list) - list_min
                return map(lambda m: (float(m-list_min))/list_range, a_list)

            # If the study mode is 'cross validation', then we should calculate the validation set's error
            if cross_validation:
                # cloned_ann = ann.clone()
                # validation error
                # v_error = cloned_ann.calculate_total_error(validation_data_set)[0]
                v_error = self.ann.calculate_total_error(validation_data_set)[0]
                cv_error[0].append(current_error)
                cv_error[1].append(v_error)


                if epoch > max(fit_epochs + 3, 11):
                    ovft_x = range(fit_epochs)
                    ovft_y = cv_error[1][-1 * fit_epochs : -1]
                    ovft_y.extend([cv_error[1][-1]])
                    # We should normalize x and y values
                    ovft_x = normalized_list(ovft_x)
                    ovft_y = normalized_list(ovft_y)

                    # This requires to add from SCiPy import linregress
                    # slope, intercept, r_value, p_value, std_err = stats.linregress(ovft_x, ovft_y)
                    ovft_slope = stats.linregress(ovft_x, ovft_y)[0]
                    # ovft_slope = np.polyfit(ovft_x, ovft_y, 1)[0]

                    if ovft_slope > self.minimum_slope_to_consider_overfitting: # positive slope means overfitting
                        # The ANN will starts over fitting, then stop training
                        print '\nStopped as the ANN starts to over fit, slope =' , ovft_slope
                        print '\nLast 10 error values: '
                        for ijk in range(-10, 0):
                            print cv_error[1][ijk],

                        break
                pass
            # ===========================================================================================
            # Store the current ANN if its error exceeds the previous error
            if epoch > 3:
                previous_error = float(error_list[-1])
                earlier_error = float(error_list[-2])
                # we will store a copy only if the previous point was a minimal fall (inverted peak), thus
                # the condition below should be fulfilled
                if current_error > previous_error and previous_error < earlier_error:
                    # Although the previous ANN performs better than the current one, we will store the current ANN
                    # not the previous one, as this is the only available ANN for now, to store the previous one
                    # requires storing the ANN each step, and this will slowdown the whole operation significantly.

                    # It will function only if the current error is less than the previously stored error
                    if current_error < stored_error:
                        stored_ann = self.ann.clone()
                        there_is_a_stored_ann = True
                        stored_error = current_error
                        minimal_error_is_at = epoch
                        print '\nError starts to increase, ... Storing a temporary copy of the current ANN'
                        print 'Current epoch is:' , str(epoch), ', and current error is:', str (current_error)


            # Adjustig learning rate to speedup convergence
            cost_slope = 0
            number_of_epochs_collections_with_positive_slope = 0

            if epoch > 2:
                # max_learning_rate = 0.95
                if abs(float(error_list[-1]) - current_error) > tolerance:
                    error_list.append(current_error)
                    if epoch % 30 == 0:
                        cost_slope = np.polyfit(np.array(range(15)), np.array(error_list[-15:]), 1)[0]
                        if cost_slope > 0:
                            number_of_epochs_collections_with_positive_slope += 1
                        else:
                            number_of_epochs_collections_with_positive_slope = 0
                        if number_of_epochs_collections_with_positive_slope > 2 :
                            print "The error is increasing for more then 90 epochs!\n Exiting iteration...."
                            break  # Escapes from the epochs For-loop
                        # print '(', str(cost_slope), ')',
                        print '({:+.6f})'.format(cost_slope),

                    if self.adapt_learning_rate:
                        if self.annealing_value > 0:
                            # This continuously decrease the learning rate with epochs
                            ann.learning_rate = self.basic_learning_rate / (1 + epoch / self.annealing_value)
                        else:
                            if epoch % 30 == 0:  # Start to adapt the learning rate
                                cost_slope = np.polyfit(np.array(range(15)), np.array(error_list[-15:]), 1)[0]
                                if cost_slope < 0 and ann.learning_rate < 0.7:
                                    if cost_slope < -0.5:
                                        ann.learning_rate *= 1.150
                                        print 'Cost slope = ', cost_slope, ' Increasing LR by 15.%, new LR= ', str(
                                                ann.learning_rate)
                                    elif cost_slope < -0.1:
                                        ann.learning_rate *= 1.100
                                        print 'Cost slope = ', cost_slope, ' Increasing LR by 10.%, new LR= ', str(
                                                ann.learning_rate)
                                    elif cost_slope < -0.01:
                                        ann.learning_rate *= 1.075
                                        print 'Cost slope = ', cost_slope, ' Increasing LR by 7.5%, new LR= ', str(
                                                ann.learning_rate)
                                    elif cost_slope < -0.005:
                                        ann.learning_rate *= 1.050
                                        print 'Cost slope = ', cost_slope, ' Increasing LR by 5.0%, new LR= ', str(
                                                ann.learning_rate)
                                    elif cost_slope < -0.001:
                                        ann.learning_rate *= 1.025
                                        print 'Cost slope = ', cost_slope, ' Increasing LR by 2.5%, new LR= ', str(
                                                ann.learning_rate)
                                else:
                                    ann.learning_rate *= 0.9
                                    print 'Cost slope = ', cost_slope, ' Decreasing LR by 10.%, new LR= ', str(
                                            ann.learning_rate)

                                    # print
                                    # print cost_slope, ann.learning_rate
                else:
                    error_list.append(current_error)

                    print '\n\nStopped as the absolute error difference is less than the given tolerance'
                    print '    The given error tolerance =', tolerance
                    print '    The error difference now  =', abs(float(error_list[-2]) - current_error)

                    break
            else:
                error_list.append(current_error)
                if epoch > 2: print 'Final slope of cost function (', str(cost_slope), ')'

        # Iteration is finished now, we have reached a satisfactory epoch

        # Now, checking whether the final error is less than the stored error or not
        if there_is_a_stored_ann:
            if current_error > stored_error:
                # If so, then we will revert to the stored ANN
                self.ann.revert_to(stored_ann)
                print "\n\nStored ANN's error is less than the current error"
                print "The stored ANN with minimal error has been restored"
                print "Graphs will be built normally for all accessed epochs, "
                print "but we will use the ANN of epoch #", str(minimal_error_is_at)
                print "where its error =", str(stored_error)
                print '\nFrom now on, (during testing stage and studies), the restored ANN will be used.'
            else:
                print "\n\nAlthough there is a stored ANN, but the last error is less than its error"
                print "Last epoch:", str(epoch)
                print "Least error =", str(current_error), 'The stroed error was:', str(stored_error)

        reached_epochs = epoch
        self.master_error_list.append(error_list)
        t2 = time.time()
        time_elapsed = elapsed_time(t1, t2)
        print "\nFor", reached_epochs + 1, "epochs, the training duration = ", time_elapsed

        # Start calculating correlation coefficient
        outputs = []
        for case in data:
            case_list = list(case)
            training_inputs = case_list[0]
            training_outputs = case_list[1]
            train_line(self, training_inputs, training_outputs, other_ann)
            outputs.append([training_outputs, ann.get_ann_outputs()])
        # sums = [0, 0, 0, 0, 0]

        r_coefficient_matrix = []
        for variable in range(len(outputs[0][0])):
            sums = [0, 0, 0, 0, 0]  # sum x, sum y, sum xy, sum x2, sum y2
            for epoch, case in enumerate(data):
                # training_outputs = list(case)[1]
                sums[0] += outputs[epoch][0][variable]  # x
                sums[1] += outputs[epoch][1][variable]  # y
                sums[2] += outputs[epoch][0][variable] * outputs[epoch][1][variable]  # xy
                sums[3] += outputs[epoch][0][variable] * outputs[epoch][0][variable]  # xx
                sums[4] += outputs[epoch][1][variable] * outputs[epoch][1][variable]  # yy
            n = len(data)
            try:
                denominator = math.sqrt(n * sums[3] - sums[0] * sums[0]) * math.sqrt(
                    n * sums[4] - sums[1] * sums[1])
                r = (n * sums[2] - sums[0] * sums[1]) / denominator
            except:
                r = 0.000000079797979797979

            r_coefficient_matrix.append(r)
        if cross_validation:
            # in cross validation mode, we add validation and test errors
            return error_list, reached_epochs, r_coefficient_matrix, outputs, t2 - t1, cv_error
        else:
            return error_list, reached_epochs, r_coefficient_matrix, outputs, t2 - t1


    def get_network_weights(self, other_ann=None):
        """
        Returna a list of lists of all the weights in an ANN
        @param other_ann: if you want the weights of an ANN other than the default one
        @return: list of lists of all the weights and biasses on the form:
                    [weights_of_i_h, weights_of_h_o, bias_of_i_h, bias_of_h_o]
        """
        ann = self.ann if other_ann is None else other_ann
        w_i_h = ann.get_weights(I_H)
        w_h_o = ann.get_weights(H_O)
        weights_of_i_h = transpose_matrix(w_i_h[0])
        weights_of_h_o = transpose_matrix(w_h_o[0], False)
        bias_of_i_h = transpose_matrix(w_i_h[1])
        bias_of_h_o = transpose_matrix(w_h_o[1], False)
        return weights_of_i_h, weights_of_h_o, bias_of_i_h, bias_of_h_o

    def save_to_file(self, save_type, data_to_save=None, file_name=None, clear_file=True):
        """
        Saves some data to a txt file
        @param save_type: 0 to 4, depends on what you want to save
                        0: "weights" on the form weights_of_i_h, weights_of_h_o, bias_of_i_h, bias_of_h_o
                            where each set is printed starting from a new line
                        1: "All outputs"  weights, labels and others
                            If selected, then data_to_save: a tuple of (results, labels)
                        2: when used for find_best_activation_function, the saved data are:
                            [function, number, cost, r, average_correlation, epochs, elapsed_time]
                        3: graph data (Prediction cloud
                        4: query results
        @param data_to_save: ...depending on save_type
        @param file_name: ...of the data
        @param clear_file: If True, the file will be cleared before writing new data,
                            otherwise, new data will be appeneded to existing data
        @return:pass, (It just prints the requested file)
        """
        types = {0: 'Weights.csv', 1: 'Outputs.txt', 2: 'Functions.txt',
                 3: 'PredictionClouds.csv', 4: 'QueryOutput.txt'}
        weights_of_i_h, weights_of_h_o, bias_of_i_h, bias_of_h_o = [], [], [], []
        ann = self.ann

        def remove_brackets(lst):
            """

            @param lst:
            @return:
            """
            lst2 = ""
            try:
                for member in lst:
                    lst2 += (str(member) + ',')
                lst2 = '=>, ' + lst2[:-1]
                return lst2
            except:
                return '=>, ' + str(lst)


        if file_name is None or file_name == False:
            file_name = self.new_folder_path + '\\' + types[save_type]
        if clear_file:
            open(file_name, "w").close()

        if save_type == 0:  # Weights
            if data_to_save is None:
                weights_of_i_h, weights_of_h_o, bias_of_i_h, bias_of_h_o = self.get_network_weights()
            all_weights = []
            for i in weights_of_i_h:
                all_weights.append(i)
            for i in weights_of_h_o:
                all_weights.append(i)
            all_weights.append(bias_of_i_h)
            all_weights.append(bias_of_h_o)
            structure = ann.get_structure()
            max_width = max(structure)  # max(len(x) for x in structure)
            adjusted_weights = []
            for line in all_weights:
                if len(line) == max_width:
                    adjusted_weights.append(line)
                else:
                    temp = [0.0] * (max_width - len(line))
                    line_x = list(line)
                    line_x.extend(temp)
                    adjusted_weights.append(line_x)

            np.savetxt(file_name, np.array(adjusted_weights), delimiter=",", fmt='%.18g')
        elif save_type == 1:  # All outputs
            results, labels = data_to_save
            ending = '\n'
            try:
                file_ann = open(file_name, "a")
            except:
                file_ann = open(file_name, "w")

            now = dt.now()
            file_ann.writelines(['======================\n', str(now) + '\n', '======================\n'])

            for i, label in enumerate(labels):
                crown = '-' * len(label) + '\n'
                file_ann.writelines([crown, label + '\n'])  # , crown])
                if i < 6:
                    if i == 2:  # Epochs errors
                        epochs = range(1, len(results[i]) + 1)
                        file_ann.writelines(remove_brackets(epochs) + '\n')
                        file_ann.writelines(remove_brackets(results[i]) + '\n')
                    else:
                        file_ann.writelines(remove_brackets(results[i]) + '\n')
                else:
                    if i != 9:
                        for res in results[i]:
                            file_ann.writelines(remove_brackets(res) + '\n')
                    else:
                        for w in range(1):
                            for layer in results[i][w]:
                                file_ann.writelines(remove_brackets(layer) + '\n')

                        file_ann.writelines(remove_brackets(results[i][2]) + '\n')
                        file_ann.writelines(remove_brackets(results[i][3]) + '\n')

            file_ann.writelines(ending)
            file_ann.close()
        elif save_type == 2:
            file_ann = open(file_name, "a")
            for line in data_to_save:
                lne = [x for x in line[0]]
                for m in range(1, len(line)):
                    lne.append(line[m])
                clean_line = str(lne)
                clean_line = clean_line.replace('[', '')
                clean_line = clean_line.replace(']', '')
                clean_line = clean_line.replace("'", "")
                file_ann.writelines(clean_line + '\n')
            file_ann.close()
        elif save_type == 3:
            file_ann = open(file_name, "a")
            reformated = []
            tmp = []
            for item in range(len(data_to_save)):
                tmp.append('Data ' + str(item))
                tmp.append('Predicted ' + str(item))
            reformated.append(tmp)

            for item in range(len(data_to_save[0])):
                tmp = []
                for var in range(len(data_to_save)):
                    tmp.append(data_to_save[var][item][0])
                    tmp.append(data_to_save[var][item][1])
                reformated.append(tmp)

            for line in reformated:
                # lne = [x for x in line[0]]
                # for m in range(1, len(line)):
                #     lne.append(line[m])
                clean_line = str(line)  # str(lne)
                clean_line = clean_line.replace('[', '')
                clean_line = clean_line.replace(']', '')
                clean_line = clean_line.replace("'", "")
                file_ann.writelines(clean_line + '\n')
            file_ann.close()
        elif save_type == 4:
            file_ann = open(file_name, "a")

            for line in data_to_save:
                clean_line = str(line)  # str(lne)
                clean_line = clean_line.replace('[', '')
                clean_line = clean_line.replace(']', '')
                clean_line = clean_line.replace("'", "")
                file_ann.writelines(clean_line + '\n')
            file_ann.close()
        pass

    def print_info(self, print_type, r=None):
        # ann = self.ann
        """
        Prints the required information to console in a formated form
        @param print_type:  0:  # print_net_weights
                            1:  # print_relative_importance
                            2:  # print_correlation_coefficient
        @param r: is the correlation_coefficients (if print_type=2)
        """
        if print_type == 0:  # print_net_weights
            weights_of_i_h, weights_of_h_o, bias_of_i_h, bias_of_h_o = self.get_network_weights()
            print_matrix('weights_of_i_h', weights_of_i_h)
            print_matrix('bias_of_i_h', bias_of_i_h)
            print_matrix('weights_of_h_o', weights_of_h_o)
            print_matrix('bias_of_h_o', bias_of_h_o)
        elif print_type == 1:  # print_relative_importance
            re100, re_negative = self.separate_relative_importance()
            print
            print_matrix('relative_importance (+ve contribution)', re100)
            print_matrix('relative_importance (real contribution)', re_negative)
        elif print_type == 2:  # print_correlation_coefficient
            print_matrix('correlation_coefficients', r)

    def prepare_output_file(self, error_list, stopping_epoch, tolerance, correlation_coefficients,
                            matrix_of_sum_of_errors, errors_collection,
                            relative_importance_100, relative_importance_negative, clear_file_state=True):

        """
        To prepare the file bfore printing
        @param error_list: a list of cost values per epoch
        @param stopping_epoch: the epoch the network converged at
        @param tolerance: the registered tollerance that was fulfilled before convergence
        @param correlation_coefficients: per input variable
        @param matrix_of_sum_of_errors: Total Error, MSE, RMSE
        @param errors_collection: Error details: (Total Error; MSE{per output}; RMSE{per output})
        @param relative_importance_100: Relative contributions (+ve only):of each input (columns) to each output (rows)
        @param relative_importance_negative: Relative contributions (real values): as above
        @param clear_file_state: if True(Default), the data will be written to a clean file, else, appended to existing
        """
        ann = self.ann
        gross_network_results = [ann.get_structure(), ann.get_activation_functions(),
                                 error_list, (stopping_epoch, tolerance), correlation_coefficients,
                                 matrix_of_sum_of_errors, errors_collection,
                                 relative_importance_100, relative_importance_negative,
                                 self.get_network_weights()]

        gross_network_labels = ['Network structure: , Inputs, Hidden, Outputs',
                                'Activation functions: 0= Sigmoid, 1= Tanh, 2= Softmax, for I-H, for H-O',
                                'Error advance: Total error at the end of each epoch',
                                'Run conditions: , Number of Epochs, Tolerance',
                                'Correlation coefficients: (A number per output)',
                                'Sum of errors: , Total Error, MSE, RMSE',
                                'Error details: (Total Error; MSE{per output}; RMSE{per output})',
                                'Relative contributions (+ve only): of each input (columns) to each output (rows)',
                                'Relative contributions (real values): of each input (columns) to each output (rows)',
                                'Weights and biases: Weights I-H (rows= inputs); a row for H bias; & other for O bias']

        self.save_to_file(1, (gross_network_results, gross_network_labels), clear_file=clear_file_state)
        # self.store_outputs_to_file(gross_network_results, gross_network_labels, clear_file_state)

        pass

    def get_normalized_input_line(self, input_line):
        """
        Convert an input dataline to normalized form
        @param input_line: the data in raw format
        @return: data in normalized format
        """
        norm_line = []
        for i, cell in enumerate(input_line):
            norm_data = self.source_data.input_variables[i].get_normalized_value(cell)
            if not isinstance(norm_data, list):  # only at the beginning
                norm_line.append(norm_data)
            else:
                norm_line.extend(norm_data)
        return norm_line

    def get_de_normalized_output_line(self, output_line):
        """
        Convert normalized outputs to readable raw format
        @param output_line: list of normalized outputs
        @return: list of readable output format
        """
        var_map = self.get_variables_info('loc')
        var_types_bool = self.get_variables_info('bool')
        output_vars = self.source_data.output_variables
        tmp = []
        finished_output_variables = []
        for o, out in enumerate(output_line):
            if var_map[1][o] not in finished_output_variables:
                if var_types_bool[1][o]:  # Numeric output
                    tmp.append(output_vars[o].get_de_normalized_value(out))
                    finished_output_variables.append(var_map[1][o])
                else:
                    rep = var_map[1].count(var_map[1][o])
                    tmp2 = output_line[o: o + rep]
                    # tmp2 = map(lambda x: 1 if x >= 0.5 else 0, tmp2)
                    tmp.append(output_vars[o].get_de_normalized_value(tmp2))
                    finished_output_variables.append(var_map[1][o])

        return tmp

    def graph_results(self, error_list, graphing_data_collection, optional_errors=None, partitioned_data=None,
                      initial_time=0):
        """
        The most important routine in the program. To plot all results in an understandable form
        @param error_list: the list of costs
        @param graphing_data_collection: the data needed to plot all results
        @param optional_errors: if there are some additional errors (like that of validation and testing)
        @param partitioned_data: The way of partitioning data to TRN:VLD:TST
        @param initial_time: the at whish the study started
        @return: pass, Just outputs graphs in pdf format or in windows format
        """
        figure_number = 0
        figure_page = []
        pages_titles = ['Cost function during simulation stages',
                        "The full neural network with weights' effects",
                        "Consolidated neural network with weights' effects",
                        "Relative importance of inputs to outputs",
                        "Prediction function and data cloud",
                        "Real vs. predicted data"]

        # Limiting data points to maximum of 1000 point to speedup drawing
        max_len = 1000
        Limited_graphing_data_collection = []
        for stage in graphing_data_collection:
            if len(stage) > max_len:
                guide = range(len(stage))
                random.shuffle(guide)
                guide = guide[:max_len]
                dummy_ = []
                for selected in guide:
                    dummy_.append(stage[selected])
                Limited_graphing_data_collection.append(dummy_)
            else:
                Limited_graphing_data_collection.append(stage)


        def save_graph_data(filename, x_and_y_list_of_tuples):
            """
            A function to print the data of each graph page to a csv file
            :param filename: the filename to be saved as
            :param x_and_y_list_of_tuples: a list of several members, each member is a tuple, each tuple contains
                    2 members; GraphTitles and GraphData, GraphTitles is a tuple contains Chart Title, X Title, Y Title
                    GraphData is a tuple contains (X, Y) each as a list.
                    Example
                    x_and_y_list_of_tuples = [(('klkl vs yuyu', 'klkl in kg', 'yuyu in m'), (X, Y)),
                                                      (('klkl vs asdr', 'klkl in kg', 'asdr in s'), (X, Z)),
                                                      (('qwer vs yuyu', 'qwer in F', 'yuyu in m'), (F, Y)),
                                                      (('asdr vs mdfg', 'asdr in s', 'mdfg in N'), (Z, R))]
            """
            file_name = self.new_folder_path + '\\' + filename
            open(file_name, "w").close()
            graph_file = open(file_name, "a")
            chart_number = 0
            for tup in x_and_y_list_of_tuples:
                if len(tup[1][0]) > 0:
                    chart_number += 1
                    graph_file.writelines("Graph #" + str(chart_number) + '\n\n')
                    graph_file.writelines(tup[0][0] + '\n\n' + tup[0][1] + ", " + tup[0][2] + '\n')
                    for j in range(len(tup[1][0])):
                        line = str(tup[1][0][j])
                        for ij in range(1, len(tup[1])):
                            line += ", " + str(tup[1][ij][j])
                        graph_file.writelines(str(line) + '\n')
                    graph_file.writelines("\n===========================" + '\n\n')

            graph_file.close()

        def draw_cost_function():
            """
            Draws the cost function(s) of training, testing, validation,
                    and all other costs like that of selecting structure
            @return: A graphs figure
            """
            data_save_lists = []  # [() for i in range (5)]
            y = error_list if optional_errors is None else optional_errors[0]
            x = range(len(y))
            plots_in_fig = (2, 6)
            # Note that, unlike matplotlib’s subplot, the index  of subplot2grid starts from 0 in gridspec.
            ax1 = plt.subplot2grid(plots_in_fig, (0, 0), colspan=2)
            plt.plot(x, y, 'r', marker='.')
            chart_title = 'Error development during training' if optional_errors is None \
                else 'Error development during early validation'
            plt.title(chart_title, fontsize=16, weight='bold', color='maroon')
            plt.xlabel('Epochs', fontsize=14, weight='bold')
            plt.ylabel('Cost/error', fontsize=14, weight='bold')
            plt.grid(True)
            # saving data to file
            data_save_lists.append(((chart_title, 'Epochs', 'Cost/error'), (x, y)))

            if optional_errors is not None:
                # ===============================================================
                # Graphing the validation error

                y = optional_errors[1]
                x = range(len(y))

                # Note that, unlike matplotlib’s subplot, the index  of subplot2grid starts from 0 in gridspec.
                ax1 = plt.subplot2grid(plots_in_fig, (0, 2), colspan=2)
                plt.plot(x, y, 'b', marker='.')
                plt.title('Error development during training', fontsize=16, weight='bold', color='maroon')
                plt.xlabel('Epochs', fontsize=14, weight='bold')
                # plt.ylabel('Cost/error', fontsize=14, weight='bold')
                plt.grid(True)
                data_save_lists.append((('Error development during training', 'Epochs', 'Cost/error'), (x, y)))

                # ===============================================================
                # Graphing the Testing error

                y = optional_errors[2]
                x = range(len(y))

                # Note that, unlike matplotlib’s subplot, the index  of subplot2grid starts from 0 in gridspec.
                ax1 = plt.subplot2grid(plots_in_fig, (0, 4), colspan=2)
                plt.plot(x, y, 'g', marker='.')
                plt.title('Error development during late testing', fontsize=16, weight='bold', color='maroon')
                plt.xlabel('Epochs', fontsize=14, weight='bold')
                # plt.ylabel('Cost/error', fontsize=14, weight='bold')
                plt.grid(True)
                data_save_lists.append((('Error development during late testing', 'Epochs', 'Cost/error'), (x, y)))
            # ===============================================================
            # Graphing the Total epochs error
            tot_err = self.master_error_list
            y = list(itertools.chain(*tot_err))

            x = range(len(y))

            # Note that, unlike matplotlib’s subplot, the index  of subplot2grid starts from 0 in gridspec.
            ax1 = plt.subplot2grid(plots_in_fig, (1, 0), colspan=6)
            plt.plot(x, y, 'deepskyblue', marker='.')
            plt.title('Error development during the whole operation', fontsize=16, weight='bold', color='maroon')
            plt.xlabel('Epochs', fontsize=14, weight='bold')
            plt.ylabel('Cost/error', fontsize=14, weight='bold')
            plt.grid(True)
            data_save_lists.append((('Error development during the whole operation', 'Epochs', 'Cost/error'), (x, y)))

            plt.interactive(False)
            plt.suptitle(pages_titles[figure_number], fontsize=25, weight='bold')

            # saving data to file
            save_graph_data('Grf1_Costs.csv', data_save_lists)

        def draw_full_ann():
            """
            Draws a full ANN, with normalized data
            @return: A graphs figure
            """
            structure = self.ann.get_structure()
            var_types_bool = self.get_variables_info('bool')
            var_locations = self.get_variables_info('loc')
            ax3 = plt.subplot2grid((2, 2), (0, 0), rowspan=2, colspan=2)
            # labels1 = self.source_data.data_style
            labels = []
            # for lab in labels1:
            #     if 'I' in lab:
            #         labels.append(lab.replace('I', ' '))
            #     if 'O' in lab:
            #         labels.append(lab.replace('O', ' '))
            for j in [0, 1]:
                for i in var_locations[j]:
                    labels.append(self.source_data.classified_briefs[j][i])
            labels = [labels, structure[0]]
            plt_net = PlotNeuralNetwork(labels, horizontal__distance_between_layers=15,
                                        vertical__distance_between_neurons=2,
                                        neuron_radius=0.5,
                                        number_of_neurons_in_widest_layer=max(structure) + 3,
                                        numeric_categoric_list=var_types_bool)

            ann = self.ann
            w_i_h = ann.get_weights(I_H)
            w_h_o = ann.get_weights(H_O)
            w12 = w_i_h[0]
            w23 = w_h_o[0]
            b12 = w_i_h[1]
            b23 = w_h_o[1]
            max_abs_val = self.max_absolute_value((w12, w23, b12, b23))
            weights12 = self.adjusted_line_thickness(w12,
                                                     max_abs_val)  # weights12 = np.asarray(ann.get_weights(I_H)[0])
            weights23 = self.adjusted_line_thickness(w23,
                                                     max_abs_val)  # weights23 = np.asarray(ann.get_weights(H_O)[0])
            bias12 = self.adjusted_line_thickness(b12, max_abs_val)  # bias12 = np.asarray(ann.get_weights(I_H)[1])
            bias23 = self.adjusted_line_thickness(b23, max_abs_val)  # bias23 = np.asarray(ann.get_weights(H_O)[1])

            plt_net.add_layer(structure[0], 'inputs', weights12)
            plt_net.add_layer(structure[1], 'hidden', weights23)
            plt_net.add_layer(structure[2], 'outputs')
            plt_net.add_bias(0, 1, bias12)
            plt_net.add_bias(1, 2, bias23)

            # Rotated_Plot = ndimage.rotate(plt_net, 90)
            #
            # plt.imshow(Rotated_Plot)  #, cmap=plt.cm.gray)
            # plt.axis('off')

            plt_net.draw(inputs_label="Normalized Inputs Layer", outputs_label="Normalized Outputs Layer")
            # plt.tight_layout()
            plt.interactive(False)
            plt.suptitle(pages_titles[figure_number], fontsize=25, weight='bold')
            return w12, w23, b12, b23, structure

        def draw_brief_ann(w12, w23, b12, b23, structure):

            """
            Draws a virtual ANN corresponding to the logic of pre-normalization
            @param w12: weights from inputs to hidden
            @param w23: weights from hidden to output
            @param b12: bias to hidden
            @param b23: bias to output
            @param structure: the ANN normalized structure
            @return: A graphs figure
            """

            def consolidate_weights(matrix, var_map, var_bool, for_inputs=True):
                """

                @param matrix:
                @param var_map:
                @param var_bool:
                @param for_inputs:
                @return:
                """
                if for_inputs:
                    mat = []
                    # first, finish all the inputs if categorical
                    for i, out in enumerate(matrix):
                        tmp = []
                        finished_var = []
                        for j, cell in enumerate(out):
                            if var_map[0][j] not in finished_var:
                                if var_bool[0][j]:  # if true then it is numeric
                                    tmp.append(cell)
                                    finished_var.append(var_map[0][j])
                                else:
                                    # find how many elments should be consolidated by the category
                                    rep = var_map[0].count(var_map[0][j])
                                    # elements = []
                                    # for r in range(j, j + rep):
                                    #     elements.append(matrix[i][r])
                                    elements = list(matrix[i][j:j + rep])

                                    tmp.append(rms_mean(elements) * dominant_sign(elements))
                                    finished_var.append(var_map[0][j])
                        mat.append(tmp)
                    return mat
                else:  # for outputs
                    fin = []
                    finished_var = []
                    if len(matrix) == 1 and self.source_data.num_outputs == 1:
                        return [rms_mean(matrix[0]) * dominant_sign(matrix[0])]
                    # if len(matrix) == 1:
                    #     matrix = matrix[0]

                    for i, out in enumerate(matrix):
                        if var_map[1][i] not in finished_var:
                            if var_bool[1][i]:  # if true then it is numeric
                                tmp = list(matrix [i]) # list(out)
                                # tmp = []
                                # for j, cell in enumerate(out):
                                #     tmp.append(cell)
                                finished_var.append(var_map[1][i])
                            else:
                                # find how many elments should be consolidated by the category
                                rep = var_map[1].count(var_map[1][i])
                                tmp = []
                                for j, cell in enumerate(out):
                                    elements = []
                                    for r in range(i, i + rep):
                                        elements.append(matrix[r][j])
                                    tmp.append(rms_mean(elements) * dominant_sign(elements))
                                finished_var.append(var_map[1][i])
                            fin.append(tmp)
                    # to flaten fin matrix
                    # if len(matrix) == 1:
                    #     fin = [item for sublist in fin for item in sublist]
                    return fin

            ax3 = plt.subplot2grid((2, 2), (0, 0), rowspan=2, colspan=2)

            labels = [self.source_data.briefs, self.source_data.num_inputs]


            plt_net = PlotNeuralNetwork(labels, horizontal__distance_between_layers=15,
                                        vertical__distance_between_neurons=2,
                                        neuron_radius=0.5, number_of_neurons_in_widest_layer=max(structure) + 3,
                                        numeric_categoric_list=self.get_variables_info('original boolean'),
                                        categoric_is_lighter=False)

            var_map, var_bool = self.get_variables_info('loc'), self.get_variables_info('bool')
            w12 = consolidate_weights(w12, var_map, var_bool)
            w23 = consolidate_weights(w23, var_map, var_bool, for_inputs=False)
            # the bias of the hidden layer needs no modifications as there is no categoric vars there
            # b12 = self.consolidate_weights([b12], var_map, var_bool)
            # b23 = consolidate_weights([b23], var_map, var_bool, for_inputs=False)
            tmp_b23 = []  # [b23]
            for item in b23:
                tmp_b23.append([item])
            b23 = consolidate_weights(tmp_b23, var_map, var_bool, for_inputs=False)
            b23 = [item for sublist in b23 for item in sublist]

            max_abs_val = self.max_absolute_value((w12, w23, b12, b23[0]))
            weights12 = self.adjusted_line_thickness(w12, max_abs_val)  # = np.asarray(ann.get_weights(I_H)[0])
            weights23 = self.adjusted_line_thickness(w23, max_abs_val)  # = np.asarray(ann.get_weights(H_O)[0])
            bias12 = self.adjusted_line_thickness(b12, max_abs_val)  # = np.asarray(ann.get_weights(I_H)[1])
            # bias23 = self.adjusted_line_thickness(b23[0], max_abs_val)  # = np.asarray(ann.get_weights(H_O)[1])
            bias23 = self.adjusted_line_thickness(b23, max_abs_val)  # = np.asarray(ann.get_weights(H_O)[1])
            # if not(isinstance(bias23, list)):
            #     # print bias23
            #     bias23 = [bias23]

            virtual_structure = [self.source_data.num_inputs, structure[1], self.source_data.num_outputs]
            plt_net.add_layer(virtual_structure[0], 'inputs', weights12)
            plt_net.add_layer(virtual_structure[1], 'hidden', weights23)
            plt_net.add_layer(virtual_structure[2], 'outputs')
            plt_net.add_bias(0, 1, bias12)
            plt_net.add_bias(1, 2, bias23)

            plt_net.draw(inputs_label="De-normalized Inputs Layer", outputs_label="De-normalized Outputs Layer")
            # plt.tight_layout()
            plt.interactive(False)
            plt.suptitle(pages_titles[figure_number], fontsize=25, weight='bold')
            pass

        def draw_relative_importance():
            """
            Draws two types of graphs,
                1- a bar graph for +ve or -ve importance (one graph per study)
                2- pie charts of +ve relative importance (one chart per output variable)
            @return: A graphs figure
            """
            data_save_lists = []
            graph_grid2, graph_grid_sub = self.find_suitable_grid(num_outputs_original + 1)
            graph_grid_sub_adjusted = (graph_grid_sub[0][0] + 0, graph_grid_sub[0][1] + 0)
            ax2 = plt.subplot2grid(graph_grid2, graph_grid_sub_adjusted, colspan=1)
            ax2.axhline(0, color='black', linewidth=4)

            def bottoms_matrix(matrix):
                """

                @param matrix:
                @return:
                """
                positives = []
                negatives = []
                for i, row_mat in enumerate(matrix):
                    tmp_p = []
                    tmp_n = []
                    for j, cell in enumerate(row_mat):
                        if cell > 0:
                            tmp_p.append(cell)
                            tmp_n.append(0.)
                        else:
                            tmp_p.append(0.)
                            tmp_n.append(cell)
                    positives.append(tmp_p)
                    negatives.append(tmp_n)

                # get cumulative sums
                positives = positives[:-1]
                negatives = negatives[:-1]
                positives.insert(0, [0.] * len(matrix[0]))
                negatives.insert(0, [0.] * len(matrix[0]))
                tmp = transpose_matrix(positives)
                tmp = [list(np.cumsum(t)) for t in tmp]
                positives = transpose_matrix(tmp)

                tmp = transpose_matrix(negatives)
                tmp = [list(np.cumsum(t)) for t in tmp]
                negatives = transpose_matrix(tmp)

                final_matrix = []
                for i, row_mat in enumerate(matrix):
                    tmp = []
                    for j, cell in enumerate(row_mat):
                        tmp.append(positives[i][j] if cell > 0 else negatives[i][j])
                    final_matrix.append(tmp)
                return final_matrix

            label_inputs,  label_outputs = self.source_data.classified_briefs

            y = []
            rel_imp = self.get_relative_importance()
            for i in range(num_outputs_original):
                sizes = rel_imp[i][1]
                y.append(sizes)

            new_y = []
            for line in y:
                if sum(line) > 0:
                    new_y.append(line)
                else:
                    new_y.append(map(lambda x: -x, line))

            y = transpose_matrix(new_y)
            ind = np.arange(num_outputs_original)  # the x locations for the groups
            width = 0.35  # the width of the bars: can also be len(x) sequence
            bars = [0] * num_inputs_original
            dat = tuple(y[0])
            bottoms = bottoms_matrix(y)
            bars[0] = plt.bar(ind, dat, width, bottom=bottoms[0], color=nesr_colors[0])

            for i in range(1, len(y)):
                dat = tuple(y[i])
                col = nesr_colors[i] if i in nesr_colors else np.random.rand(3, 1)
                bars[i] = plt.bar(ind, dat, width, bottom=bottoms[i], color=col)  # , yerr=menStd)

            legend_info = []
            for bar in bars:
                legend_info.append(bar[0])
            legend_info = tuple(legend_info)

            axes = plt.gca()
            # axes.set_xlim([xmin, xmax])
            axes.set_ylim([-100, 100])

            plt.ylabel('Relative contribution', fontsize=14, weight='bold')
            plt.title('Contribution of inputs to outputs', fontsize=16, weight='bold', color='maroon')
            plt.xticks(ind + width / 2., tuple(label_outputs))
            leg = plt.legend(legend_info, tuple(label_inputs), loc=0)
            # , bbox_to_anchor=(0.5, -0.05))  # , fancybox=True, shadow=True, ncol=5)
            leg.get_frame().set_alpha(0.5)

            # ===============================================================
            # Graphing relative importance pie charts
            # cur_plt = 4
            # graphing relative importance 1
            colors = []

            for i in range(num_inputs_original):
                colors.append(nesr_colors[i])

            for p in range(num_outputs_original):
                # cur_plt += 1
                graph_grid_sub_adjusted = (graph_grid_sub[p + 1][0] + 0, graph_grid_sub[p + 1][1] + 0)
                ax5 = plt.subplot2grid(graph_grid2, graph_grid_sub_adjusted, colspan=1)
                # plt.subplot(num_rows, num_cols, cur_plt)  # rows, cols, numPlot
                labels = self.source_data.classified_briefs[0]
                explode = []
                for i in range(num_inputs_original):
                    # labels.append('input ' + str(i))
                    if i % 3 == 0:
                        explode.append(0.12)
                    else:
                        explode.append(0.06)

                sizes = rel_imp[p][0]

                # explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
                _, _, autotexts = plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                                          autopct=lambda (p): '{:.0f}'.format(p * sum(sizes) / 100),
                                          shadow=True, startangle=90, counterclock=True)  # autopct='%1.1f%%'
                plt.axis('equal')
                margin = 2  # max(sizes)
                plt.axis([-margin, margin, -margin, margin])
                # plt.title('Effects on output#' + str(p) + ' (%)', fontsize=16)
                plt.title('Effects on ' + str(self.source_data.classified_titles[1][p]) + ' (%)',
                                          fontsize=16, weight='bold', color='maroon')
                for i, atx in enumerate(autotexts):
                    atx.set_color(nesr_colors[i + 1])
                    atx.set_weight('bold')
                # Set aspect ratio to be equal so that pie is drawn as a circle.
                # plt.tight_layout()
            plt.interactive(False)
            # plt.tight_layout()
            plt.suptitle(pages_titles[figure_number], fontsize=25, weight='bold')
            # export to text file
            data_save_lists = []
            for i, out_var in enumerate(label_outputs):
                data_save_lists.append((("Relative importance of " + label_outputs[i],
                                         "Inputs", 'Real RI value %, Absolute RI value %'),
                                        (label_inputs, rel_imp[i][1], rel_imp[i][0])))
            save_graph_data('Grf2_Importance.csv', data_save_lists)
            pass

        def draw_prediction_cloud():
            """
            Draws line charts for all data points sorted from less to highest with the prediction in lines,
                                while originals in dots
            @return: A graphs figure
            """
            data_save_lists = []
            out_graph_n = {}  # A list of normalized outputs
            for n_out in range(self.num_outputs_normalized):
                out_graph_n[n_out] = []

            out_graph_o = {}  # list of original outputs
            for n_out in range(num_outputs_original):
                out_graph_o[n_out] = []

            var_types_bool = self.get_variables_info('bool')
            var_locations = self.get_variables_info('loc')
            # print var_types_bool
            num_outs = self.num_outputs_normalized
            point_dict = {'vld': [[] for i in range(num_outs)],
                          'trn': [[] for i in range(num_outs)],
                          'tst': [[] for i in range(num_outs)]}
            for outputs_to_graph in Limited_graphing_data_collection:

                for dta in outputs_to_graph:
                    for n_out in range(self.num_outputs_normalized):
                        out_graph_n[n_out].append((dta[0][n_out], dta[1][n_out]))
                        if dta[0] in partitioned_data[0]:
                            point_dict['trn'][n_out].append(dta[0][n_out])  # ((dta[0][n_out], dta[01][n_out]))

                        elif dta[0] in partitioned_data[1]:
                            point_dict['vld'][n_out].append(dta[0][n_out])  # ((dta[0][n_out], dta[01][n_out]))

                        elif dta[0] in partitioned_data[2]:
                            point_dict['tst'][n_out].append(dta[0][n_out])  # ((dta[0][n_out], dta[01][n_out]))

            # Limiting out_graph list to maximum of 1000 datapoint to speedup drawing
            max_len = 1000
            Limited_out_graph_n = []
            guide = range(len(out_graph_n[0]))
            random.shuffle(guide)
            guide = guide[:max_len]
            # clg_vars =[]
            for out in out_graph_n:
                if len(out_graph_n[out]) > max_len:
                    dummy_ = []
                    for selected in guide:
                        dummy_.append(out_graph_n[out][selected])
                    Limited_out_graph_n.append(dummy_)
                else:
                    Limited_out_graph_n.append(out_graph_n[out])

            # sorting within the dictionary by the second item
            scatter_categoric = {}
            finished_output_variables = []
            for n_out in range(self.num_outputs_normalized):
                if var_locations[1][n_out] not in finished_output_variables:
                    # sorting only numeric outputs
                    if var_types_bool[1][n_out] is True:  # out_graph_n[n_out].sort(key=lambda x: x[1])
                        out_graph_o[var_locations[1][n_out]] = sorted(Limited_out_graph_n[n_out], key=lambda x: x[1])
                        finished_output_variables.append(var_locations[1][n_out])
                    else:  # For categoric variables only
                        # find how many elments should be consolidated by the category
                        rep = var_locations[1].count(var_locations[1][n_out])
                        elements = []
                        for r in range(n_out, n_out + rep):
                            elements.append(Limited_out_graph_n[r])
                        scatter_categoric[var_locations[1][n_out]] = []
                        for elm in elements:
                            scatter_categoric[var_locations[1][n_out]] += elm
                        compound = []
                        for item in range(len((elements[0]))):
                            tmp = []
                            for el in range(len(elements)):
                                tmp.append(elements[el][item][0])
                                tmp.append(elements[el][item][1])

                            compound.append(tmp)
                        compound.sort(key=lambda x: x[1])
                        compound_data = []
                        num_components = len(compound[0]) / 2
                        inv_num_com = 1. / num_components
                        for item in compound:
                            sss = 0.
                            for j in range(1, len(item), 2):
                                sss += item[j]
                            compound_data.append((inv_num_com, sss / num_components))
                        out_graph_o[var_locations[1][n_out]] = compound_data
                        if var_locations[1][n_out] not in finished_output_variables:
                            finished_output_variables.append(var_locations[1][n_out])

            self.save_to_file(3, out_graph_o)

            graph_grid2, graph_grid_sub = self.find_suitable_grid(num_outputs_original)
            for o in range(num_outputs_original):
                if var_types_bool[1][var_locations[1].index(o)]:
                    # print o, 'Numeric' if var_types_bool[1][o] is True else 'Categoric'
                    x = range(len(out_graph_o[0]))
                    y1 = []
                    y2 = []
                    y_tr = []
                    y_vl = []
                    y_ts = []
                    x_tr = []
                    x_vl = []
                    x_ts = []
                    norm_out_index = var_locations[1][o]
                    for i in x:
                        cur_point = out_graph_o[o][i][0]
                        y1.append(cur_point)
                        y2.append(out_graph_o[o][i][1])
                        if cur_point in point_dict['vld'][norm_out_index]:
                            y_vl.append(cur_point)
                            x_vl.append(i)
                        if cur_point in point_dict['trn'][norm_out_index]:
                            y_tr.append(cur_point)
                            x_tr.append(i)
                        if cur_point in point_dict['tst'][norm_out_index]:
                            y_ts.append(cur_point)
                            x_ts.append(i)

                            # print out_graph[o][i][0], out_graph[o][i][1]
                    graph_grid_sub_adjusted = (graph_grid_sub[o][0] + 0, graph_grid_sub[o][1] + 0)
                    ax6 = plt.subplot2grid(graph_grid2, graph_grid_sub_adjusted, colspan=1)
                    plt.plot(x, y1, '.', c='black', zorder=1)
                    plt.plot(x_vl, y_vl, 'o', markersize=3, markeredgewidth=0.1, markeredgecolor='black',
                             markerfacecolor='r', fillstyle=None, zorder=5)
                    plt.plot(x_tr, y_tr, 'o', markersize=4, markeredgewidth=0.2, markeredgecolor='w',
                             markerfacecolor='b', fillstyle=None, zorder=3)
                    plt.plot(x_ts, y_ts, 'o', markersize=3, markeredgewidth=0.1, markeredgecolor='yellow',
                             markerfacecolor='g', fillstyle=None, zorder=4)
                    plt.plot(x, y2, linewidth=4, linestyle="-", c='black', zorder=8)  # c=nesr_colors_dark[o]
                    plt.xlabel('data point', fontsize=14, weight='bold')
                    plt.ylabel('Actual/Predicted', fontsize=14, weight='bold')
                    # plt.title('Output ' + str(o) + ' n')
                    plt.title(self.source_data.classified_titles[1][o],
                              fontsize=16, weight='bold', color='maroon')
                    plt.grid(True)

                    data_save_lists.append(((self.source_data.classified_titles[1][o],
                                             'General Datapoints', 'Predicted values, Actual values'), (x, y1, y2)))
                    data_save_lists.append(((self.source_data.classified_titles[1][o],
                                             'Validation Datapoints', 'Actual values'), (x_vl, y_vl)))
                    data_save_lists.append(((self.source_data.classified_titles[1][o],
                                             'Training Datapoints', 'Actual values'), (x_tr, y_tr)))
                    data_save_lists.append(((self.source_data.classified_titles[1][o],
                                             'Testing Datapoints', 'Actual values'), ( x_ts, y_ts)))

                else:  # Categoric
                    x = range(len(out_graph_o[0]))
                    y = []
                    for i in range(len(out_graph_o[0])):
                        y.append(out_graph_o[o][i][1] - out_graph_o[o][i][0])
                    graph_grid_sub_adjusted = (graph_grid_sub[o][0] + 0, graph_grid_sub[o][1] + 0)
                    ax6 = plt.subplot2grid(graph_grid2, graph_grid_sub_adjusted, colspan=1)
                    plt.plot(x, y, '.', c=nesr_colors_dark[o])

                    plt.xlabel('data point', fontsize=14, weight='bold')
                    plt.ylabel('Error', fontsize=14, weight='bold')
                    # plt.title('Output ' + str(o) + ' c')
                    plt.title(str(self.source_data.classified_titles[1][o]) + ' prediction error',
                              fontsize=16, weight='bold', color='maroon')
                    plt.grid(True)
                    data_save_lists.append(((self.source_data.classified_titles[1][o],
                                             'data point', 'Error/Bias'), (x, y)))

            plt.interactive(False)
            # plt.tight_layout()
            plt.suptitle(pages_titles[figure_number], fontsize=25, weight='bold')
            save_graph_data('Grf3_Clouds.csv', data_save_lists)
            return out_graph_o, graph_grid_sub, graph_grid2, scatter_categoric

        def draw_real_vs_forecasted():
            """
            Draws line charts, one per output. Each chart is a 45 degree chart for matching inputs with outputs
            @return: A graphs figure
            """
            data_save_lists = []
            var_types_bool = self.get_variables_info('bool')
            var_locations = self.get_variables_info('loc')
            for o in range(num_outputs_original):
                if var_types_bool[1][var_locations[1].index(o)]:
                    # print o, 'Numeric' if var_types_bool[1][o] is True else 'Categoric'
                    x = range(len(out_graph_o[0]))
                    y1 = []
                    y2 = []

                    norm_out_index = var_locations[1][o]
                    for i in x:
                        cur_point = out_graph_o[o][i][0]
                        y1.append(cur_point)
                        y2.append(out_graph_o[o][i][1])

                    graph_grid_sub_adjusted = (graph_grid_sub[o][0] + 0, graph_grid_sub[o][1] + 0)
                    ax6 = plt.subplot2grid(graph_grid2, graph_grid_sub_adjusted, colspan=1)

                    plt.plot(y1, y2, 'o', markersize=3, markeredgewidth=0.1, markeredgecolor='black',
                             markerfacecolor=nesr_colors_dark[o], fillstyle=None, zorder=5)

                    plt.plot([0, max(max(y1), max(y2))], [0, max(max(y1), max(y2))],
                             linewidth=2, linestyle="-", c='blue', zorder=8)  # c=nesr_colors_dark[o]
                    plt.xlabel('Given data', fontsize=14, weight='bold')
                    plt.ylabel('Predicted  data', fontsize=14, weight='bold')
                    # plt.title('Output ' + str(o) + ' n')
                    plt.title(self.source_data.classified_titles[1][o],
                              fontsize=16, weight='bold', color='maroon')  # + ' prediction error')
                    plt.grid(True)
                    data_save_lists.append(((self.source_data.classified_titles[1][o],
                                             'Given data', 'Predicted  data'), (y1, y2)))

                else:  # Categoric
                    # x = range(len(out_graph_o[0]))
                    y1 = []
                    y2 = []
                    h = []
                    for i in range(len(scatter_categoric[o])):
                        y1.append(scatter_categoric[o][i][0])
                        y2.append(scatter_categoric[o][i][1])
                        h.append(y1[i] - y2[i])
                    graph_grid_sub_adjusted = (graph_grid_sub[o][0], graph_grid_sub[o][1])
                    ax6 = plt.subplot2grid(graph_grid2, graph_grid_sub_adjusted, colspan=1)

                    my_bins = [-1, -0.5, -0.1, -0.01, 0.01, 0.1, 0.5, 1]
                    plt.hist(h, bins=my_bins, color=nesr_colors_dark[o])  # , '.', c=nesr_colors_dark[o]), bins=10

                    plt.xlabel('bias of actual from predicted', fontsize=14, weight='bold')
                    plt.ylabel('Frequency', fontsize=14, weight='bold')
                    # plt.title('Output ' + str(o) + ' c')
                    plt.title(self.source_data.classified_titles[1][o] + ' prediction correctness',
                              fontsize=16, weight='bold', color='maroon')
                    plt.grid(True)
                    data_save_lists.append(((self.source_data.classified_titles[1][o],
                                             'Given data', 'Predicted  data'), (y1, y2)))
                    data_save_lists.append(((self.source_data.classified_titles[1][o] + ' (for histogram)',
                                             'point', 'difference'), (range(len(h)), h)))

            plt.interactive(False)
            # plt.tight_layout()
            plt.suptitle(pages_titles[figure_number], fontsize=25, weight='bold')
            save_graph_data('Grf4_Predictions.csv', data_save_lists)
            pass

        def draw_parametric_graphs(figure_number, mother_study):
            """
            Selects 3 base cases at (25%, 50%, and 75% of the data,
                then it changes each input variable in a range between its minimum and maximum values
                Then it plots it. If the variable is categoric, it selects three cases of it if available.
            @param figure_number: the starting figure number, as each variable will have a separate figure
            @param mother_study: the study in which the graphing is performed.
            @return: A graphs figure
            """

            def get_three_normal_means(study, base, var_locations, var_bool):
                """
                Returns 3 base cases at (25%, 50%, and 75% of the data
                @param study: the current study
                @param base: the main base (50%)
                @param var_locations: a list of 2 lists (input and output), contains numbers of original variables
                            (before normalization) on a normalized length:
                            example [[0, 1, 1, 2, 3, 3, 3],[0, 0, 0, 1, 2, 2]] if the input vars 1, 3 are categoric
                            with 2 and 3 members respectively, and the output variables 0, 2 are categoric with 3, 2
                            members respectively. The other variables are Numeric
                @param var_bool: a list of True / False, depending on the variable type NUmeric or Categoric
                @return: a list of the three_normal_means
                """
                three_normal_means = []
                normalized_mean = []
                for i, mean in enumerate(base):
                    if isinstance(mean, str):
                        normalized_mean.append(0)  # for categoric, it is either 0 or 1
                    else:
                        normalized_mean.append(study.source_data.input_variables[var_locations[0][i]].
                                               single_mini_max(mean))
                for i in range(3):
                    three_normal_means.append(copy.deepcopy(normalized_mean))
                # three_normal_means = three_normal_means[0]
                get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
                finished_vars = []
                for j, bol in enumerate(var_bool[0]):
                    var_num = var_locations[0][j]
                    if var_num not in finished_vars:
                        if bol:  # Numeric input
                            finished_vars.append(var_num)
                            # if i=1 then we want 1/4, else we want 3/4
                            var_stat = study.source_data.input_variables[var_num].get_basic_stats()
                            # if i=1 min_max = minimum, else maximum
                            for i in [1, 2]:
                                min_max = var_stat[i]
                                new_avg = (min_max + var_stat[0]) / 2.
                                three_normal_means[i][j] = study.source_data.input_variables[var_num].single_mini_max(
                                    new_avg)
                        else:  # categoric input
                            finished_vars.append(var_num)
                            # we will take two variables, one with [1, 0, ...0], and the other with [0, 0, .. , 0, 1]
                            var_indices = get_indexes(var_num, var_locations[0])
                            first_index, last_index = var_indices[0], var_indices[-1]
                            three_normal_means[1][first_index] = 1
                            three_normal_means[2][last_index] = 1
                            pass
                return three_normal_means

            data_save_lists = []
            marker_sizes = [10, 12, 14]
            marker_shapes = ['D', '*', 'o']
            marker_colors = ['r', 'y', 'b']
            line_color = ['indigo', 'g', 'black']
            marker_zorder = [9, 10, 8]
            marker_edge = [0.3, 0.5, 0.2]
            marker_edge_color = ['black', 'black', 'black']
            marker_transperency = [0.75, 1.0, 0.5]
            charts = [0] * self.source_data.num_inputs * self.source_data.num_outputs
            # print charts
            chart = -1
            means = self.source_data.get_mean_row()
            var_types_bool = self.get_variables_info('bool')
            # var_locations = self.get_variables_info('loc')
            var_map = self.get_variables_info('loc')
            normalized_means = get_three_normal_means(mother_study, means, var_map, var_types_bool)

            # def float_range(x, y, jump=1.0):
            #     """
            #     A generator quoted from http://j.mp/1V7BE5g by Akseli Palén
            #     @param x:Starting float value
            #     @param y:Ending float value
            #     @param jump:Step
            #     @return: yields a float function similar to range function
            #     """
            #     '''Range for floats. float range'''
            #     i = 0.0
            #     x = float(x)  # Prevent yielding integers.
            #     x0 = x
            #     epsilon = jump / 2.0
            #     yield x  # yield always first value
            #     while x + epsilon < y:
            #         i += 1.0
            #         x = x0 + i * jump
            #         yield x

            output_vars = self.source_data.output_variables
            in_norm = -1
            cur_var_num = -1
            finished_input_variables = []
            for in_var in range(self.source_data.num_inputs):
                cur_var = self.source_data.input_variables[in_var]
                cur_var_num += 1
                if cur_var not in finished_input_variables:
                    pages_titles.append("Effects of " + str(self.source_data.classified_titles[0][cur_var_num])
                                        + " on all output variables")
                    figure_number += 1
                    figure_page.append(plt.figure(figure_number))
                    if cur_var.data_type == 'Numeric':  # the input is numeric (continous x axis)
                        finished_input_variables.append(cur_var)
                        in_norm += 1  # the normalized number of output
                        stats = cur_var.get_basic_stats()  # [avg, min, max, std]
                        # # Base case
                        # outputs_base = self.ann.get_predictions(normalized_mean)
                        super_results = []
                        for rept in range(len(normalized_means)):
                            results_matrix = []
                            # print 'in_norm', in_norm
                            changed_input = copy.deepcopy(normalized_means[rept])
                            for vals in float_range(stats[1], stats[2], (stats[2] - stats[1]) / 20.):
                                changed_input[in_norm] = cur_var.get_normalized_value(vals)
                                outputs_temp = self.ann.get_predictions(changed_input)
                                tmp = []
                                finished_output_variables = []
                                for o, out in enumerate(outputs_temp):
                                    if var_map[1][o] not in finished_output_variables:
                                        if var_types_bool[1][o]:  # Numeric output
                                            tmp.append(output_vars[o].get_de_normalized_value(out))
                                            finished_output_variables.append(var_map[1][o])
                                        else:
                                            rep = var_map[1].count(var_map[1][o])
                                            tmp2 = outputs_temp[o: o + rep]
                                            # tmp2 = map(lambda x: 1 if x >= 0.5 else 0, tmp2)
                                            tmp.append(output_vars[o].get_de_normalized_value(tmp2))
                                            finished_output_variables.append(var_map[1][o])
                                results_matrix.append((vals, tmp))
                            super_results.append(results_matrix)

                        # Draw figures here
                        for o in range(num_outputs_original):
                            if var_types_bool[1][o]:  # Numeric output (lines values to values)
                                temp_xy = [[], []]
                                graph_grid_sub_adjusted = (graph_grid_sub[o][0] + 0, graph_grid_sub[o][1] + 0)
                                chart += 1
                                charts[chart] = plt.subplot2grid(graph_grid2, graph_grid_sub_adjusted, colspan=1)
                                for rept in range(len(normalized_means)):
                                    x = []
                                    y = []
                                    for i, res in enumerate(super_results[rept]):
                                        x.append(res[0])
                                        y.append(res[1][o])
                                    plt.plot(x, y, marker_shapes[rept], markersize=marker_sizes[rept],
                                             markeredgewidth=marker_edge[rept],
                                             markeredgecolor=marker_edge_color[rept],
                                             markerfacecolor=marker_colors[rept], fillstyle=None,
                                             zorder=marker_zorder[rept], alpha=marker_transperency[rept])

                                    plt.plot(x, y, linewidth=2, linestyle="-", c=line_color[rept],
                                             zorder=7)  # c=nesr_colors_dark[o]
                                    temp_xy[0].append(x)
                                    temp_xy[1].append(y)
                                # temp_xy123 =[]
                                # for m, member in enumerate(temp_xy[0][0]):
                                #     temp_tuple = [member]
                                #     for kn in range(len(temp_xy[1])):
                                #         temp_tuple.append(temp_xy[1][kn])
                                #     temp_xy123.append(tuple(temp_tuple))
                                temp_xy123 = [temp_xy[0][0]]
                                for kn in range(len(temp_xy[1])):
                                    temp_xy123.append(temp_xy[1][kn])

                                plt.xlabel(self.source_data.classified_titles[0][in_var], fontsize=14, weight='bold')
                                plt.ylabel(self.source_data.classified_titles[1][o], fontsize=14, weight='bold')
                                plt.title(self.source_data.classified_titles[0][in_var] +
                                          ' effect on ' + self.source_data.classified_titles[1][o],
                                          fontsize=16, weight='bold', color='maroon')
                                plt.grid(True)
                                data_save_lists.append(((self.source_data.classified_titles[0][in_var] +
                                                         ' effect on ' + self.source_data.classified_titles[1][o],
                                                         self.source_data.classified_titles[0][in_var],
                                                         'Scenario #1, Scenario #2, Scenario #3'),
                                                         temp_xy123))
                            else:  # Categoric output (bars values to cats)
                                temp_xy = [[], []]
                                out_var = self.source_data.output_variables[o]
                                no_match_y = len(out_var.members_indices)
                                graph_grid_sub_adjusted = (graph_grid_sub[o][0], graph_grid_sub[o][1])
                                chart += 1
                                charts[chart] = plt.subplot2grid(graph_grid2, graph_grid_sub_adjusted, colspan=1)

                                for rept in range(len(normalized_means)):
                                    x = []
                                    y = []
                                    for i, res in enumerate(super_results[rept]):
                                        x.append(res[0])
                                        if res[1][o] in out_var.members_indices:
                                            y.append(out_var.members_indices[res[1][o]])
                                        else:
                                            y.append(no_match_y)

                                    mod_x = [0]
                                    mod_x.extend(x)
                                    mod_x.append(0)

                                    mod_y = [-1]
                                    mod_y.extend(y)
                                    mod_y.append(no_match_y + 1)

                                    plt.plot(mod_x, mod_y, marker_shapes[rept], markersize=marker_sizes[rept],
                                             markeredgewidth=marker_edge[rept],
                                             markeredgecolor=marker_edge_color[rept],
                                             markerfacecolor=marker_colors[rept], fillstyle=None,
                                             zorder=marker_zorder[rept], alpha=marker_transperency[rept])

                                    temp_xy[0].append(mod_x)
                                    temp_xy[1].append(mod_y)

                                # temp_xy123 =[]
                                # for m, member in enumerate(temp_xy[0][0]):
                                #     temp_tuple = [member]
                                #     for kn in range(len(temp_xy[1])):
                                #         temp_tuple.append(temp_xy[1][kn])
                                #     temp_xy123.append(tuple(temp_tuple))
                                temp_xy123 = [temp_xy[0][0]]
                                for kn in range(len(temp_xy[1])):
                                    temp_xy123.append(temp_xy[1][kn])

                                major_yticks = np.arange(-1, len(out_var.members) + 2, 1)
                                charts[chart].set_yticks(major_yticks, minor=False)
                                y_labels = ['']
                                y_labels.extend(copy.deepcopy(out_var.members))
                                y_labels.extend(['*No match*', ''])
                                charts[chart].set_yticklabels(tuple(y_labels))

                                plt.xlabel(self.source_data.classified_titles[0][in_var], fontsize=14, weight='bold')
                                plt.ylabel(self.source_data.classified_titles[1][o], fontsize=14, weight='bold')
                                plt.title(self.source_data.classified_titles[0][in_var] +
                                          ' effect on ' + self.source_data.classified_titles[1][o],
                                          fontsize=16, weight='bold', color='maroon')
                                plt.grid(True)
                                data_save_lists.append(((self.source_data.classified_titles[0][in_var] +
                                                         ' effect on ' + self.source_data.classified_titles[1][o],
                                                         self.source_data.classified_titles[0][in_var],
                                                         'Scenario #1, Scenario #2, Scenario #3'),
                                                         temp_xy123))

                        plt.interactive(False)
                        plt.suptitle(pages_titles[figure_number], fontsize=25, weight='bold')
                        print figure_number,
                        mng = plt.get_current_fig_manager()
                        mng.window.showMaximized()
                        if not self.display_graph_windows:
                            plt.close()

                    else:  # categoric input variable
                        finished_input_variables.append(cur_var)
                        stats = cur_var.get_basic_stats()  # [for ctg it is just list of values]
                        super_results = []
                        for rept in range(len(normalized_means)):
                            results_matrix = []
                            changed_input = copy.deepcopy(normalized_means[rept])
                            for category in range(len(stats)):
                                changed_input[in_norm + category + 1] = 1
                                outputs_temp = self.ann.get_predictions(changed_input)
                                tmp = []
                                finished_output_variables = []
                                for o, out in enumerate(outputs_temp):
                                    if var_map[1][o] not in finished_output_variables:
                                        if var_types_bool[1][o]:  # Numeric output
                                            tmp.append(output_vars[o].get_de_normalized_value(out))
                                            finished_output_variables.append(var_map[1][o])
                                        else:
                                            rep = var_map[1].count(var_map[1][o])
                                            tmp2 = outputs_temp[o: o + rep]
                                            # tmp2 = map(lambda x: 1 if x >= 0.5 else 0, tmp2)
                                            tmp.append(output_vars[o].get_de_normalized_value(tmp2))
                                            finished_output_variables.append(var_map[1][o])
                                results_matrix.append((cur_var.members[category], tmp))
                            super_results.append(results_matrix)
                        # Draw figures here
                        for o in range(num_outputs_original):
                            if var_types_bool[1][o]:  # Numeric output (bars cats to values)
                                temp_xy = [[], []]
                                graph_grid_sub_adjusted = (graph_grid_sub[o][0] + 0, graph_grid_sub[o][1] + 0)
                                chart += 1
                                charts[chart] = plt.subplot2grid(graph_grid2, graph_grid_sub_adjusted, colspan=1)
                                # to determine the maximum y value

                                y = []
                                for rept in range(len(normalized_means)):
                                    for res in super_results[rept]:
                                        y.append(res[1][o])
                                max_y = (int(max(y) / 10) + 1) * 10


                                for rept in range(len(normalized_means)):
                                    x = []
                                    y = []
                                    for i, res in enumerate(super_results[rept]):
                                        x.append(res[0])
                                        y.append(res[1][o])

                                    ind_x = list(range(-1, len(x) + 1))

                                    mod_y = [0]
                                    mod_y.extend(y)
                                    # max_y = (int(max(y) / 10) + 1) * 10
                                    mod_y.extend([max_y])

                                    plt.plot(ind_x, mod_y, marker_shapes[rept], markersize=marker_sizes[rept],
                                             markeredgewidth=marker_edge[rept],
                                             markeredgecolor=marker_edge_color[rept],
                                             markerfacecolor=marker_colors[rept], fillstyle=None,
                                             zorder=marker_zorder[rept], alpha=marker_transperency[rept])

                                    temp_xy[0].append(ind_x)
                                    temp_xy[1].append(mod_y)

                                # temp_xy123 =[]
                                # for m, member in enumerate(temp_xy[0][0]):
                                #     temp_tuple = [member]
                                #     for kn in range(len(temp_xy[1])):
                                #         temp_tuple.append(temp_xy[1][kn])
                                #     temp_xy123.append(tuple(temp_tuple))
                                temp_xy123 = [temp_xy[0][0]]
                                for kn in range(len(temp_xy[1])):
                                    temp_xy123.append(temp_xy[1][kn])

                                major_xticks = np.arange(-1, len(super_results[0]) + 1, 1)
                                charts[chart].set_xticks(major_xticks, minor=False)

                                x_labels = ['']
                                x_labels.extend(copy.deepcopy(cur_var.members))
                                x_labels.append('')

                                charts[chart].set_xticklabels(tuple(x_labels))

                                # plt.xlabel('Input variable # ' + str(in_var), fontsize=16)
                                # plt.ylabel('Output variable # ' + str(o), fontsize=16)
                                # plt.title('Input-Output relationship' + str(o) + ' n')
                                plt.xlabel(self.source_data.classified_titles[0][in_var], fontsize=14, weight='bold')
                                plt.ylabel(self.source_data.classified_titles[1][o], fontsize=14, weight='bold')
                                plt.title(self.source_data.classified_titles[0][in_var] +
                                          ' effect on ' + self.source_data.classified_titles[1][o],
                                          fontsize=16, weight='bold', color='maroon')
                                plt.grid(True)
                                data_save_lists.append(((self.source_data.classified_titles[0][in_var] +
                                                         ' effect on ' + self.source_data.classified_titles[1][o],
                                                         self.source_data.classified_titles[0][in_var],
                                                         'Scenario #1, Scenario #2, Scenario #3'),
                                                         temp_xy123))
                            else:  # Categoric output (bars cats to cats)
                                temp_xy = [[], []]
                                out_var = self.source_data.output_variables[o]
                                no_match_y = len(out_var.members_indices)
                                graph_grid_sub_adjusted = (graph_grid_sub[o][0], graph_grid_sub[o][1])
                                chart += 1
                                # print chart
                                charts[chart] = plt.subplot2grid(graph_grid2, graph_grid_sub_adjusted, colspan=1)
                                for rept in range(len(normalized_means)):
                                    x = []
                                    y = []
                                    for i, res in enumerate(super_results[rept]):

                                        x.append(cur_var.members_indices[res[0]])

                                        if res[1][o] in out_var.members_indices:
                                            y.append(out_var.members_indices[res[1][o]])
                                        else:
                                            y.append(no_match_y)

                                    ind_x = list(range(-1, len(x) + 1))
                                    # np.array(range(-1, len(x) + 1))  # the x locations for the groups
                                    mod_y = [-1]
                                    mod_y.extend(y)
                                    mod_y.append(no_match_y + 1)

                                    plt.plot(ind_x, mod_y, marker_shapes[rept], markersize=marker_sizes[rept],
                                             markeredgewidth=marker_edge[rept],
                                             markeredgecolor=marker_edge_color[rept],
                                             markerfacecolor=marker_colors[rept], fillstyle=None,
                                             zorder=marker_zorder[rept], alpha=marker_transperency[rept])

                                    temp_xy[0].append(ind_x)
                                    temp_xy[1].append(mod_y)

                                # temp_xy123 =[]
                                # for m, member in enumerate(temp_xy[0][0]):
                                #     temp_tuple = [member]
                                #     for kn in range(len(temp_xy[1])):
                                #         temp_tuple.append(temp_xy[1][kn])
                                #     temp_xy123.append(tuple(temp_tuple))
                                temp_xy123 = [temp_xy[0][0]]
                                for kn in range(len(temp_xy[1])):
                                    temp_xy123.append(temp_xy[1][kn])

                                major_xticks = np.arange(-1, len(cur_var.members) + 1, 1)
                                charts[chart].set_xticks(major_xticks, minor=False)
                                x_labels = ['']
                                x_labels.extend(copy.deepcopy(cur_var.members))
                                x_labels.append('')
                                charts[chart].set_xticklabels(tuple(x_labels))

                                major_yticks = np.arange(-1, len(out_var.members) + 2, 1)
                                charts[chart].set_yticks(major_yticks, minor=False)
                                y_labels = ['']
                                y_labels.extend(copy.deepcopy(out_var.members))
                                y_labels.extend(['*No match*', ''])
                                charts[chart].set_yticklabels(tuple(y_labels))

                                # plt.xlabel('Input variable # ' + str(in_var), fontsize=16)
                                # plt.ylabel('Output variable # ' + str(o), fontsize=16)
                                # plt.title('Input-Output relationship' + str(o) + ' n')
                                plt.xlabel(self.source_data.classified_titles[0][in_var], fontsize=14, weight='bold')
                                plt.ylabel(self.source_data.classified_titles[1][o], fontsize=14, weight='bold')
                                plt.title(self.source_data.classified_titles[0][in_var] +
                                          ' effect on ' + self.source_data.classified_titles[1][o],
                                          fontsize=16, weight='bold', color='maroon')
                                plt.grid(True)
                                data_save_lists.append(((self.source_data.classified_titles[0][in_var] +
                                                         ' effect on ' + self.source_data.classified_titles[1][o],
                                                         self.source_data.classified_titles[0][in_var],
                                                         'Scenario #1, Scenario #2, Scenario #3'),
                                                         temp_xy123))

                        plt.interactive(False)
                        # plt.tight_layout()
                        plt.suptitle(pages_titles[figure_number], fontsize=25, weight='bold')

                        save_graph_data('Grf5_Relationships.csv', data_save_lists)

                        print figure_number,
                        mng = plt.get_current_fig_manager()
                        mng.window.showMaximized()
                        if not self.display_graph_windows:
                            plt.close()
                        in_norm += len(stats)  # - 1
            pass

        # this for loop to partition the routine to several parts
        w12, w23, b12, b23, structure = 0, 0, 0, 0, 0
        for section in range(8):
            if section == 0:
                # definitions and basic information
                # set of 20 colors, dark then light
                nesr_colors = {0: 'blue', 1: 'gold', 2: 'green', 3: 'yellow', 4: 'purple', 5: 'white',
                               6: 'red', 7: 'bisque', 8: 'maroon', 9: 'aqua', 10: 'black', 11: 'lime',
                               12: 'indigo', 13: 'fuchsia', 14: 'darkcyan', 15: 'gold', 16: 'navi',
                               17: 'khaki', 18: 'saddlebrown', 19: 'lightsteelblue'}
                # set of dark colors only
                nesr_colors_dark = {}
                for i in range(0, 20, 2):
                    nesr_colors_dark[i / 2] = nesr_colors[i]
                num_inputs_original = self.source_data.num_inputs
                num_outputs_original = self.source_data.num_outputs
                print '...Now drawing graphs',
            elif section == 1:
                # Displaying errors and -ve contribution chart
                # $^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$

                # ===============================================================

                # Graphing the training error
                # cur_plt = 0
                try:
                    figure_page.append(plt.figure(figure_number))
                    # cur_plt = 1
                    # =-.-=-.-=-.-=-.-=-.-=-.-=-.-=-.-=-.-=-.-=
                    draw_cost_function()  # =-.-=-.-=-.-=-.-=
                    # =-.-=-.-=-.-=-.-=-.-=-.-=-.-=-.-=-.-=-.-=
                    print figure_number,
                    mng = plt.get_current_fig_manager()
                    mng.window.showMaximized()
                    if not self.display_graph_windows:
                        plt.close()
                except Exception, err:
                    print "\nError graphing the training error, panel 1."
                    print err
            elif section == 2:
                # $^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$
                try:
                    figure_number += 1
                    figure_page.append(plt.figure(figure_number))

                    # Graphing the artificial neural network diagram

                    # =-.-=-.-=-.-=-.-=-.-=-.-=-.-=-.-=-.-=-.-=
                    w12, w23, b12, b23, structure = draw_full_ann()  # =-.-=-.-=-.-=-.-=
                    # =-.-=-.-=-.-=-.-=-.-=-.-=-.-=-.-=-.-=-.-=

                    print figure_number,
                    mng = plt.get_current_fig_manager()
                    mng.window.showMaximized()
                    if not self.display_graph_windows:
                        plt.close()
                except Exception, err:
                    print "\nError graphing the artificial neural network diagram, panel 2."
                    print err
            elif section == 3:
                # $^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$
                try:
                    figure_number += 1
                    figure_page.append(plt.figure(figure_number))
                    # ===============================================================
                    # Graphing the artificial BRIEF neural network diagram

                    # =-.-=-.-=-.-=-.-=-.-=-.-=-.-=-.-=-.-=-.-=
                    draw_brief_ann(w12, w23, b12, b23, structure)  # =-.-=-.-=-.-=-.-=
                    # =-.-=-.-=-.-=-.-=-.-=-.-=-.-=-.-=-.-=-.-=

                    print figure_number,
                    mng = plt.get_current_fig_manager()
                    mng.window.showMaximized()
                    if not self.display_graph_windows:
                        plt.close()
                except Exception, err:

                    print "\nError graphing the BRIEF neural network diagram, panel 3."
                    print err

            elif section == 4:
                # ===============================================================
                # =========================
                # The next page of plots.
                # =========================
                # $^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$
                try:
                    figure_number += 1
                    figure_page.append(plt.figure(figure_number, facecolor='white'))
                    # ===============================================================
                    # Graphing relative importance (-ve)

                    # =-.-=-.-=-.-=-.-=-.-=-.-=-.-=-.-=-.-=-.-=
                    draw_relative_importance()  # =-.-=-.-=-.-=-.-=
                    # =-.-=-.-=-.-=-.-=-.-=-.-=-.-=-.-=-.-=-.-=

                    print figure_number,
                    mng = plt.get_current_fig_manager()
                    mng.window.showMaximized()
                    if not self.display_graph_windows:
                        plt.close()
                except Exception, err:
                    print "\nError graphing the relative_importance diagram, panel 4."
                    print err
            elif section == 5:
                # ===============================================================
                # =========================
                # The next page of plots.
                # =========================
                # $^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$
                try:
                    figure_number += 1
                    figure_page.append(plt.figure(figure_number))

                    # =-.-=-.-=-.-=-.-=-.-=-.-=-.-=-.-=-.-=-.-=
                    out_graph_o, graph_grid_sub, graph_grid2, scatter_categoric = draw_prediction_cloud()  # =-.-=-.-=-.-=-.-=
                    # =-.-=-.-=-.-=-.-=-.-=-.-=-.-=-.-=-.-=-.-=

                    print figure_number,
                    mng = plt.get_current_fig_manager()
                    mng.window.showMaximized()
                    if not self.display_graph_windows:
                        plt.close()
                except Exception, err:
                    print "\nError graphing the prediction clouds, panel 5."
                    print err
            elif section == 6:
                # $^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$^$
                try:
                    figure_number += 1
                    figure_page.append(plt.figure(figure_number))

                    # =-.-= -.-= -.-= -.-= -.-= -.-= -.-= -.-= -.-= -.-=
                    draw_real_vs_forecasted()  # =-.-=-.-=-.-=-.-=
                    # =-.-=-.-=-.-=-.-=-.-=-.-=-.-=-.-=-.-=-.-=

                    print figure_number,
                    mng = plt.get_current_fig_manager()
                    mng.window.showMaximized()
                    if not self.display_graph_windows:
                        plt.close()
                except Exception, err:
                    print "\nError graphing the real vs. forecasted graphs, panel 6."
                    print err

            elif section == 7:
                try:
                    # =-.-= -.-= -.-= -.-= -.-= -.-= -.-= -.-= -.-= -.-=
                    draw_parametric_graphs(figure_number, self)
                    # =-.-= -.-= -.-= -.-= -.-= -.-= -.-= -.-= -.-= -.-=
                except Exception, err:
                    print "\nError graphing one of the parametric graphs, panel 7+"
                    print err

                pass

        if initial_time != 0:
            print "\nElapsed time throughout the study: \n **Execluding time of 'showing' the graphs**\n", \
                elapsed_time(initial_time, time.time())

        # def shorten_time_stamp(time_stamp_text):
        #     short_stamp = ""
        #     split_stamp = map(lambda x: int(x), re.findall('.{1,2}', time_stamp_text))
        #     short_stamp += chr(split_stamp[0] + 50)
        #     short_stamp += str(split_stamp[1]) if split_stamp[1] < 10 else chr(split_stamp[1] + 55)
        #     short_stamp += str(split_stamp[2]) if split_stamp[2] < 10 else chr(split_stamp[2] + 55)
        #     short_stamp += str(split_stamp[3]) if split_stamp[3] < 10 else chr(split_stamp[3] + 55)
        #     short_stamp += str(split_stamp[4])
        #     short_stamp += str(split_stamp[5])
        #     return short_stamp

        # Closing The log file of the console.
        self.log_file.close()
        # Converting the text file to pdf
        time_stamp = self.time_stamp
        console_pdf_name = 'NrCh_PdfLog_' +  time_stamp[-4:] + '.pdf'
        # console_pdf_file = self.convert_text_to_pdf("NesrLog.txt", console_pdf_name)
        self.convert_text_to_pdf(self.new_folder_path + '\\' + "NrChLog.txt", console_pdf_name)
        # Making the pdf of all charts
        pdf_name = 'NrCh_OutputCharts_' + time_stamp[-4:] + '.pdf'

        # # Create a folder and put all files in:
        # current_folder = os.getcwd()
        # directory_name = 'NrCh' + time_stamp
        # new_folder_path = current_folder + "\\" + directory_name
        # if not os.path.exists(new_folder_path):
        #     os.makedirs(new_folder_path)

        with PdfPages(pdf_name) as pdf:
            for page_num, page in enumerate(figure_page):
                pdf.attach_note(pages_titles[page_num])
                pdf.savefig(page)
            # We can also set the file's metadata via the PdfPages object:
            d = pdf.infodict()
            d['Title'] = 'NeuroCharter, A python open source neural networks simulator'
            d['Author'] = 'Dr. Mohammad Elnesr'  # u'Jouni K. Sepp\xe4nen'
            d['Subject'] = 'NeuroCharter Simulation Results'
            d['Keywords'] = 'Neural networks AWC-KSU King Saud University Alamoudi Water Chair'
            d['CreationDate'] = dt(2016, 4, 13)
            d['ModDate'] = dt.today()

        # Moving files
        def move_file_to_folder(old_name, new_name, rename=False):
            destination = self.new_folder_path
            source = self.new_folder_path if rename else self.current_folder
            try:
                os.rename(source + "\\" + old_name, destination + "\\" + new_name)
            except:
                # rename if exist with a random number
                rand_num  = str(int(random.random() * 100000))
                os.rename(source + "\\" + old_name, destination + "\\" + rand_num + new_name)

        # Saving Graph Data


        move_file_to_folder (console_pdf_name, console_pdf_name)
        move_file_to_folder (pdf_name, pdf_name)
        # Rename Other files

        sub_stamp = time_stamp[-4:]

        move_file_to_folder ("NrChLog.txt", "NrCh__Log__" +  sub_stamp + ".txt", rename=True)
        move_file_to_folder ("PredictionClouds.csv", "NrCh_Clouds_" +  sub_stamp + ".csv", rename=True)
        move_file_to_folder ("NeuroCharterNet.nsr", "NrCh_StoredANN_" +  sub_stamp + ".nsr", rename=True)
        move_file_to_folder ("Outputs.txt", "NrCh_Outputs_" +  sub_stamp + ".txt", rename=True)
        move_file_to_folder ("Weights.csv", "NrCh_Weights_" +  sub_stamp + ".csv", rename=True)
        move_file_to_folder ("NormalizedData.csv", "NrCh_NormData_" +  sub_stamp + ".csv", rename=True)

        charts_data_files = [("Grf1_Costs.csv", "NrCh_C1_Costs_"),
                             ("Grf2_Importance.csv", "NrCh_C2_Importance_"),
                             ("Grf3_Clouds.csv", "NrCh_C3_Clouds_"),
                             ("Grf4_Predictions.csv", "NrCh_C4_Predictions_"),
                             ("Grf5_Relationships.csv", "NrCh_C5_Studies_")]
        for data_file in charts_data_files:
            try:
                move_file_to_folder (data_file[0], data_file[1] +  sub_stamp + ".csv", rename=True)
            except:
                pass

        if self.display_graph_pdf:
            Popen(self.new_folder_path + "\\" + pdf_name, shell=True)
        if self.display_graph_windows:
            plt.show()

        pass


    @staticmethod
    def convert_text_to_pdf(text_file_name, pdf_file_name, maximum_text_width=85):
        """
        A function to convert text file to pdf file
        @param text_file_name: the text file you want to convert to pdf
        @param pdf_file_name: the name of the pdf file that will be crated
        @param maximum_text_width: maximum number of characters allowed in a line.
        @return: None
        """
        # from reportlab.pdfbase import pdfmetrics
        # from reportlab.pdfbase.ttfonts import TTFont
        #
        # pdfmetrics.registerFont(TTFont('Courier New Regular', 'COUR_0.ttf'))
        # pdfmetrics.registerFont(TTFont('Courier New Italic', 'COURI_0.ttf'))
        # pdfmetrics.registerFont(TTFont('Courier New Bold', 'COURBD_0.ttf'))
        # pdfmetrics.registerFont(TTFont('Courier New Bold Italic', 'COURBI_0.ttf'))

        temp_file_read = open(text_file_name, "r")  # text file I need to convert
        lines = temp_file_read.readlines()
        temp_file_read.close()
        i = 750
        line_number = 0

        # Open an empty pdf file
        pdf_file = canvas.Canvas(pdf_file_name)
        # pdf_file.setFont('Courier New Regular', 12)

        # finding actual number of lines
        actual_lines = []
        for line in lines:
            line = line.replace('\n', '')
            if len(line) < maximum_text_width:
                actual_lines.append(line)
            else:
                line_list = line.split(",")
                temp_line = ""
                for r in range(len(line_list)):
                    if len(temp_line) + len(str(line_list[r]) + ",") < maximum_text_width:
                        temp_line += str(line_list[r])
                        if r != len(line_list) - 1:
                            temp_line += ","
                    else:
                        actual_lines.append(temp_line)
                        temp_line = "     " + str(line_list[r])
                        if r != len(line_list) - 1:
                            temp_line += ","
                else:
                    actual_lines.append(temp_line)
        # Start printing lines to the pdf file
        while line_number < len(actual_lines):
            if line_number - len(actual_lines) < 60:  # to finish the page every 60 line
                i = 750  # the initial Y location
                for sub_line in actual_lines[line_number:line_number + 60]:
                    pdf_file.drawString(15, i, sub_line.strip())
                    line_number += 1
                    i -= 12
                pdf_file.showPage()
            else:
                i = 750  # the initial Y location
                for sub_line in actual_lines[line_number:]:
                    pdf_file.drawString(15, i, sub_line.strip())
                    line_number += 1
                    i -= 12
                pdf_file.showPage()
            pdf_file.save()

    @staticmethod
    def find_suitable_grid(num):
        """
        A list to find the suitable grid of graphs in each figure
        @param num: the number of charts needed to be plotted
        @return: a tuple of (tuple of maximum grid, tuple of current location)
                    example: if num = 11, then it returns ((3, 4), (2, 3))
        """
        suit = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1), 4: (0, 2), 5: (1, 2), 6: (2, 0),
                7: (2, 1), 8: (2, 2), 9: (0, 3), 10: (1, 3), 11: (2, 3), 12: (3, 0), 13: (3, 1),
                14: (3, 2), 15: (3, 3), 16: (0, 4), 17: (1, 4), 18: (2, 4), 19: (3, 4), 20: (4, 0),
                21: (4, 1), 22: (4, 2), 23: (4, 3), 24: (4, 4)}

        grid_dict = {0: (1, 1), 1: (1, 2), 2: (2, 2), 3: (2, 2), 4: (2, 3),
                     5: (2, 3), 6: (3, 3), 7: (3, 3), 8: (3, 3), 9: (3, 4),
                     10: (3, 4), 11: (3, 4), 12: (4, 4), 13: (4, 4), 14: (4, 4), 15: (4, 4),
                     16: (4, 5), 17: (4, 5), 18: (4, 5), 19: (4, 5), 20: (5, 5), 21: (5, 5),
                     22: (5, 5), 23: (5, 5), 24: (5, 5)}
        # tmp = [2, int(round(num / 2, 0))]
        # for i in range(2, num):
        #     if i in tmp:
        #         break
        #     elif num % i == 0:
        #         if i not in tmp:
        #             tmp.append(i)
        # dim1 = max (tmp)
        # dim2 = round(num / dim1, 0)
        # grid = (min(dim1, dim2), max(dim1, dim2))
        # for i in range(num):
        #     suit[i] = (i % grid[0], int(i / grid[0]))
        return grid_dict[num - 1], suit

    def print_to_console(self, correlation_coefficients, errors_collection,
                         matrix_of_sum_of_errors, stopping_epoch, maximum_epochs):
        """
        A function to print some results to the console
        @param correlation_coefficients: of each variable
        @param errors_collection: for each variable
        @param matrix_of_sum_of_errors: total error, MSE, RMSE
        @param stopping_epoch: on convergence
        @param maximum_epochs: as selected
        """
        ann = self.ann
        print ann
        self.print_info(0)  # self.print_net_weights(ann)
        self.print_info(1)  # self.print_relative_importance(ann)
        self.print_info(2, correlation_coefficients)  # self.print_correlation_coefficient(correlation_coefficients)
        print_matrix('MSE per output', errors_collection[2])
        print_matrix('RMSE per output', map((lambda x: x ** 0.5), errors_collection[2]))
        print_matrix('ANN total error per output', errors_collection[1])
        print_matrix('Final ANN total error, MSE, RMSE ', matrix_of_sum_of_errors)
        print "Stopped after maximum epochs" if stopping_epoch >= maximum_epochs - 1 else \
            "Reached maximum tolerence after " + str(stopping_epoch) + " epochs"
        pass

    def retrain_net(self, randomize_biases=True):

        """
        Perform retraining with the same weights and with different biases
        @param randomize_biases: if we should randomize biases or not
                Note: if False, then the networc results will remain as is
        @return: a trained net
        """

        def read_weights_from_txt(study):
            """
            Reads the weights only from stored text file
            @param study: the study where the graph belongs to
            @return: tuple of (weights i_h, weights h_o, bias i_h, bias h_o)
            """
            data = np.array(list(csv.reader(open(self.new_folder_path + '\\' + 'Weights.csv', "rb"), delimiter=','))).astype('float')
            weights_sets = []
            for case in data:
                weights_sets.append(list(case))

            weights_i_h = weights_sets[:study.num_inputs_normalized]
            weights_h_o = weights_sets[
                          study.num_inputs_normalized:study.num_inputs_normalized + study.num_outputs_normalized]
            bias_i_h = weights_sets[
                       study.num_inputs_normalized + study.num_outputs_normalized: study.num_inputs_normalized +
                                                                                   study.num_outputs_normalized + 1][
                0]  # Only one line for biases
            bias_h_o = weights_sets[study.num_inputs_normalized + study.num_outputs_normalized + 1:][0][
                       :study.num_outputs_normalized]  # Only one line for biases

            weights_i_h = transpose_matrix(weights_i_h)
            weights_h_o = transpose_matrix(weights_h_o)
            i_h = []
            for i in weights_i_h:
                for j in i:
                    i_h.append(j)
            h_o = []
            for i in weights_h_o:
                for j in i:
                    h_o.append(j)
            return i_h, h_o, bias_i_h, bias_h_o

        ann = self.ann
        # data_set  = self.normalized_data
        structure = ann.get_structure()
        active_func = ann.get_activation_functions()
        weights = read_weights_from_txt(self)
        if randomize_biases:
            nnn = ann
            # nnn = NeuralNetwork(structure[0], structure[1], structure[2],
            #                     activation_functions=active_func,
            #                     hidden_layer_weights=weights[0],
            #                     output_layer_weights=weights[1])
        else:
            nnn = NeuralNetwork(structure[0], structure[1], structure[2],
                                activation_functions=active_func,
                                hidden_layer_weights=weights[0],
                                hidden_layer_bias=weights[2],
                                output_layer_weights=weights[1],
                                output_layer_bias=weights[3],
                                parent_study=self)
        return self.train_net(nnn)

    def find_best_activation_function(self, data_file, num_inputs, num_hidden, num_outputs, try_functions):
        # , tolerance=0.0001):
        # maximum_epochs = 500

        # @staticmethod
        """
        Seeks the best activation function of the network
        @param data_file: the data file to train the network with
        @param num_inputs: number of input neurons
        @param num_hidden: number of hidden neurons
        @param num_outputs: number of output neurons
        @param try_functions: the functions we should try
        @return: It just prints the results to console.
        """

        def convert_function(function):
            """
            convert the requested function number to the function name
            @param function: the number of the function
            @return: the function name
            """
            act_fun = list(function)
            acf = ACTIVATION_FUNCTIONS
            f = [acf[act_fun[0]]]
            # temp = []
            for i in act_fun[1]:
                f.append(acf[i])
            # f.append(temp)
            return f

        results = []
        if not isinstance(try_functions, tuple):
            return 'No functions to select from'
        # print list(itertools.combinations(try_functions,3))
        # print list(itertools.permutations(try_functions))
        trials = [p for p in itertools.product(try_functions, repeat=3)]
        print len(trials), '\n', trials
        functions = []
        for trl in trials:
            trial = list(trl)
            temp = [trial[0], tuple(trial[1:])]
            functions.append(tuple(temp))
        print functions

        for i, function in enumerate(functions):
            start_time = time.time()
            structure = (num_inputs, num_hidden, num_outputs)
            ann, training_sets = self.create_net(data_file, structure)
            _, epochs, correlation_coefficients, time_elapsed = self.train_net()
            errors_collection = ann.calculate_total_error(training_sets)
            ecl = errors_collection  # To rename the variable for easier writing
            m_err = [ecl[0], sum(ecl[2]), sum(map((lambda x: x ** 0.5), ecl[2]))]  # matrix_of_sum_of_errors
            average_correlation = sum(correlation_coefficients) / len(correlation_coefficients)
            elapsed_time = time.time() - start_time
            case = [convert_function(function), i + 1, m_err[0], m_err[2],
                    average_correlation, epochs, elapsed_time]
            results.append(case)
            self.save_to_file(2, results)
            # self.store_structure(results, clear_file=True, file_name='Functions.txt')
            print i + 1, m_err[0], epochs, elapsed_time
        pass

    def network_save(self):
        """
        Saves the study to an encrypted format in order to retrieve it again if needed to predict.
        """
        shelf = shelve.open(self.new_folder_path + '\\' + 'NeuroCharterNet.nsr')
        # shelf['theNetwork'] = self.ann
        # shelf['theData'] = self.source_data
        # saving the study parameters
        shelf['activation_functions'] = self.activation_functions
        shelf['adapt_learning_rate'] = self.adapt_learning_rate
        shelf['annealing_value'] = self.annealing_value
        shelf['categorical_extra_divisor'] = self.categorical_extra_divisor
        shelf['data_file_has_brief_titles'] = self.data_file_has_brief_titles
        shelf['data_file_has_titles'] = self.data_file_has_titles
        shelf['data_partition'] = self.data_partition
        shelf['data_style'] = self.source_data.data_style
        shelf['display_graph_pdf'] = self.display_graph_pdf
        shelf['display_graph_windows'] = self.display_graph_windows
        shelf['find_activation_function'] = self.find_activation_function
        shelf['layer_size_range'] = self.layer_size_range
        shelf['learning_rate'] = self.learning_rate
        shelf['master_error_list'] = self.master_error_list
        shelf['maximum_epochs'] = self.maximum_epochs
        # shelf['normalized_data'] = self.normalized_data
        shelf['refresh_weights_after_determining_structure'] = self.refresh_weights_after_determining_structure
        shelf['start_time'] = self.start_time
        shelf['structure'] = self.structure
        shelf['tolerance'] = self.tolerance
        shelf['try_different_structures'] = self.try_different_structures
        shelf['validation_epochs'] = self.validation_epochs

        # saving variables parameters
        shelf['num_inputs'] = self.source_data.num_inputs
        shelf['num_outputs'] = self.source_data.num_outputs
        shelf['out_var_bool'] = self.get_variables_info('bool')[1]
        var_info = []
        for i, var in enumerate(self.source_data.input_variables):
            if var.data_type == 'Numeric':
                var_info.append([i, var.name, var.brief, var.data_type, var.min, var.max, var.avg, var.stdev])
            else:
                var_info.append([i, var.name, var.brief, var.data_type, var.num_categories, var.unique_values,
                                 var.normalized_lists, var.members_indices])
        for i, info in enumerate(var_info):
            shelf['input_var' + str(i)] = info

        var_info = []
        for i, var in enumerate(self.source_data.output_variables):
            if var.data_type == 'Numeric':
                var_info.append([i, var.name, var.brief, var.data_type, var.min, var.max, var.avg, var.stdev])
            else:
                var_info.append([i, var.name, var.brief, var.data_type, var.num_categories, var.unique_values,
                                 var.normalized_lists, var.members_indices])
        for i, info in enumerate(var_info):
            shelf['output_var' + str(i)] = info

        # saving ann weights
        weights_of_i_h, weights_of_h_o, bias_of_i_h, bias_of_h_o = self.get_network_weights()
        shelf['weights_of_i_h'] = weights_of_i_h
        shelf['weights_of_h_o'] = weights_of_h_o
        shelf['bias_of_i_h'] = bias_of_i_h
        shelf['bias_of_h_o'] = bias_of_h_o

        shelf.close()


        pass

    def network_load(self, previous_study_data_file='NeuroCharterNet.nsr'):
        """
        loads a previously saved network to use it in predictions
        @param previous_study_data_file: the stored encripted file
        """
        shelf = shelve.open(previous_study_data_file)

        # self.ann = shelf['theNetwork']
        # self.source_data = shelf['theData']

        # self.main_normalized_data = self.source_data.normalized_data

        self.activation_functions = shelf['activation_functions']
        self.adapt_learning_rate = shelf['adapt_learning_rate']
        self.annealing_value = shelf['annealing_value']
        self.categorical_extra_divisor = shelf['categorical_extra_divisor']
        self.data_file_has_brief_titles = shelf['data_file_has_brief_titles']
        self.data_file_has_titles = shelf['data_file_has_titles']
        self.data_partition = shelf['data_partition']
        self.data_style = shelf['data_style']

        self.display_graph_pdf = shelf['display_graph_pdf']
        self.display_graph_windows = shelf['display_graph_windows']
        self.find_activation_function = shelf['find_activation_function']
        self.layer_size_range = shelf['layer_size_range']
        self.learning_rate = shelf['learning_rate']
        self.master_error_list = shelf['master_error_list']
        self.maximum_epochs = shelf['maximum_epochs']
        # self.normalized_data = shelf['normalized_data']
        self.refresh_weights_after_determining_structure = shelf['refresh_weights_after_determining_structure']
        self.start_time = shelf['start_time']
        self.structure = shelf['structure']
        self.tolerance = shelf['tolerance']
        self.try_different_structures = shelf['try_different_structures']
        self.validation_epochs = shelf['validation_epochs']

        self.num_inputs_normalized = self.structure[0]
        self.num_outputs_normalized = self.structure[2]

        # getting variables parameters
        self.num_inputs = shelf['num_inputs']
        self.num_outputs = shelf['num_outputs']

        var_info_input = []
        for i in range(self.num_inputs):
            temp = shelf['input_var' + str(i)]
            var_info_input.append(temp)
        var_info_output = []
        for i in range(self.num_outputs):
            temp = shelf['output_var' + str(i)]
            var_info_output.append(temp)

        self.temporary['var_info_input'] = var_info_input
        self.temporary['var_info_output'] = var_info_output

        # getting ann weights
        self.temporary['weights_of_i_h'] = shelf['weights_of_i_h']
        self.temporary['weights_of_h_o'] = shelf['weights_of_h_o']
        self.temporary['bias_of_i_h'] = shelf['bias_of_i_h']
        self.temporary['bias_of_h_o'] = shelf['bias_of_h_o']
        self.temporary['out_var_bool'] = shelf['out_var_bool']

        shelf.close()

        pass

# t1 = time.time()
# study1 = Study('dataN.csv', 'full run', num_outputs=4 , tolerance=0.001, learning_rate=0.45, maximum_epochs=500)
# study1.run()
# study2 = Study('dataN.csv', 'validation run', num_outputs=4, data_partition=(60, 25),
#                tolerance=0.0001, learning_rate=0.45, maximum_epochs=500)
# study2.run()
# study3 = Study('dataN.csv', 'optimization', num_outputs=4, data_partition=(60, 25),
#                tolerance=0.001, learning_rate=0.45, maximum_epochs=1000,
#                refresh_weights_after_determining_structure=False, layer_size_range=(0.5, 2, 1),
#                find_activation_function=False, validation_epochs=30, start_time=t1, adapt_learning_rate=True)
# study4 = Study('dataSW2.csv', 'optimization', num_outputs=3, data_partition=(65, 15),
#                tolerance=0.0001, learning_rate=0.3, maximum_epochs=10000,
#                refresh_weights_after_determining_structure=False, layer_size_range=(0.75, 1.25, 1),
#                find_activation_function=False, validation_epochs=30, start_time=t1, adapt_learning_rate=True)

# Study('dataNT.csv', 'validation run', num_outputs=4, data_partition=(65, 15),
#                tolerance=0.001, learning_rate=0.4, maximum_epochs=300,
#                refresh_weights_after_determining_structure=False, layer_size_range=(0.75, 1.25, 1),
#                find_activation_function=False, validation_epochs=30, start_time=t1,
#                adapt_learning_rate=False, annealing_value=2000,
#                display_graph_windows=False, display_graph_pdf=True, categorical_extra_divisor=2,
#                data_file_has_titles=True, data_file_has_brief_titles=True)

# study6 = Study('QueryN.csv', "query", previous_study_data_file="NeuroCharterNet.nsr", start_time=t1)
# from NeuroCharter import *

# Study('dataNT.csv', 'cross validation', num_outputs=4, data_partition=(65, 15),
#       tolerance=0.001, learning_rate=0.4, maximum_epochs=300,
#       adapt_learning_rate=False, annealing_value=2000,
#       display_graph_windows=True, display_graph_pdf=True,
#       data_file_has_titles=True, data_file_has_brief_titles=True)


# Study('TestSW_data.csv', 'cross validation', num_outputs=3, data_partition=(75, 15),
#       tolerance=0.001, learning_rate=0.4, maximum_epochs=8000,
#       adapt_learning_rate=False, annealing_value=2000,
#       display_graph_windows=False, display_graph_pdf=True,
#       data_file_has_titles=True, data_file_has_brief_titles=True,
#       minimum_slope_to_consider_overfitting=3, number_of_epochs_for_overfit_check=10)


# Study('dataSW3.csv', 'cross validation', num_outputs=3, data_partition=(75, 15),
#       tolerance=0.000001, learning_rate=0.4, maximum_epochs=3000,
#       adapt_learning_rate=False, annealing_value=2000,
#       display_graph_windows=False, display_graph_pdf=True,
#       data_file_has_titles=True, data_file_has_brief_titles=True,
#       minimum_slope_to_consider_overfitting=2, number_of_epochs_for_overfit_check=10)

# Study(purpose= "advanced query", previous_study_data_file="NeuroCharterNet.nsr", start_time=t1,
#       input_parameters_suggested_values= (4, ('VGN', 'KSG'), [50, 150, 50], [15, 19, 2],
#                                           6.2, [-.5, 1, 0.5], [0.25, 1, 0.25], 0.047))

# Study(purpose= "advanced query", previous_study_data_file="NeuroCharterNet.nsr", start_time=t1,
#       input_parameters_suggested_values= ([1, 8, 2], ('VGN', 'KSG'), [30, 360, 30], [11, 19, 2],
#                                           6.2, [-.97, 0.97, 0.25], [0.25, 1, 0.25], 0.047))
# from NeuroCharter import *

# Study('Method6.5.csv', 'cross validation', num_outputs=3, data_partition=(75, 15),
#       tolerance=0.000001, learning_rate=0.4, maximum_epochs=650,
#       adapt_learning_rate=False, annealing_value=2000,
#       display_graph_windows=False, display_graph_pdf=False,
#       data_file_has_titles=True, data_file_has_brief_titles=True,
#       minimum_slope_to_consider_overfitting=2, number_of_epochs_for_overfit_check=10,
#       relative_importance_method="M")



# Study('datan.csv', 'cross validation', num_outputs=4, data_partition=(75, 15),
#
# for study in range(2, 5):
#     filename = 'M%02d.csv' % study
#     prefix = 'M%02d_'% study
#     # print filename, prefix
#
#     Study(filename, 'cross validation', num_outputs=3, data_partition=(80, 5),
#           tolerance=0.000001, learning_rate=0.4, maximum_epochs=500,
#           adapt_learning_rate=False, annealing_value=2000,
#           display_graph_windows=False, display_graph_pdf=False,
#           data_file_has_titles=True, data_file_has_brief_titles=True,
#           minimum_slope_to_consider_overfitting=2, number_of_epochs_for_overfit_check=10,
#           relative_importance_method="M", folder_prefix=prefix )

methods = [('M01', '11110111001100001110'), ('M02', '11110111001100001101'), ('M03', '11101111001100001110'),
           ('M04', '11101111001100001101'), ('M05', '11110101110000001110'), ('M06', '11110101110000001101'),
           ('M07', '11101101110000001110'), ('M08', '11101101110000001101'), ('M09', '11110111011100001110'),
           ('M10', '11110111011100001101'), ('M11', '11101111011100001110'), ('M12', '11101111011100001101'),
           ('M13', '11110111001100011110'), ('M14', '11110111001100011101'), ('M15', '11101111001100011110'),
           ('M16', '11101111001100011101'), ('M17', '11110101110000011110'), ('M18', '11110101110000011101'),
           ('M19', '11101101110000011110'), ('M20', '11101101110000011101'), ('M21', '11110111011100011110'),
           ('M22', '11110111011100011101'), ('M23', '11101111011100011110'), ('M24', '11101111011100011101'),
           ('M25', '11110111001111101110'), ('M26', '11110111001111101101'), ('M27', '11101111001111101110'),
           ('M28', '11101111001111101101'), ('M29', '11110101110011101110'), ('M30', '11110101110011101101'),
           ('M31', '11101101110011101110'), ('M32', '11101101110011101101'), ('M33', '11110111011111101110'),
           ('M34', '11110111011111101101'), ('M35', '11101111011111101110'), ('M36', '11101111011111101101'),
           ('M37', '11110111001111111110'), ('M38', '11110111001111111101'), ('M39', '11101111001111111110'),
           ('M40', '11101111001111111101'), ('M41', '11110101110011111110'), ('M42', '11110101110011111101'),
           ('M43', '11101101110011111110'), ('M44', '11101101110011111101'), ('M45', '11110111011111111110'),
           ('M46', '11110111011111111101'), ('M47', '11101111011111111110'), ('M48', '11101111011111111101'),
           ('M49', '11111111111111111110'), ('M50', '11111111111111111101'), ('M51', "11111111111111111111")]

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# Study('DataN.csv', 'cross validation', num_outputs=4, data_partition=(75, 15),
#       tolerance=0.0000001, learning_rate=0.4, maximum_epochs=350,
#       adapt_learning_rate=False, annealing_value=2000,
#       display_graph_windows=False, display_graph_pdf=False,
#       data_file_has_titles=True, data_file_has_brief_titles=True,
#       minimum_slope_to_consider_overfitting=2, number_of_epochs_for_overfit_check=10,
#       relative_importance_method="M")
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
_dummy = None

# for repeat in range(1):
#     # for method in methods:
#     # for merging text files goto command prompt and execute this: copy *.txt @02newfile.txt
#     methods_selected = [5] # now write the actual number of the method, not the index of it
#     methods_selected = map(lambda y: y - 1, methods_selected)  # range(51)[::-1]  # [4, 8, 20]
#     random.shuffle(methods_selected)
#     for num_method in methods_selected:  # (50,49,-1)[::-1]:
#                                           # (51)[::-1]:  # (51, 48, -1) 8=M09, 4=M05, 20=M21, 50=M51
#         method = methods[num_method]
#         prefix = method[0] + '_'
#         key = method [1]
#         num = int(prefix[1:3])
#         outs = 4 if num == 51 else 3
#         # if num in [5, 9, 21]:
#         try:
#             # if num <= 20:
#             Study('GrossData3V.csv', 'cross validation', num_outputs=outs, data_partition=(75, 15),
#                   tolerance=0.0000001, learning_rate=0.4, maximum_epochs=500,
#                   adapt_learning_rate=False, annealing_value=2000,
#                   display_graph_windows=False, display_graph_pdf=False,
#                   data_file_has_titles=True, data_file_has_brief_titles=True,
#                   minimum_slope_to_consider_overfitting=2, number_of_epochs_for_overfit_check=10,
#                   relative_importance_method="M", folder_prefix=prefix,
#                   variable_selection_key=key)
#         except Exception, err:
#             print '\n\n\nUnfortunately, the following method failed:' + str(method)
#             # traceback.print_exc()
#             # traceback.print_stack()
#             print err

_dummy2 = None
# Study(purpose= "advanced query", previous_study_data_file="NeuroCharterNet.nsr", start_time=time.time(),
#       input_parameters_suggested_values= ([1, 8, 2], ('VGN', 'KSG'), [30, 360, 30], [11, 19, 2],
#                                           6.2, [-.97, 0.97, 0.25], [0.25, 1, 0.25], 0.047))
#
# Study(purpose= "advanced query", previous_study_data_file="M05.nsr", start_time=time.time(),
#       input_parameters_suggested_values= ((2, 4), ('VGN', 'KSG'), (240, 360), 'SND',
#                                           6.2, [-.95, 0.80, 0.25], [0.25, 1, 0.25], 0.047))
#
# Study(purpose= "advanced query", previous_study_data_file="M09.nsr", start_time=time.time(),
#       input_parameters_suggested_values= ((2, 4), ('VGN', 'KSG'), (240, 360), 'SND',
#                                           6.2, [0.08, 0.39, .035], [-.95, 0.80, 0.25], 0.047, .045, .43))
#
# Study(purpose= "advanced query", previous_study_data_file="M21.nsr", start_time=time.time(),
#       input_parameters_suggested_values= ((2, 4), ('VGN', 'KSG'), (240, 360), 'SND',
#                                           6.2, [0.08, 0.39, .035], [-.95, 0.80, 0.25], 0.047, .045, .43, 0))
# print 'OK'
# Study(purpose= "advanced query", previous_study_data_file="M05.nsr", start_time=time.time(),
#       input_parameters_suggested_values= ((2, 4), ('VGN', 'KSG'), (240, 360), 'SND',
#                                           6.2,
#                                           (0.986111111, 0.958333333, 0.916666667, 0.875, 0.833333333, 0.666666667,
#                                            0.5, 0.333333333, 0, -0.732888922, -0.81010134, -0.887313757,
#                                            -0.932480126),
#                                           (0.104514623, 0.197689766, 0.290864909, 0.384040052,0.477215195, 0.570390338,
#                                            0.663565481, 0.756740623,0.849915766),
#                                           0.047))
#
# Study(purpose= "advanced query", previous_study_data_file="M09.nsr", start_time=time.time(),
#       input_parameters_suggested_values= ((2, 4), ('VGN', 'KSG'), (240, 360), 'SND',
#                                           6.2,
#                                           (0.08523813, 0.12111056, 0.15698299, 0.19285542, 0.22872785, 0.26460028,
#                                            0.30047271, 0.33634514, 0.37221757),
#                                           (0.986111111, 0.958333333,0.916666667, 0.875, 0.833333333, 0.666666667,
#                                            0.5, 0.333333333, 0, -0.732888922, -0.81010134, -0.887313757, -0.932480126),
#                                           0.047, .045, .43))
#
# Study(purpose= "advanced query", previous_study_data_file="M21.nsr", start_time=time.time(),
#       input_parameters_suggested_values= ((2, 4), ('VGN', 'KSG'), (240, 360), 'SND',
#                                           6.2,
#                                           (0.08523813, 0.12111056, 0.15698299, 0.19285542, 0.22872785, 0.26460028,
#                                            0.30047271, 0.33634514, 0.37221757),
#                                           (0.986111111, 0.958333333, 0.916666667, 0.875, 0.833333333, 0.666666667,
#                                            0.5, 0.333333333, 0, -0.732888922, -0.81010134, -0.887313757, -0.932480126),
#                                           0.047, .045, .43, (0, 1)))
# print 'OK'

# methods = [('M01', '111101110011000011'), ('M02', '111101110011000011'), ('M03', '111011110011000011'),
#            ('M04', '111011110011000011'), ('M05', '111101011100000011'), ('M06', '111101011100000011'),
#            ('M07', '111011011100000011'), ('M08', '111011011100000011'), ('M09', '111101110111000011'),
#            ('M10', '111101110111000011'), ('M11', '111011110111000011'), ('M12', '111011110111000011'),
#            ('M13', '111101110011000111'), ('M14', '111101110011000111'), ('M15', '111011110011000111'),
#            ('M16', '111011110011000111'), ('M17', '111101011100000111'), ('M18', '111101011100000111'),
#            ('M19', '111011011100000111'), ('M20', '111011011100000111'), ('M21', '111101110111000111'),
#            ('M22', '111101110111000111'), ('M23', '111011110111000111'), ('M24', '111011110111000111'),
#            ('M25', '111101110011111011'), ('M26', '111101110011111011'), ('M27', '111011110011111011'),
#            ('M28', '111011110011111011'), ('M29', '111101011100111011'), ('M30', '111101011100111011'),
#            ('M31', '111011011100111011'), ('M32', '111011011100111011'), ('M33', '111101110111111011'),
#            ('M34', '111101110111111011'), ('M35', '111011110111111011'), ('M36', '111011110111111011'),
#            ('M37', '111101110011111111'), ('M38', '111101110011111111'), ('M39', '111011110011111111'),
#            ('M40', '111011110011111111'), ('M41', '111101011100111111'), ('M42', '111101011100111111'),
#            ('M43', '111011011100111111'), ('M44', '111011011100111111'), ('M45', '111101110111111111'),
#            ('M46', '111101110111111111'), ('M47', '111011110111111111'), ('M48', '111011110111111111'),
#            ('M49', '111111111111111111')]
#
#
#
# for dummy_repeat in range(5):
#     # for method in methods:
#     # for merging text files goto command prompt and execute this: copy *.txt @02newfile.txt
#     selected_range = [4, 8, 20] # range(49)[::-1]  # [4, 8, 20]
#     random.shuffle(selected_range)
#     for num_method in selected_range:
#         method = methods[num_method]
#         prefix = method[0] + '_'
#         key = method [1]
#         num = int(prefix[1:3])
#         outs = 2
#         # if num in [5, 9, 21]:
#         try:
#             # if num <= 20:
#             Study('d_ABLim.csv', 'cross validation', num_outputs=outs, data_partition=(85, 5),
#                   tolerance=0.0000001, learning_rate=0.4, maximum_epochs=1000,
#                   adapt_learning_rate=False, annealing_value=2000,
#                   display_graph_windows=False, display_graph_pdf=False,
#                   data_file_has_titles=True, data_file_has_brief_titles=True,
#                   minimum_slope_to_consider_overfitting=2, number_of_epochs_for_overfit_check=10,
#                   relative_importance_method="M", folder_prefix=prefix,
#                   variable_selection_key=key)
#         except:
#             print '\n\n\nUnfortunately, the following method failed:' + str(method)

q_vals = (1, 2, 3, 4, 6, 8, 12)
models = ('VGN', 'Dul', 'MVG', 'BrC', 'KSG', 'VG2')
duration = (30, 60, 90, 120, 180, 240, 360)
texture = ('ClL', 'SCL', 'SND', 'SnC', 'LSn', 'SnL', 'Lom', 'Slt')
fem = (1.5, 2.5, 6.2, 10, 12.5, 20)
time_factor = (0.986111111, 0.958333333, 0.916666667, 0.875, 0.833333333, 0.666666667,
                0.5, 0.333333333, 0, -0.732888922, -0.81010134, -0.887313757, -0.932480126)
eff_sat = (0.104514623, 0.197689766, 0.290864909, 0.384040052,0.477215195, 0.570390338,
                                           0.663565481, 0.756740623,0.849915766)
sat_cond = (.047, .1, .2, .24319, .3, .4, .495)
print '.'
#
# Study(purpose= "advanced query", previous_study_data_file="M05n3V.nsr", start_time=time.time(),
#       input_parameters_suggested_values= ((2, 4), ('VGN', 'KSG'), (240, 360), ('SND', 'Lom'),
#                                           6.2,
#                                           (0.986111111, 0.958333333, 0.916666667, 0.875, 0.833333333, 0.666666667,
#                                            0.5, 0.333333333, 0, -0.732888922, -0.81010134, -0.887313757,
#                                            -0.932480126),
#                                           (0.104514623, 0.197689766, 0.290864909, 0.384040052,0.477215195, 0.570390338,
#                                            0.663565481, 0.756740623,0.849915766),
#                                           0.495))
#
# Study(purpose= "advanced query", previous_study_data_file="M09n3V.nsr", start_time=time.time(),
#       input_parameters_suggested_values= ((2, 4), ('VGN', 'KSG'), (240, 360), ('SND', 'Lom'),
#                                           6.2,
#                                           (0.08523813, 0.12111056, 0.15698299, 0.19285542, 0.22872785, 0.26460028,
#                                            0.30047271, 0.33634514, 0.37221757),
#                                           (0.986111111, 0.958333333,0.916666667, 0.875, 0.833333333, 0.666666667,
#                                            0.5, 0.333333333, 0, -0.732888922, -0.81010134, -0.887313757, -0.932480126),
#                                           0.495, .045, .43))
#
# Study(purpose= "advanced query", previous_study_data_file="M21n3V.nsr", start_time=time.time(),
#       input_parameters_suggested_values= ((2, 4), ('VGN', 'KSG'), (240, 360), ('SND', 'Lom'),
#                                           6.2,
#                                           (0.08523813, 0.12111056, 0.15698299, 0.19285542, 0.22872785, 0.26460028,
#                                            0.30047271, 0.33634514, 0.37221757),
#                                           (0.986111111, 0.958333333, 0.916666667, 0.875, 0.833333333, 0.666666667,
#                                            0.5, 0.333333333, 0, -0.732888922, -0.81010134, -0.887313757, -0.932480126),
#                                           0.495, .045, .43, (0, 1)))