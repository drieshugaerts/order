# from autocomplete import run, test_synthetic
from autocomplete.main import run

import pandas
from random import randrange, seed, random, shuffle
from random import choice as random_choice
# from tacle import tables_from_csv
from os import listdir
from os.path import isfile, join
# import glob
# import csv
import tacle
import numpy as np
import pickle
import time
import logging


class Result:
    def __init__(self, filename, num_unknowns, type_unknowns, random_seed):
        self.scores = {}
        self.filename = filename
        self.num_unknowns = num_unknowns
        self.type_unknowns = type_unknowns
        self.random_seed = random_seed
        self.best_sorting = None
        # TODO: self.orientation

    def print_scores(self):
        for type, score in self.scores.items():
            print("Score for {} is {:.4f} ".format(type, score), end="")
        print('')

    def print_improvements(self):
        if self.scores['rand'] is None:
            print('No random score')
        else:
            for type, score in self.scores.items():
                if type is not 'rand' and self.scores["rand"] != 0:
                    improvement = 100 * (score - self.scores['rand']) / self.scores['rand']
                    print("Improvement for {} is {:.2f}%".format(type, improvement), end=" ")
            print('')

    def get_targets(self):
        tables = tacle.tables_from_csv(self.filename)
        table = tables[0]

        rows = table.data.shape[0]
        cols = table.data.shape[1]

        removed = 0
        cols_index = []
        rows_index = []

        seed(self.random_seed)
        while removed < num_remov:
            random_index_c = randrange(cols)
            random_index_r = randrange(rows)
            if not is_in(random_index_c, random_index_r, cols_index, rows_index):
                for block in table.blocks:
                    bounds = block.bounds.bounds
                    if bounds[2] <= random_index_c < bounds[3]:
                        if bounds[0] <= random_index_r < bounds[1]:
                            cols_index.append(random_index_c)
                            rows_index.append(random_index_r)
                            removed += 1
        return cols_index, rows_index

    def find_best_sorting(self):
        self.best_sorting = [k for k, v in self.scores.items() if v == max(self.scores.values())]
        if 'best' in self.best_sorting:
            self.best_sorting.remove('best')
        self.scores['best'] = self.scores[self.best_sorting[0]]


def is_in(x, y, xlist, ylist):
    for i in range(len(xlist)):
        if x == xlist[i] and y == ylist[i]:
            return True
    return False


def sorting(xlist, ylist, type_sorting):
    if type_sorting == 'tblr':
        for i in range(len(xlist)):
            for j in range(len(xlist) - i - 1):
                if (xlist[j] > xlist[j + 1] or (xlist[j] == xlist[j + 1] and ylist[j] > ylist[j + 1])):
                    tempx = xlist[j]
                    xlist[j] = xlist[j + 1]
                    xlist[j + 1] = tempx
                    tempy = ylist[j]
                    ylist[j] = ylist[j + 1]
                    ylist[j + 1] = tempy
    elif type_sorting == 'lrtb':
        (ylist, xlist) = sorting(ylist, xlist, 'tblr')
    elif type_sorting == 'tbrl':
        xlist_neg = [-x for x in xlist]
        (xlist_neg, ylist) = sorting(xlist_neg, ylist, 'tblr')
        xlist = [-x for x in xlist_neg]
    elif type_sorting == 'rltb':
        xlist_neg = [-x for x in xlist]
        (xlist_neg, ylist) = sorting(xlist_neg, ylist, 'lrtb')
        xlist = [-x for x in xlist_neg]
    elif type_sorting == 'btlr':
        ylist_neg = [-y for y in ylist]
        (xlist, ylist_neg) = sorting(xlist, ylist_neg, 'tblr')
        ylist = [-y for y in ylist_neg]
    elif type_sorting == 'btrl':
        ylist_neg = [-y for y in ylist]
        (xlist, ylist_neg) = sorting(xlist, ylist_neg, 'tbrl')
        ylist = [-y for y in ylist_neg]
    elif type_sorting == 'lrbt':
        ylist_neg = [-y for y in ylist]
        (xlist, ylist_neg) = sorting(xlist, ylist_neg, 'lrtb')
        ylist = [-y for y in ylist_neg]
    elif type_sorting == 'rlbt':
        ylist_neg = [-y for y in ylist]
        (xlist, ylist_neg) = sorting(xlist, ylist_neg, 'rltb')
        ylist = [-y for y in ylist_neg]
    elif type_sorting == 'diag_cw':
        for i in range(len(xlist)):
            for j in range(len(xlist) - i - 1):
                if ((xlist[j] + ylist[j]) > (xlist[j + 1] + ylist[j + 1]) or (
                        (xlist[j] + ylist[j]) == (xlist[j + 1] + ylist[j + 1]) and xlist[j] > xlist[j + 1])):
                    tempx = xlist[j]
                    xlist[j] = xlist[j + 1]
                    xlist[j + 1] = tempx
                    tempy = ylist[j]
                    ylist[j] = ylist[j + 1]
                    ylist[j + 1] = tempy
    elif type_sorting == 'diag_acw':
        for i in range(len(xlist)):
            for j in range(len(xlist) - i - 1):
                if ((xlist[j] + ylist[j]) > (xlist[j + 1] + ylist[j + 1]) or (
                        (xlist[j] + ylist[j]) == (xlist[j + 1] + ylist[j + 1]) and xlist[j] < xlist[j + 1])):
                    tempx = xlist[j]
                    xlist[j] = xlist[j + 1]
                    xlist[j + 1] = tempx
                    tempy = ylist[j]
                    ylist[j] = ylist[j + 1]
                    ylist[j + 1] = tempy
    elif type_sorting == 'radial_cw':
        for i in range(len(xlist)):
            for j in range(len(xlist) - i - 1):
                if ((xlist[j] ** 2 + ylist[j] ** 2) > (xlist[j + 1] ** 2 + ylist[j + 1] ** 2) or (
                        (xlist[j] ** 2 + ylist[j] ** 2) == (xlist[j + 1] ** 2 + ylist[j + 1] ** 2) and xlist[j] > xlist[j + 1])):
                    tempx = xlist[j]
                    xlist[j] = xlist[j + 1]
                    xlist[j + 1] = tempx
                    tempy = ylist[j]
                    ylist[j] = ylist[j + 1]
                    ylist[j + 1] = tempy
    elif type_sorting == 'radial_acw':
        for i in range(len(xlist)):
            for j in range(len(xlist) - i - 1):
                if ((xlist[j] ** 2 + ylist[j] ** 2) > (xlist[j + 1] ** 2 + ylist[j + 1] ** 2) or (
                        (xlist[j] ** 2 + ylist[j] ** 2) == (xlist[j + 1] ** 2 + ylist[j + 1] ** 2) and xlist[j] < xlist[j + 1])):
                    tempx = xlist[j]
                    xlist[j] = xlist[j + 1]
                    xlist[j + 1] = tempx
                    tempy = ylist[j]
                    ylist[j] = ylist[j + 1]
                    ylist[j + 1] = tempy
    elif type_sorting == 'box_cw':
        for i in range(len(xlist)):
            for j in range(len(xlist) - i - 1):
                if ((max(xlist[j], ylist[j]) > max(xlist[j + 1], ylist[j + 1])) or (
                        max(xlist[j], ylist[j]) == max(xlist[j + 1], ylist[j + 1]) and ylist[i] > ylist[i + 1]) or (
                        max(xlist[j], ylist[j]) == max(xlist[j + 1], ylist[j + 1]) and ylist[i] == ylist[i + 1] and
                        xlist[i] < xlist[i + 1])):
                    tempx = xlist[j]
                    xlist[j] = xlist[j + 1]
                    xlist[j + 1] = tempx
                    tempy = ylist[j]
                    ylist[j] = ylist[j + 1]
                    ylist[j + 1] = tempy
    elif type_sorting == 'box_cw':
        for i in range(len(xlist)):
            for j in range(len(xlist) - i - 1):
                if ((max(xlist[j], ylist[j]) > max(xlist[j + 1], ylist[j + 1])) or (
                        max(xlist[j], ylist[j]) == max(xlist[j + 1], ylist[j + 1]) and xlist[i] > xlist[i + 1]) or (
                        max(xlist[j], ylist[j]) == max(xlist[j + 1], ylist[j + 1]) and xlist[i] == xlist[i + 1] and
                        ylist[i] < ylist[i + 1])):
                    tempx = xlist[j]
                    xlist[j] = xlist[j + 1]
                    xlist[j + 1] = tempx
                    tempy = ylist[j]
                    ylist[j] = ylist[j + 1]
                    ylist[j + 1] = tempy
    elif type_sorting == 'rand_perm':
        seed()
        r = random()
        shuffle(xlist, lambda: r)
        shuffle(ylist, lambda: r)

    return xlist, ylist


def sortarray_rows(array, rows):
    # places the rows with unknown values at the end of the array in order

    short_list = []
    for i in range(len(rows) - 1, -1, -1):
        if rows[i] not in short_list:
            short_list = [rows[i]] + short_list

    num_rows_arr = array.shape[0]
    num_unknown = len(short_list)

    for i in range(num_unknown):
        # switch index short_list[-i-1] and -i-1
        temp = array[short_list[-i - 1], :].copy()
        array[short_list[-i - 1], :] = array[-i - 1, :]
        array[-i - 1, :] = temp[:]
        for j in range(num_unknown):
            if short_list[j] == short_list[-i - 1]:
                short_list[j] = num_rows_arr - i - 1
            elif short_list[j] == num_rows_arr - i - 1:
                short_list[j] == short_list[-i - 1]
    return array


def end_rows(rows, length_of_array):
    end_list = []
    used_indices = []
    pos = length_of_array - 1
    for i in reversed(rows):
        new_row_of_i = pos
        if i in used_indices:
            for j in range(len(used_indices)):
                if used_indices[j] == i:
                    new_row_of_i = end_list[j]

                    break
        end_list = [new_row_of_i] + end_list
        used_indices = [i] + used_indices
        pos -= 1

    return end_list


def get_types(file_name):
    dataframe = pandas.read_csv(file_name, header=None)

    types = dataframe.applymap(type)
    return types.values

def count_str_float(data):
    str_count = 0
    float_count = 0
    for _, elem in np.ndenumerate(data):
        try:
            float(elem)
            float_count += 1
        except ValueError:
            str_count += 1
    return str_count, float_count



def run_once(file_name, num_remov, order, type_unknowns=None, currrent_random_seed=None):
    tables = tacle.tables_from_csv(file_name)

    table = tables[0]

    rows = table.data.shape[0]
    cols = table.data.shape[1]

    removed = 0
    cols_index = []
    rows_index = []

    seed(currrent_random_seed)
    while removed < num_remov:
        random_index_c = randrange(cols)
        random_index_r = randrange(rows)
        if not is_in(random_index_c, random_index_r, cols_index, rows_index):
            if type_unknowns is None or type_unknowns is 'both':
                cols_index.append(random_index_c)
                rows_index.append(random_index_r)
                removed += 1
            else:
                try:
                    float(table.data[random_index_r][random_index_c])
                    if type_unknowns is 'float':
                        cols_index.append(random_index_c)
                        rows_index.append(random_index_r)
                        removed += 1
                except ValueError:
                    if type_unknowns is 'str':
                        cols_index.append(random_index_c)
                        rows_index.append(random_index_r)
                        removed += 1

            # for block in table.blocks:
            #     bounds = block.bounds.bounds
            #     if bounds[2] <= random_index_c < bounds[3]:
            #         if bounds[0] <= random_index_r < bounds[1]:
            #             cols_index.append(random_index_c)
            #             rows_index.append(random_index_r)
            #             removed += 1

    # print(table.data)
    cols_index, rows_index = sorting(cols_index, rows_index, order)
    # print(rows_index, cols_index)
    table.data = sortarray_rows(table.data, rows_index)
    new_rows_index = end_rows(rows_index, rows)

    # print(new_rows_index)
    # print(table.data)

    # print(cols_index)
    # print(table.relative_range)

    average_score = 0
    for i in range(num_remov):
        scores = run([(0, new_rows_index[i], cols_index[i])], tables, constraints=[],
                     ignore_extra_columns_per_table=cols_index[i + 1:], mix='none', order='not_exhaustive')
        score = scores[0]
        average_score += score[0] / num_remov
    return average_score

if __name__ == '__main__':
    file_names = [f for f in listdir('./csv') if isfile(join('./csv', f))]

    file_names.remove('small.csv')
    file_names.remove('blanks.csv')
    file_names.remove('test.csv')
    file_names.remove('icecream_sales.csv')
    file_names.remove('.DS_Store')
    # files with more then 1 table:
    file_names.remove('9_1.csv')
    file_names.remove('10.csv')
    file_names.remove('11.csv')
    file_names.remove('15.csv')
    file_names.remove('17.csv')
    file_names.remove('20.csv')
    file_names.remove('22_1.csv')
    file_names.remove('22_2.csv')
    file_names.remove('22_3.csv')
    file_names.remove('24.csv')
    file_names.remove('25.csv')
    file_names.remove('26_1.csv')
    file_names.remove('26_2.csv')
    file_names.remove('28.csv')
    # import problems
    file_names.remove('21.csv')
    file_names.remove('14.csv')
    # too large
    file_names.remove('3_1.csv')
    file_names.remove('3_2.csv')
    file_names.remove('19.csv')
    # RuntimeError: Illegal state: 0, vertical, T1
    file_names.remove('16.csv')
    file_names.remove('23.csv')
    file_names.remove('9_2.csv')
    file_names.remove('1.csv')
    # ValueError: Found array with 0 sample(s) (shape=(0, 1)) while a minimum of 1 is required.
    file_names.remove('18_2.csv')
    file_names.remove('13_3.csv')
    file_names.remove('13_1.csv')
    file_names.remove('18_4.csv')
    file_names.remove('13_2.csv')

    print(file_names)

    # bad_filenames = []
    # reasons_removed = {}
    # max_time_per_prediction = 3  # s

    num_elements_in_file = {}
    num_str_in_file = {}
    num_float_in_file = {}
    orientation_of_file = {}
    percentage_unknowns = 0.1
    for file_name in file_names:
        tables = tacle.tables_from_csv('csv/' + file_name)
        table = tables[0]
        num_elements_in_file[file_name] = table.data.size
        num_str_in_file[file_name], num_float_in_file[file_name] = count_str_float(table.data)

        orientation_of_file[file_name] = table.orientations[0]

    # print(num_elements_in_file)

    try:
        pkl_file = open('data.pkl', 'rb')
        result_log = pickle.load(pkl_file)
    except (OSError, IOError) as e:
        result_log = []

    currrent_random_seed = randrange(1000000)

    ocurred_errors = []

    try:
        while True:

            file_name = random_choice(file_names)
            path = 'csv/' + file_name

            print('Iteration: {}'.format(1 + len(result_log)))
            print('Current file: {}'.format(file_name))

            try:
                if len(result_log) % 3 == 0:
                    type_unknowns = 'str'
                    num_unknowns = round(num_str_in_file[file_name] * percentage_unknowns)
                elif len(result_log) % 3 == 1:
                    type_unknowns = 'float'
                    num_unknowns = round(num_float_in_file[file_name] * percentage_unknowns)
                else:
                    type_unknowns = 'both'
                    num_unknowns = round(num_elements_in_file[file_name] * percentage_unknowns)

                result = Result(path, num_unknowns, type_unknowns, currrent_random_seed)
                # start_time = time.perf_counter()
                #
                # result_random = run_once(path, num_unknowns, 'rand', type_unknowns, currrent_random_seed)
                #
                # end_time = time.perf_counter()



                # if end_time - start_time > max_time_per_prediction*num_unknowns:
                #     print( '{:.1f}s is too long. Removed file: {}'.format(end_time - start_time, file_name))
                #     bad_filenames.append(file_name)
                #     file_names.remove(file_name)
                #     reasons_removed[file_name] = 'time: {}s'.format(end_time - start_time)
                #
                # else:
                result.scores['rand'] = run_once(path, num_unknowns, 'rand', type_unknowns, currrent_random_seed) #result_random
                result.scores['lrtb'] = run_once(path, num_unknowns, 'lrtb', type_unknowns, currrent_random_seed)
                result.scores['tblr'] = run_once(path, num_unknowns, 'tblr', type_unknowns, currrent_random_seed)
                result.scores['rlbt'] = run_once(path, num_unknowns, 'rlbt', type_unknowns, currrent_random_seed)
                result.scores['btrl'] = run_once(path, num_unknowns, 'btrl', type_unknowns, currrent_random_seed)
                result.scores['btlr'] = run_once(path, num_unknowns, 'btlr', type_unknowns, currrent_random_seed)
                result.scores['lrbt'] = run_once(path, num_unknowns, 'lrbt', type_unknowns, currrent_random_seed)
                result.scores['tbrl'] = run_once(path, num_unknowns, 'tbrl', type_unknowns, currrent_random_seed)
                result.scores['rltb'] = run_once(path, num_unknowns, 'rltb', type_unknowns, currrent_random_seed)
                result.scores['diag_cw'] = run_once(path, num_unknowns, 'diag_cw', type_unknowns, currrent_random_seed)
                result.scores['diag_acw'] = run_once(path, num_unknowns, 'diag_acw', type_unknowns, currrent_random_seed)
                result.scores['box_cw'] = run_once(path, num_unknowns, 'box_cw', type_unknowns, currrent_random_seed)
                result.scores['box_acw'] = run_once(path, num_unknowns, 'box_acw', type_unknowns, currrent_random_seed)
                result.find_best_sorting()

                result.scores['rand100'] = run_once(path, num_unknowns, 'rand_perm', type_unknowns, currrent_random_seed)
                for iteration in range(99):
                    current_score = run_once(path, num_unknowns, 'rand_perm', type_unknowns, currrent_random_seed)
                    # print(current_score)
                    if current_score > result.scores['rand100']:
                        result.scores['rand100'] = current_score

                result.print_scores()
                # result.print_improvements()

                result_log.append(result)


            except KeyboardInterrupt:
                raise
            except Exception as e:
                logging.error(e)
                ocurred_errors.append('{} on file {}'.format(e, file_name))

            currrent_random_seed += 1

    except KeyboardInterrupt:
        pass

    output = open('data.pkl', 'wb')
    pickle.dump(result_log, output)
    output.close()

    output2 = open('errors.pkl', 'wb')
    pickle.dump(ocurred_errors, output2)
    output.close()

print(ocurred_errors)

# print('The bad filenames:', end='')
# print(bad_filenames)
# print(reasons_removed)
