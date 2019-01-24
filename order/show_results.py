import pickle
import matplotlib.pyplot as plt
#from .main import Result
class Result:
    def __init__(self, filename, num_unknowns, type_unknowns, random_seed):
        self.scores = {}
        self.filename = filename
        self.num_unknowns = num_unknowns
        self.type_unknowns = type_unknowns
        self.random_seed = random_seed
        self.best_sorting = None
        #TODO: self.orientation



    def print_scores(self):
        for type, score in self.scores.items():
            print("Score for {} is {:.4f} ".format(type, score), end="")
        print('')

    def print_improvements(self):
        if self.scores['rand'] is None:
            print('No random score')
        else:
            for type, score in self.scores.items():
                if type is not 'rand':
                    improvement = 100*(score - self.scores['rand'])/self.scores['rand']
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
        self.best_sorting.remove('best')
        self.scores['best'] = self.scores[self.best_sorting[0]]

class individual_scores:
    def __init__(self):
        self.scores = {}
        self.scores['rand'] = {}
        self.scores['lrtb'] = {}
        self.scores['tblr'] = {}
        self.scores['rltb'] = {}
        self.scores['tbrl'] = {}
        self.scores['lrbt'] = {}
        self.scores['btlr'] = {}
        self.scores['rlbt'] = {}
        self.scores['btrl'] = {}
        self.scores['diag_cw'] = {}
        self.scores['diag_acw'] = {}
        self.scores['radial_cw'] = {}
        self.scores['radial_acw'] = {}
        self.scores['best'] = {}
        self.type_unknowns = {}
        self.length = None
        self.averages = {}
        self.improvement = {}
        self.file_names_index = {}
        self.file_names = set()
        self.best_sorting = {}
        self.best_sorting_count = {}

    def load_results(self, filename, create_best = None):
        pkl_file = open(filename, 'rb')
        result_log = pickle.load(pkl_file)
        self.length = len(result_log)

        for index_result in range(self.length):
            if create_best:
                result_log[index_result].find_best_sorting()

            self.type_unknowns[index_result] = result_log[index_result].type_unknowns
            self.file_names_index[index_result] = result_log[index_result].filename
            self.file_names.add(result_log[index_result].filename)
            try:
                self.best_sorting[index_result] = result_log[index_result].best_sorting
            except AttributeError:
                pass

            for type_sorting, score in result_log[index_result].scores.items():
                self.scores[type_sorting][index_result] = score

    def compare_results(self, type_unknowns,  *args, file_name = None):

        if file_name is None:
            list_usable_results = {}
            for type_sorting in args:
                list_usable_results[type_sorting] = []
                index = 0
                for key in self.scores[type_sorting]:
                    score = self.scores[type_sorting][key]
                    if type_unknowns == self.type_unknowns[key] or type_unknowns is None:
                        list_usable_results[type_sorting].append(score)
                        index += 1
        else:
            list_usable_results = {}
            for type_sorting in args:
                list_usable_results[type_sorting] = []
                index = 0
                for key in self.scores[type_sorting]:
                    score = self.scores[type_sorting][key]
                    if (type_unknowns == self.type_unknowns[key] or type_unknowns is None) and (self.file_names_index[key] == file_name or file_name is None):
                        list_usable_results[type_sorting].append(score)
                        index += 1
        for type_sorting in args:
            plt.plot(list_usable_results[type_sorting], label=type_sorting)
            plt.ylabel('Average Score')
        plt.legend()
        plt.show()


    def compare_results_running_average(self, type_unknowns, *args, file_name=None):
        if file_name is None:
            list_usable_results = {}
            for type_sorting in args:
                list_usable_results[type_sorting] = []
                index = 0
                for key in self.scores[type_sorting]:
                    score = self.scores[type_sorting][key]
                    if type_unknowns == self.type_unknowns[key] or type_unknowns is None:
                        list_usable_results[type_sorting].append(score)
                        if list_usable_results[type_sorting] is not [score]:
                            list_usable_results[type_sorting][index] = list_usable_results[type_sorting][index - 1]*index/(index+1) + list_usable_results[type_sorting][index]/(index+1)
                        index += 1
        else:
            list_usable_results = {}
            for type_sorting in args:
                list_usable_results[type_sorting] = []
                index = 0
                for key in self.scores[type_sorting]:
                    score = self.scores[type_sorting][key]
                    if (type_unknowns == self.type_unknowns[key] or type_unknowns is None) and (self.file_names_index[key] == file_name or file_name is None):
                        list_usable_results[type_sorting].append(score)
                        if list_usable_results[type_sorting] is not [score]:
                            list_usable_results[type_sorting][index] = list_usable_results[type_sorting][index - 1] * index / (index + 1) + list_usable_results[type_sorting][index] / (index + 1)
                        index += 1



        for type_sorting in args:
            plt.plot(list_usable_results[type_sorting], label=type_sorting)
            plt.ylabel('Average Score')
        plt.legend()
        plt.show()

    def average_score_per_file(self):
        for file in self.file_names:
            print('{}:'.format(file))
            for type_sorting in ['rand', 'lrtb', 'tblr', 'rltb', 'tbrl', 'lrbt', 'btlr', 'rlbt', 'btrl', 'diag_cw', 'diag_acw', 'radial_cw', 'radial_acw']:
                index = 0
                sum_scores = 0
                for key in self.scores[type_sorting]:
                    if self.file_names_index[key] == file:
                        sum_scores += self.scores[type_sorting][key]
                        index += 1
                if index is not 0:
                    print('Average score {}: {:.4f}'.format(type_sorting, sum_scores/index))


    def calculate_averages(self):
        for type_sorting in self.scores:
            counter = 0
            self.averages[type_sorting] = 0
            for score in self.scores[type_sorting].values():
                self.averages[type_sorting] += score
                counter += 1
            if counter is 0:
                self.averages[type_sorting] = None
            else:
                self.averages[type_sorting] = self.averages[type_sorting] / float(counter)
                print("Average score for {} is {:.4f} ".format(type_sorting, self.averages[type_sorting]), end="")
        print("")

    def calculate_improvements(self):
        if self.averages == {}:
            self.calculate_averages()
        if self.averages['rand'] is not None:
            for type_sorting in self.scores:
                if type_sorting is not 'rand' and self.averages[type_sorting] is not None:
                    self.improvement[type_sorting] = 100*(self.averages[type_sorting] - self.averages['rand'])/self.averages['rand']
                    print("{} is an improvement of {:.2f}%. ".format(type_sorting, self.improvement[type_sorting]), end="")
            print("")
        else:
            print('No random scores given')

    def calculate_best(self):
        for index in range(self.length):
            if index not in self.scores['best'].keys():
                self.scores['best'][index] = 0
                self.best_sorting[index] = []
                for sorting in self.scores:
                    if (index in self.scores[sorting] and self.scores[sorting][index] is self.scores['best'][index] and sorting is not 'best'):
                        self.best_sorting[index].append(sorting)
                    elif index in self.scores[sorting] and self.scores[sorting][index] > self.scores['best'][index] and sorting is not 'best':
                        self.best_sorting[index] = [sorting]
                        self.scores['best'][index] = self.scores[sorting][index]
                for best_sorting in self.best_sorting[index]:
                    if best_sorting in self.best_sorting_count.keys():
                        self.best_sorting_count[best_sorting] += 1
                    else:
                        self.best_sorting_count[best_sorting] = 1










h = individual_scores()
h.load_results('data_1.pkl')
#h.compare_results_running_average(None, 'rand', 'lrtb', 'tblr', 'rlbt', 'btrl', 'diag_cw', 'radial_cw')
h.calculate_averages()
h.calculate_improvements()
h.average_score_per_file()
h.calculate_best()
#h.compare_results_running_average(None, 'rand', 'lrtb', 'tblr', 'rlbt', 'btrl', 'diag_cw', 'radial_cw','best')
print(h.best_sorting_count)
