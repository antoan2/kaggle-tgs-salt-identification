import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import pickle


class Logger():
    def __init__(self):
        self.logs = {}
        for dataset in ['train', 'validation']:
            self.logs[dataset] = {}
            for log_type in ['loss', 'accuracy']:
                self.logs[dataset][log_type] = defaultdict(list)

    def add_log(self, keys, value):
        dataset, log_type, fold = keys
        self.logs[dataset][log_type][fold].append(value)

    def plot_logs(self):

        fig = plt.figure()
        ax = dict()
        ax['loss'] = fig.add_subplot(211)
        ax['loss'].set_yscale("log", nonposy='clip')
        ax['accuracy'] = fig.add_subplot(212)
        for log_type in ['loss', 'accuracy']:
            for dataset, color in zip(['train', 'validation'],
                                      ['red', 'blue']):
                np_plot_values = None
                for log_values in self.logs[dataset][log_type].values():
                    batches, plot_values = zip(*log_values)
                    if np_plot_values is None:
                        np_plot_values = np.asarray(plot_values)
                    else:
                        np_plot_values = np.vstack((np_plot_values,
                                                    np.asarray(plot_values)))
                    print(np_plot_values.shape)

                x = np.asarray(batches)
                y = np_plot_values.mean(axis=0)
                yerr = np_plot_values.std(axis=0)
                ax[log_type].errorbar(x, y, yerr, color=color, ecolor=color)

        fig.savefig('plot.png')
        plt.close()

    def save(self, path):
        with open(path, 'wb') as file_id:
            pickle.dump(self.logs, file_id, pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        with open(path, 'rb') as file_id:
            self.logs = pickle.load(file_id)
