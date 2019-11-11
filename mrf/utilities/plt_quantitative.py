import math
import os
import typing

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import plotly as py
import plotly.graph_objs as go
import plotly.tools as tls

import mrf.data.data as data


def get_unit(map_: typing.Union[str, data.FileTypes]):
    if isinstance(map_, data.FileTypes):
        map_ = map_.name

    if map_ == data.FileTypes.T1H2Omap.name:
        return '$\mathrm{T1_{H2O}}$ (ms)'
    elif map_ == data.FileTypes.FFmap.name:
        return 'FF'
    elif map_ == data.FileTypes.B1map.name:
        return 'B1 (a.u.)'
    else:
        raise ValueError('Map {} not supported'.format(map_.replace('map', '')))


class QuantitativePlotter:

    def __init__(self, plot_path: str, metrics: tuple = ('NRMSE', ), plot_format: str = 'png'):
        self.path = plot_path
        self.metrics = metrics
        self.plot_format = plot_format

        os.makedirs(self.path, exist_ok=True)

    def plot(self, csv_file: str, file_name: str = 'summary', plot_images: bool = True):
        df = pd.read_csv(csv_file, sep=';')
        experiment, _ = os.path.splitext(os.path.basename(csv_file))

        plotly_figs = []
        plotly_titles = []
        plotly_yaxis = []
        plotly_minmax = []

        for map_ in df['MAP'].unique():
            for mask in df['MASK'].unique():

                values = df[(df['MAP'] == map_) & (df['MASK'] == mask)]

                # not all map mask combinations have been calculated on CSV file generation
                if values.count().any() == 0:
                    continue

                for metric in self.metrics:
                    data = values[metric].values

                    if plot_images:
                        self.plot_box(
                            os.path.join(self.path, '{}_{}_{}.{}'.format(map_, metric, mask, self.plot_format)),
                            data,
                            '{} ({})\n{}'.format(map_, mask, experiment),
                            '',
                            self._text_of_metric(metric),
                            self._get_min_of_metric(metric), self._get_max_of_metric(metric)
                        )

                    plotly_figs.append(go.Box(y=values[metric],
                                              text=values['ID'],
                                              boxpoints='outliers',
                                              marker=dict(color='rgb(0, 0, 0)'),
                                              fillcolor='rgba(255,255,255,0)',
                                              boxmean='sd',
                                              name=''
                                              ))
                    plotly_titles.append(map_)
                    plotly_yaxis.append(self._text_of_metric(metric))
                    plotly_minmax.append((self._get_min_of_metric(metric) if self._get_min_of_metric(metric) is not None else min(values[metric]) * 0.9,
                                          self._get_max_of_metric(metric) if self._get_max_of_metric(metric) is not None else max(values[metric]) * 1.1))

        cols = len(self.metrics)
        rows = math.ceil(len(plotly_figs) / cols)
        fig = tls.make_subplots(rows=rows, cols=cols, subplot_titles=plotly_titles, print_grid=False)

        fig_idx = 0
        for row in range(1, rows + 1):
            for col in range(1, cols + 1):
                if fig_idx < len(plotly_figs):
                    fig.append_trace(plotly_figs[fig_idx], row, col)
                    fig_idx += 1

        fig['layout'].update(title=experiment, showlegend=False, height=rows*400)
        for idx in range(1, len(plotly_figs) + 1):
            # todo: solution to update only min OR max? or get autorange value? currently, we multiply min/max above
            fig['layout']['yaxis{}'.format(idx)].update(range=plotly_minmax[idx - 1], title=plotly_yaxis[idx - 1])
            fig['layout']['xaxis{}'.format(idx)].update(title='')

        py.offline.plot(fig, filename=os.path.join(self.path, file_name + '.html'), auto_open=False)

    @staticmethod
    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)

    @staticmethod
    def set_box_format(bp):
        plt.setp(bp['caps'], linewidth=0)
        plt.setp(bp['medians'], linewidth=1.5)
        plt.setp(bp['fliers'], marker='.')
        plt.setp(bp['fliers'], markerfacecolor='black')
        plt.setp(bp['fliers'], alpha=1)

    def plot_box(self, file_path: str, data, title: str, x_label: str, y_label: str,
                 min_: float = None, max_: float = None):
        fig = plt.figure(figsize=plt.rcParams["figure.figsize"][::-1])  # figsize defaults to (width, height)=(6.4, 4.8)
        # for boxplots, we want the ratio to be inversed
        ax = fig.add_subplot(111)  # create an axes instance (nrows=ncols=index)
        bp = ax.boxplot(data, widths=0.6)
        self.set_box_format(bp)

        ax.set_title(title)
        ax.set_ylabel(y_label)
        if x_label is not None:
            ax.set_xlabel(x_label)

        # remove frame
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # thicken frame
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)

        # adjust min and max if provided
        if min_ is not None or max_ is not None:
            min_original, max_original = ax.get_ylim()
            min_ = min_ if min_ is not None and min_ < min_original else min_original
            max_ = max_ if max_ is not None and max_ > max_original else max_original
            ax.set_ylim(min_, max_)

        plt.savefig(file_path)
        plt.close()

    @staticmethod
    def _get_min_of_metric(metric: str):
        if metric == 'MAE':
            return 0
        if metric == 'MSE':
            return 0
        if metric == 'RMSE':
            return 0
        if metric == 'NRMSE':
            return 0
        if metric == 'PSNR':
            return 0
        if metric == 'SSIM':
            return 0
        else:
            return None  # we do not raise an error

    @staticmethod
    def _get_max_of_metric(metric: str):
        if metric == 'MAE':
            return None
        if metric == 'MSE':
            return None
        if metric == 'RMSE':
            return None
        if metric == 'NRMSE':
            return None
        if metric == 'PSNR':
            return None
        if metric == 'SSIM':
            return 1
        else:
            return None  # we do not raise an error

    @staticmethod
    def _text_of_metric(metric: str):
        if metric == 'MAE':
            return 'Mean absolute error'
        if metric == 'MSE':
            return 'Mean squared error'
        if metric == 'RMSE':
            return 'Root mean squared error'
        if metric == 'NRMSE':
            return 'Normalized root mean squared error'
        if metric == 'PSNR':
            return 'Peak signal noise ratio'
        if metric == 'SSIM':
            return 'Structural similarity index metric'
        else:
            return metric
