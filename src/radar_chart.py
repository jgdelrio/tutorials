"""
Radar chart (aka radar or star chart)
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class RadarChart:
    def __init__(self):
        self.colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', '-o']
        self.df = None
        self.range = None
        self.__angles = None
        self.__ax = None
        self.title = None
        self.show = True
        self.labels_from_column = True      # If true, labels are taken from column names
        self.items_from_index = True        # If true, the index if the ref to the item name

    def set_params(self, params):
        if not isinstance(params, dict):
            raise TypeError('params must be in dictionary form')
        for p_name, p_val in params.items():
            if hasattr(self, p_name):
                setattr(self, p_name, p_val)
            else:
                raise KeyError('The parameter {} does not exist in the RadarChart class'.format(p_name))

    def draw(self, df=None, kwargs=None):
        if df is not None:
            self.df = df
        if kwargs is not None:
            self.set_params(kwargs)

        # Labels of items:
        if self.items_from_index:
            df = self.df
        else:
            df = self.df.set_index(self.df.columns[0])
        labels = df.index.values

        # Get categories
        n_item = 0
        if self.labels_from_column:
            categories = df.columns.values
        else:
            categories = df.iloc[0, :]
            n_item += 1
        # Clean from new lines and spaces
        categories = [str(s).rstrip() for s in categories]
        labels = [str(s).rstrip() for s in labels]

        # Base of radar chart
        self.__angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
        self.__angles = np.concatenate((self.__angles, [self.__angles[0]]))

        fig = plt.figure()
        self.__ax = fig.add_subplot(111, polar=True)
        self.__ax.set_thetagrids(self.__angles * 180 / np.pi, categories)
        self.__ax.grid(True)
        ax = self.__ax
        if self.title is not None:
            self.__ax.set_title('Spider Chart: {}'.format(self.title))

        items = range(n_item, len(labels))
        for i in items:
            color = self.colors[i]
            self.add2plot(df.iloc[i, :].values, color)

        ax.legend(labels, loc='upper right', bbox_to_anchor=(0.75, 0.6, 0.5, 0.5))

        mx = max(self.df[categories].max())
        top = np.ceil(mx / 2.) * 2
        ticks = np.arange(top/5, top, top/5)
        plt.yticks(ticks, list(map(lambda x: str(round(x, 1)), ticks)), color="grey", size=7)

        if self.show:
            plt.show()

    def add2plot(self, stats, color='-o'):
        # Close the plot
        stats = np.concatenate((stats, [stats[0]]))
        self.__ax.plot(self.__angles, stats, color, linewidth=2)
        self.__ax.fill(self.__angles, stats, alpha=0.25)

    def df2radar(self, df, index_col=None, data_col=None, filter_zeros=True):
        """
        Receives a dataframe in which each column is a different radar plot and the categories of the plot are in rows
        :param df:        pandas dataframe
        :param index_col: column that will become the categories shown in the radar chart
        :param data_col:  columns to plot
        :return:          None
        """
        if (index_col is None) or (data_col is None):
            raise ValueError("index and data columns must be specified")
        if not isinstance(index_col, str):
            raise TypeError("'index_col' must be a string")
        if isinstance(data_col, str):
            data_col = list(data_col)
        elif not isinstance(data_col, list):
            raise TypeError("'data_col' must be a list or string")

        df_radar = df[[index_col, *data_col]]

        if filter_zeros:
            # As entries consisting only in zeros just clutter the graph they can be eliminated
            idx = df_radar[data_col[0]] > 0
            for dc in data_col[1:]:
                idx = idx | (df_radar[dc] > 0)
            df_radar = df_radar[idx]

        df_radar = df_radar.set_index(index_col).T
        self.draw(df_radar)


class RadarChartOld(object):
    def __init__(self):
        self.colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
        self.df = None
        self.range = None
        self.__angles = None
        self.__ax = None

    def set_params(self, params):
        if not isinstance(params, dict):
            raise TypeError('params must be a dictionary')
        for p_name, p_val in params.values():
            if hasattr(self, p_name):
                setattr(self, p_name, p_val)
            else:
                raise KeyError('The parameter {} does not exist in the SpyderChart class'.format(p_name))

    def draw(self, df=None, rng=None, colors=None):
        self.df = df
        self.range = rng
        if (colors is not None) and isinstance(colors, list):
            self.colors = colors

        # ------- PART 1: Create background
        # number of variable
        categories = list(self.df)[1:]
        N = len(categories)
        # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
        self.__angles = [n / float(N) * 2 * np.pi for n in range(N)]
        self.__angles += self.__angles[:1]

        # Initialise the radar plot
        self.__ax = plt.subplot(111, polar=True)

        # Set the first axis on top:
        self.__ax.set_theta_offset(np.pi / 2)
        self.__ax.set_theta_direction(-1)

        # Draw one axe per variable + add labels labels yet
        plt.xticks(self.__angles[:-1], categories, alpha=0.5)

        self.__ax.grid(True)

        # Draw ylabels
        self.__ax.set_rlabel_position(0)
        if rng is None:
            mx = max(self.df[categories].max())
        else:
            mx = rng

        if mx > 1:
            top = mx + (10 - mx % 10)
            ticks = np.arange(10, top, 10)
            plt.yticks(ticks, list(map(lambda x: str(int(x)), ticks)), color="grey", size=7)
        else:
            ticks = np.arange(0.2, 1, 0.2)
            plt.yticks(ticks, list(map(lambda x: str(round(x, 1)), ticks)), color="grey", size=7)

        plt.ylim(0, mx)
        self.add_plot()
        # Add legend
        self.__ax.legend(loc='upper right', bbox_to_anchor=(0.05, 0.2))
        plt.show()

    def add_plot(self):
        # ------- PART 2: Add plots
        grp_field = list(self.df)[0]

        # Plot each individual = each line of the data
        for idx, clr in zip(self.df.index, self.colors):
            grp_name = self.df.loc[idx, 'group']

            values = self.df.loc[idx].drop(grp_field).values.flatten().tolist()
            values += values[:1]
            self.__ax.plot(self.__angles, values, linewidth=1, linestyle='solid', label=grp_name)
            self.__ax.fill(self.__angles, values, clr, alpha=0.1)


def test_radar():
    # Set data
    df = pd.DataFrame({
        'group': ['A', 'B', 'C', 'D'],
        'var1': [38, 1.5, 30, 4],
        'var2': [29, 10, 9, 34],
        'var3': [8, 39, 23, 24],
        'var4': [7, 31, 33, 14],
        'var5': [28, 15, 32, 14]
    })
    radar = RadarChart()
    radar.draw(df, {'items_from_index': False})
    plt.show(block=True)


def test_radarOld():
    df = pd.DataFrame({
        'group': ['A', 'B', 'C', 'D'],
        'var1': [38, 1.5, 30, 4],
        'var2': [29, 10, 9, 34],
        'var3': [8, 39, 23, 24],
        'var4': [7, 31, 33, 14],
        'var5': [28, 15, 32, 14],
        'var6': [1, 35, 64, 14]
    })
    radar = RadarChartOld()
    radar.draw(df)


if __name__ == '__main__':
    test_radar()
