import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_barplot(df_plot, x_name, y_name, title):
    """
    Display column stacked column plot of weights for criteria for `x_name == Weighting methods`
    and column plot of ranks for alternatives `x_name == Alternatives`

    Parameters
    ----------
        df_plot : dataframe
            dataframe with criteria weights for different weighting methods
            or with alternaives rankings for different weighting methods
        x_name : str
            name of x axis, Alternatives or Weighting methods
        y_name : str
            name of y axis, Ranks or Weight values
        title : str
            name of plot title, Weighting methods or Criteria
    """
    list_rank = np.arange(1, len(df_plot) + 1, 1)
    stacked = True
    width = 0.5
    if x_name == 'Alternatives':
        stacked = False
        width = 0.8
    else:
        df_plot = df_plot.T
    ax = df_plot.plot(kind='bar', width = width, stacked=stacked, edgecolor = 'black', figsize = (9,4))
    ax.set_xlabel(x_name, fontsize = 12)
    ax.set_ylabel(y_name, fontsize = 12)

    if x_name == 'Alternatives':
        ax.set_yticks(list_rank)

    ax.set_xticklabels(df_plot.index, rotation = 'horizontal')
    ax.tick_params(axis = 'both', labelsize = 12)

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
    ncol=4, mode="expand", borderaxespad=0., edgecolor = 'black', title = title, fontsize = 12)

    ax.grid(True, linestyle = '--')
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig('output/histogram_' + x_name + '.pdf')
    plt.show()


def draw_heatmap(df_new_heatmap, title):
    """
    Display heatmap with correlations of compared rankings generated using different methods

    Parameters
    ----------
    df_new_heatmap : dataframe
        dataframe with correlation values between compared rankings
    title : str
        title of plot containing name of used correlation coefficient
    """
    plt.figure(figsize = (8, 6))
    sns.set(font_scale=1.1)
    heatmap = sns.heatmap(df_new_heatmap, annot=True, fmt=".4f", cmap="RdYlBu",
                          linewidth=0.5, linecolor='w')
    plt.yticks(va="center")
    plt.xlabel('Weighting methods')
    plt.title('Correlation coefficient: ' + title)
    plt.tight_layout()
    plt.savefig('output/' + 'correlations.pdf')
    plt.show()