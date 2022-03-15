import numpy as np
import pandas as pd
from correlations import *
from create_dictionary import *
from normalizations import *
from rank_preferences import *
from weighting_methods import *
from topsis import TOPSIS
from visualizations import *


def main():
    filename = 'input/dataset_cars.csv'
    data = pd.read_csv(filename, index_col = 'Ai')
    df_data = data.iloc[:len(data) - 1, :]
    types = data.iloc[len(data) - 1, :].to_numpy()

    matrix = df_data.to_numpy()

    list_alt_names = [r'$A_{' + str(i) + '}$' for i in range(1, df_data.shape[0] + 1)]
    cols = [r'$C_{' + str(j) + '}$' for j in range(1, data.shape[1] + 1)]

    # part 1 - study with single weighting method

    # Determine criteria weights with chosen weighting method
    weights = entropy_weighting(matrix, types)

    # Create the TOPSIS method object
    topsis = TOPSIS()

    # Calculate alternatives preference function values with TOPSIS method
    pref = topsis(matrix, weights, types)

    # rank alternatives according to preference values
    rank = rank_preferences(pref, reverse = True)

    # save results in dataframe in csv file
    df_results = pd.DataFrame(index = list_alt_names)
    df_results['Pref'] = pref
    df_results['Rank'] = rank
    df_results.to_csv('output/single_weighting_method.csv')

    # part 2 - study with several weighting methods
    # Create a list with weighting methods that you want to explore
    weighting_methods = [
        entropy_weighting,
        #std_weighting,
        critic_weighting,
        gini_weighting,
        merec_weighting,
        stat_var_weighting,
        #cilos_weighting,
        idocriw_weighting,
        angle_weighting,
        coeff_var_weighting
    ]
    

    #df_weights = pd.DataFrame(weights.reshape(1, -1), index = ['Weights'], columns = cols)
    # Create dataframes for weights, preference function values and rankings determined using different weighting methods
    df_weights = pd.DataFrame(index = cols)
    df_preferences = pd.DataFrame(index = list_alt_names)
    df_rankings = pd.DataFrame(index = list_alt_names)

    # Create the TOPSIS method object
    topsis = TOPSIS()
    for weight_type in weighting_methods:
        weights = weight_type(matrix, types)
        df_weights[weight_type.__name__[:-10].upper()] = weights
        pref = topsis(matrix, weights, types)
        rank = rank_preferences(pref, reverse = True)
        df_preferences[weight_type.__name__[:-10].upper()] = pref
        df_rankings[weight_type.__name__[:-10].upper()] = rank
        

    df_weights.to_csv('output/weights.csv')
    df_preferences.to_csv('output/preferences.csv')
    df_rankings.to_csv('output/ranks.csv')

    # plot criteria weights distribution using bex chart
    df_weights_t = df_weights.T
    plot_boxplot(df_weights_t)

    # plot stacked column chart of criteria weights
    plot_barplot(df_weights, 'Weighting methods', 'Weight value', 'Criteria')

    # plot column chart of alternatives rankings
    plot_barplot(df_rankings, 'Alternatives', 'Rank', 'Weighting methods')

    # Plot heatmaps of rankings correlation coefficient
    # Create dataframe with rankings correlation values
    results = copy.deepcopy(df_rankings)
    method_types = list(results.columns)
    dict_new_heatmap_rw = Create_dictionary()

    for el in method_types:
        dict_new_heatmap_rw.add(el, [])

    for i, j in [(i, j) for i in method_types[::-1] for j in method_types]:
        dict_new_heatmap_rw[j].append(weighted_spearman(results[i], results[j]))
            
    df_new_heatmap_rw = pd.DataFrame(dict_new_heatmap_rw, index = method_types[::-1])
    df_new_heatmap_rw.columns = method_types

    # Plot heatmap with rankings correlation
    draw_heatmap(df_new_heatmap_rw, r'$r_w$')


if __name__ == '__main__':
    main()