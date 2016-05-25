import numpy as np
import matplotlib.pyplot as plt

"""
Generate feature dependence plot from trained model. 

Example:
import sklearn
import pandas as pd
import numpy as np
from sklearn import datasets
import sklearn.cross_validation
import sklearn.ensemble
import matplotlib.pyplot as plt
import matplotlib
from fdplot import fdplot

# SETUP
# load data
boston = sklearn.datasets.load_boston()
#print(boston.DESCR)

# Get data into dataframe
columns = [('x', ft) for ft in boston.feature_names.tolist()] + [('t','medv')]
data = np.c_[boston.data, boston.target]

df = pd.DataFrame(data=data, columns=pd.MultiIndex.from_tuples(columns))

# regression model
rgr = sklearn.ensemble.RandomForestRegressor(n_estimators=10, criterion='mse', 
                                       max_depth=None, min_samples_split=2,
                                       min_samples_leaf=1,
                                       min_weight_fraction_leaf=0.0, 
                                       max_features='auto', max_leaf_nodes=None,
                                       bootstrap=True, oob_score=False, n_jobs=1, 
                                       random_state=None, verbose=0, warm_start=False)
rgr.fit(df.x.values, df.t.values.T[0])

# PLOTTING
# setup
data = df.x.loc[:, :]
model = rgr
feature = 'AGE'
ft_ind = data.columns.values.tolist().index(feature)

# Get Curves
curves,  feature_values = model_feature_curves(data.values, model.predict, ft_ind)
curves_centered, centers = center_curves(curves, method='mean', ft_val_ind=150)
curves_ordering = centers.argsort()

# Plot
fig, axes = fdplot(curves_centered, curves_ordering, 
                   centers=centers, feature_values=feature_values,
                   cmap='viridis')
"""


def model_feature_curves(data, model_fun, feature):
    """
        Routine that calculates the curves for given data, vectorized

        Params
        ------
        data: 2-D numpy array
            Dataset,  rows are the samples, columns are featues
        
        model_fun: callable
            Function that gives the response of the model. It has to accept 
            array of samples and return array of scalar responses
        
        feature: integer
            Index of a feature (column in the data array) for which the 
            curves will be generated
            
        
        Returns
        -------
        curves: 2-D numpy array
            Array of curves (1-d arrays), one for each sample in the data
        
        feature_values: numpy array
            Array of unique feature values observed in the data, 
            the domain of all curves.
    """
    N, p = data.shape
    mask = np.zeros(p ,dtype=bool)
    mask[feature] = 1
    
    # Sorted unique values of the selected feature that is being plotted
    feature_values = np.unique(data[:, feature])

    # allocate array for curves
    curves = np.zeros((N, len(feature_values)))
    
    samples = data.copy()
    for j, val in enumerate(feature_values):    
        samples[:, mask] = val
        curves[:,j] = model_fun(samples)   
    return curves,  feature_values


def center_curves(curves, method = 'mean', **kwargs):
    """
        Center curves using one of the selected methods
        
    Params
    ------
    curves: 2-D numpy array
        n*m array, each row of the array is a curve, columns correspond 
        to evaluations of the curves at feature values
        
    method: string, Optional (default='mean')
        'mean' = subtract mean from each curve
        'ft_val' = subtract value given by 'ft_val_ind' from each curve
        'min' = subtract min from each curve
        'max' = subtract max from each curve
        
    ft_val_ind: integer, Optional (default=0), only with method='ft_val'
        for 'ft_val' option, selects which (ft_val-th) value to subtract 
        from each curve
        
    Returns
    -------
    centered_curves: 2-D numpy array 
        (n*m)
        
    centers: 1-D numpy array 
        (length n)    
    """
    
    def center_mean(curves, kwargs):
        center = curves.mean(axis=1)
        return (curves.T - center).T, center
    
    def center_ft_val(curves, kwargs):
        center = curves[:, kwargs.get('ft_val_ind', 0)]
        return (curves.T - center).T, center
    
    def center_min(curves, kwargs):
        center = curves.min(axis=1)
        return (curves.T - center).T, center
    
    def center_max(curves, kwargs):
        center = curves.max(axis=1)
        return (curves.T - center).T, center
    
    switcher={
        'mean': center_mean,
        'ft_val': center_ft_val,
        'min': center_min, 
        'max': center_max
    }
    
    center_fun = switcher.get(method)
    return center_fun(curves, kwargs)

def fdplot(curves_centered, curves_ordering, **kwargs):
    """
    Function fot plotting feature dependence plots
    
    Params
    ------
    curves_centered: numpy array
        n by m array containing curves as rows
    
    curves_ordering: numpy integer array
        Array of n integers that gives the ordering of the 
        curves that will be displayed
        
    centers: numpy array, ptional ()
        Array of n floats that gives the "center" (mean, max, ...) on 
        which each curve is centerd
    
    feature_values: numpy array
        Array of m floats, feature values that give the domain of the curves. 
        Plotted on ax_feature.
        
    
    kwargs: optional plotting parameters
        'figsize' - size of the figure
        'gridspec_kw' - gridspec kwargs
        'cmap' - colormap used in the heatmap
        'xlabel' - label for the x-axes
        'ylabel' - label for the y-axes
        
    The plot can be further manipulated by modyfying the 
    figure and axes returned by the function:
    
    Returns
    -------
    figure, ((ax_centers, ax_curves),(ax_info, ax_feature), ax_colorbar)
    """
    
    # dimensions
    n, m = curves_centered.shape

    #grid of plots
    gridspec_kw = {'width_ratios':[1, 3],
                   'height_ratios':[4, 1], 
                   'wspace':0.0,
                   'hspace':0.0}
    fig, axes = plt.subplots(2,2, 
                            figsize=kwargs.get('figsize', (10,10)),
                            gridspec_kw=kwargs.get('gridspec_kw', gridspec_kw))
    axes = axes.tolist()
    ((ax_centers, ax_curves),(ax_info, ax_feature)) = axes
    
    # main graph - curves
    im = ax_curves.imshow(curves_centered[curves_ordering], 
                          aspect='auto', 
                          cmap=kwargs.get('cmap', 'viridis'), 
                          interpolation="nearest")
    ax_curves.set_xticklabels([])
    ax_curves.set_yticklabels([])
    ax_curves.set_axis_off()

    # colorbar
    cax = fig.add_axes([0.91, 0.29, 0.025, 0.61])
    fig.colorbar(im, cax=cax, orientation='vertical')
    axes.append(cax)

    # bottom graph - feature values
    if 'feature_values' in kwargs:
        ax_feature.plot(np.arange(m), kwargs.get('feature_values'))
    ax_feature.set_xlim((0, m))
    ax_feature.yaxis.tick_right() 
    ax_feature.set_xlabel(kwargs.get('xlabel', '$X_S$ values'))

    # left graph - centers
    if 'centers' in kwargs:
        ax_centers.plot(kwargs.get('centers')[curves_ordering], range(0,n))
    ax_centers.set_ylim((n, 0))
    ax_centers.set_xlim((max(centers), min(centers)))
    ax_centers.xaxis.tick_top()
    ax_centers.set_ylabel(kwargs.get('ylabel', 'curve center values'))

    # bottom left graph - info
    ax_info.set_axis_off()

    # plot points
    #ax_curves.scatter(np.asarray(ft_vals_ordering)[classes], 
    #            curves_ordering[classes], 
    #            c = np.asarray(colors)[classes], marker='o')
    #ax_curves.scatter(np.asarray(ft_vals_ordering)[~classes], 
    #            curves_ordering[~classes], 
    #            c = np.asarray(colors)[~classes], marker='>')
    return fig, axes


def ice_plot(curves, **kwargs):
    """
    Plot the feature dependence curves as graphs (individual 
    conditional expectation (ICE) curves)
    
    Params
    ------
    curves: numpy array
        n by m array containing curves as rows
    
    feature_values: numpy array
        Array of m floats, feature values that give the domain of the curves.
        Plotted on ax_feature.
        
    kwargs: optional plotting parameters
        'figsize' - size of the figure
        'gridspec_kw' - gridspec kwargs
        'xlabel' - label for the x-axes
        'ylabel' - label for the y-axes
        
    The plot can be further manipulated by modyfying the 
    figure and axes returned by the function:
    
    Returns
    -------
    figure, (ax_curves, ax_feature)
    """
    
    n,m = curves.shape
    
    #grid of plots
    gridspec_kw = {'height_ratios':[3, 1], 
                   'wspace':0.0, 
                   'hspace':0.0}
    fig, (ax_curves, ax_feature) = plt.subplots(2,1, 
                                      gridspec_kw=gridspec_kw, 
                                      figsize=kwargs.get('figsize', (6,6)))
    
    # top graph - curves
    for curve in curves:
        ax_curves.plot(np.arange(m), curve)
    ax_curves.set_xticklabels([])
    ax_curves.set_xlim((0, m))
    ax_curves.set_ylabel(kwargs.get('ylabel', 'decision function $\Delta$'))
        
    # bottom graph - feature values
    if 'feature_values' in kwargs:
        ax_feature.plot(np.arange(m), kwargs.get('feature_values'))
    ax_feature.set_xlim((0, m)) 
    ax_feature.set_xlabel(kwargs.get('xlabel', '$X_S$ values'))
    
    return fig, (ax_curves, ax_feature)
    