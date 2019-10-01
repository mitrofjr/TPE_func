import shap
import pickle
import operator
import numpy as np
import pandas as pd
import xgboost as xgb
from math import log
from sklearn.model_selection import train_test_split
from matplotlib import pylab as plt
from matplotlib.ticker import FormatStrFormatter

    
PARAMETERS = [
    'RS_alpha_plus_beta',
    'RS_omega',
    'RS_alpha',
    'XC_LDA_X',
    'XC_GGA_X_B88',
    'XC_GGA_X_PBE',
    'XC_GGA_X_B86',
    'XC_GGA_X_PW91',
    'XC_GGA_X_SOGGA',
    'XC_LDA_C_XALPHA',
    'XC_LDA_C_VWN',
    'XC_LDA_C_RPA',
    'XC_GGA_C_PBE',
    'XC_GGA_C_PW91',
    'XC_GGA_C_LYP',
]


def get_best_loss(l):
    best_losses = []
    for k in range(1, len(l) + 1):
        best_losses.append(min(l[:k]))
    return best_losses


def get_average_loss(l):
    best_losses = []
    for k in range(1, len(l) + 1):
        best_losses.append(sum(l[:k]) / k)
    return best_losses


def learning_curve(trials_filename, title=''):
    with open(trials_filename, 'rb') as f:
        trials = pickle.load(f, encoding='latin1')

    x = [i for i, t in enumerate(trials.trials) ]
    y = [log(t['result']['loss']) for t in trials.trials ]
    y_min = get_best_loss(y)
    y_avg = get_average_loss(y)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(title)
    ax.plot(x, y, 'ro')
    ax.plot(x, y_min, 'b-', label='min value')
    ax.plot(x, y_avg, 'g-', label='avg value')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax.set_xlabel('iteration')
    ax.set_ylabel('log(loss), a.u.')
    ax.legend(loc="upper right")
    fig.savefig(f'{title}.png', dpi=300)


def get_X_y_sorted(filename):
    with open(filename, 'rb') as f:
        trials = pickle.load(f, encoding='latin1')

    best_indices = np.argsort([t['result']['loss'] for t in trials.trials]).tolist()
    best_trials = [trials.trials[i] for i in best_indices]

    X, y = [], []

    for i, itr in enumerate(best_trials):
        params = [p[1] for p in sorted(itr['result']['params'].items(), key=operator.itemgetter(0))]
        X.append(params[:-1])
        y.append(log(itr['result']['loss']))

    X = pd.DataFrame(X, columns=PARAMETERS)
    y = np.array(y)

    return X, y


def shap_analysis(trials_filename, title=''):
    X, y = get_X_y_sorted(trials_filename)
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.2, random_state=10)
    dt = xgb.DMatrix(Xt, label=yt)
    dv = xgb.DMatrix(Xv, label=yv)
    model = xgb.XGBRegressor()
    model.fit(Xt, yt)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(Xv)
    shap_interaction_values = explainer.shap_interaction_values(Xv)

    plt.figure()
    plt.title(f'{title}')
    shap.summary_plot(shap_values, Xv, auto_size_plot=True, max_display=10, show=False)
    plt.tight_layout()
    plt.savefig(f'{title}_summary_plot.png', dpi=300)

    plt.figure()
    plt.title(f'{title}')
    shap.summary_plot(shap_values, Xv, plot_type="bar", auto_size_plot=True, max_display=10, show=False)
    plt.tight_layout()
    plt.savefig(f'{title}_summary_plot_bar.png', dpi=300)
    
    tmp = np.abs(shap_interaction_values).sum(0)
    for i in range(tmp.shape[0]):
        tmp[i,i] = 0
    inds = np.argsort(-tmp.sum(0))[:50]
    tmp2 = tmp[inds,:][:,inds]
    plt.figure(figsize=(5, 5))
    plt.imshow(tmp2)
    plt.yticks(range(tmp2.shape[0]), X.columns[inds], rotation=0, horizontalalignment="right")
    plt.xticks(range(tmp2.shape[0]), X.columns[inds], rotation=90, horizontalalignment="left")
    plt.gca().xaxis.tick_top()
    plt.title(title)
    plt.colorbar(fraction=0.045, pad=0.05)
    plt.tight_layout()
    plt.savefig(f'{title}_interaction_values.png', dpi=300)