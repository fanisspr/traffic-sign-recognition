'''
- Plot metric graphs. 
- Change the home folder to GTSRB or DFG 
  depending on the dataset the model was trained with.
'''

# import matplotlib as mpl
# from cProfile import label
import os
import re
import pandas as pd
import numpy as np
from cycler import cycler
import itertools
import matplotlib.pyplot as plt
import QT_QPA_PLATFORM_PLUGIN_PATH


def get_metric(file: str, metric: str) -> float:
    '''
    Gets metric data from txt file
    '''
    # pattern = re.compile(r'(?<=loss: ).{5,6}(e-\d+)?')
    # pattern = re.compile(rf'(?<={metric}: ).{5,6}(e-\d+)?')
    pattern = re.compile(r'(?<=%s: ).{5,6}(e-\d+)?' % metric)
    match = pattern.finditer(file)
    metr = [num for num in match][0]
    return float(metr.group())


def my_plotter(loss, acc, **kwargs):
    """
    A helper function to make a graph.
    """
    # make a plot
    ax.plot(loss,
            label=f'{optimizer} loss',
            **kwargs)

    ax2.plot(acc,
             label=f'{optimizer} acc',
             **kwargs)

    return


'''
Folders and Files
'''
DATASET = 'DFG'
homedir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(homedir, DATASET)
rundir = os.path.join(datadir, 'Models')
if os.path.isdir(rundir):
    models = [x for x in os.listdir(rundir) if os.path.isdir(
        os.path.join(rundir, x))]

optimizers = ['Adam', 'RMSprop', 'SGD', 'SGDW']
train_val_file = 'results.csv'
test_file = 'testing.txt'


'''
Plot parameters
'''
default_cycler = (cycler(color=['b', 'g', 'r', 'y']) +
                  cycler(markevery=[2]*4) +
                  cycler(ls=['-', '--', ':', '-.']))


# cycler(marker=['s', 'o', 'v', 'x']) +
# marker = itertools.chain(['s', 'o', 'v', 'x', '*', 's', 'o', 'v']*4)
marker = itertools.cycle(['s', 'o', 'v', 'x', '*', 's', 'o', 'v'])
plt.rc('axes', prop_cycle=default_cycler)
plt.rc('axes', linewidth=0.8)
# plt.rc('axes', markevery=4)
plt.rc('legend', markerscale=0.7)
plt.rc('legend', framealpha=0.6)
plt.rc('legend', labelspacing=0.4)
# plt.rc('legend', borderpad=0.6)
plt.rc('figure', figsize=(6.3, 4.3))

'''
Plotting
'''
# print(models)
for model in models:
    print(model)
    for step in ['Train', 'Validation']:
        # print(f'{step} step:')
        # create figure and axis objects with subplots()
        fig, ax = plt.subplots()
        # twin object for two different y-axis on the sample plot
        ax2 = ax.twinx()

        for optimizer in optimizers:
            # print(model, optimizer)
            curr_dir = os.path.join(rundir, model, optimizer)
            train_val_metrics = pd.read_csv(
                os.path.join(curr_dir, train_val_file))
            # print(train_val_metrics.head())

            # Print test loss, accuracy
            if step == 'Validation':
                with open(os.path.join(curr_dir, test_file), "r") as f:
                    txt = f.read()
                test_loss = get_metric(txt, 'loss')
                test_acc = get_metric(txt, 'accuracy')
                print(
                    f'{model} - {optimizer}: test loss, test accuracy = {test_loss}, {test_acc}')

            # Plot
            if step == 'Train':
                ax.plot(train_val_metrics.loss,
                        label=f'{optimizer} loss',
                        # markevery=5,
                        marker=next(marker))

                ax2.plot(train_val_metrics.accuracy,
                         label=f'{optimizer} acc',
                         #  markevery=5,
                         marker=next(marker))

                # my_plotter(train_val_metrics.loss,
                #            train_val_metrics.accuracy,
                #            markevery=5)
            else:
                ax.plot(train_val_metrics.val_loss,
                        label=f'{optimizer} loss',
                        # markevery=5,
                        marker=next(marker))

                ax2.plot(train_val_metrics.val_accuracy,
                         label=f'{optimizer} acc',
                         #  markevery=5,
                         marker=next(marker))
                # my_plotter(train_val_metrics.val_loss,
                #            train_val_metrics.val_accuracy,
                #            markevery=5,
                #            marker=next(marker))

        ax.set_title(f'{model}: {step}', fontsize=14)
        # set x-axis label
        ax.set_xlabel("epoch")
        # set y-axis label
        ax.set_ylabel(f'{step} loss')
        ax2.set_ylabel(f'{step} accuracy')

        # lines, labels = ax.get_legend_handles_labels()
        # lines2, labels2 = ax2.get_legend_handles_labels()
        # ax2.legend(lines + lines2, labels + labels2, bbox_to_anchor=(1.0, 0.5),
        #            loc='center right', fontsize=9)
        fig.legend(bbox_to_anchor=(1.0, 0.55),
                   loc='center right',
                   bbox_transform=ax.transAxes,
                   fontsize=9)

        ax.set_xlim(0)
        if (model == 'CNN' or model == 'STN') and step == 'Train':
            ax.set_ylim(-0.01, 0.4)
        else:
            ax.set_ylim(0.002, 0.4)
        ax2.set_ylim(0.85, 1.01)
        ax.margins(0.1, 0.1)
        ax.grid(alpha=.5)
        ax.minorticks_on()
        plt.tight_layout()
        plt.show()
        # save the plot as a file
        save_plot_dir = os.path.join(homedir, 'screenshots', f'{DATASET}')
        if not os.path.exists(save_plot_dir):
            os.mkdir(save_plot_dir)
        fig.savefig(os.path.join(save_plot_dir, f'{DATASET}_{model}_{step}.png'),
                    format='png',
                    bbox_inches='tight')

    print('-'*20)
plt.close('all')
