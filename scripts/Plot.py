import scipy.stats as stats
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('SVG')
import matplotlib.pyplot as plt

def Dot3CMatrix(x, y, z, xlabel = 'x', ylabel = 'y', zlabel = 'z', fname = 'dot3cmatrix.png'):
    data = pd.DataFrame(columns=[xlabel, ylabel, zlabel])
    for i in range(len(x)):
        line = {xlabel : int(x[i]), ylabel : int(y[i]), zlabel : int(z[i])}
        data = data.append(line, ignore_index=True)
    fig = sns.pairplot(data=data, vars=data.iloc[:,0:3], diag_kind='kde', markers='+')
#    dotplot = fig.get_figure() #有错
#    dotplot.savefig(fname, dpi=400)
    fig.savefig(fname, dpi=400)

def DistributionPlot(data,columns):
    x = np.random.randn(100)
    x = pd.Series(x, name = "x variable")
    ax = sns.displot(x)
    hist_fig = ax.get_figure()
    hist_fig.savefig('tt.png')
    '''
    df = pd.DataFrame(data, columns=columns)
    print(df)
    fig = sns.displot(data=df, x=columns[3], kde=True)

    displot = fig.get_figure()
    displot.savefig('test.png')
#    fig.remove()
    '''

def HistPlot(data, name):
    ax = sns.histplot(data, kde=True)
    fig = ax.get_figure()
    fig.savefig(name, dpi=400)

def LinearRegMD(data, name, x, y):
    fig = sns.jointplot(x=x, y=y, data=data,
                  kind="reg", truncate=False,
                  xlim=(0, 100), ylim=(0, 100),
                  color="m", height=7)
    fig.savefig(name, dpi=400)

