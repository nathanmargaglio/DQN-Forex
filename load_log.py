#!/usr/bin/env python3

import click
import os

@click.group()
def cli():
    pass

@click.command()
def ls():
    for f in os.listdir('sessions'):
        click.echo(f)

@click.command()
@click.option('--n', default=5, help='rows to print')
@click.option('--head', default=False, help='print head instead of tail')
@click.option('--plot', default=0, help='plot the data')
@click.argument('name')
def load(name, n, head, plot):
    logpath = 'sessions/' + name + '/log.json'
    if name not in os.listdir('sessions'):
        click.echo('Session not found (try "ls" command)')
        return

    import pandas as pd
    import json

    with open(logpath) as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    if not head:
        print(df.tail(n))
    else:
        print(df.head(n))

    print('--------------------')
    print(df.describe())

    if plot:
        try:
            import matplotlib.pyplot as plt
            x = df.episode
            y = pd.rolling_mean(df.episode_reward, plot)
            plt.plot(x[y.notnull()], y[y.notnull()])
            plt.show()
        except:
            click.echo('Matplotlib error, switching to terminalplot')
            from terminalplot import plot as tplot
            x = df.episode
            y = pd.rolling_mean(df.episode_reward, plot)
            tplot(list(x[y.notnull()]), list(y[y.notnull()]))

cli.add_command(ls)
cli.add_command(load)

if __name__ == '__main__':
    cli()

