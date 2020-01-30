import os
import pandas as pd
import datetime
import matplotlib.pyplot as plt

ENCODER_BASE_NAME = 'stacked_autoencoder'
PATH_OUTPUT = 'output_data'
PATH_INPUT = 'input_data'
PATH_IMAGE = 'images'

seed = 42


def logging(input):
    filename = os.path.join(os.getcwd(), PATH_OUTPUT, "log.txt")
    input = str(datetime.datetime.now()) + ': ' + input
    print(input)
    with open(filename, 'a') as out_file:
        out_file.writelines(input + '\n')


def save_fig(fig_id, dpi=150, tight_layout=False):
    path = os.path.join(os.getcwd(), PATH_IMAGE, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=dpi)


def get_df_from_csv(filename):
    path = os.path.join(os.getcwd(), PATH_INPUT, filename)

    def periodToFractionOfYear(period):
        if period[-7:][:3][-1] == 'M':
            return float(period[-7:][:3][:2]) / 12.
        elif period[-7:][:3][-1] == 'Y':
            return float(period[-7:][:3][:2])

    df = {}
    df['data'] = pd.read_csv(path, sep=';')
    df['data']['TradeDate'] = pd.to_datetime(df['data']['TradeDate'], format='%Y%m%d')
    df['data'].set_index('TradeDate', inplace=True)
    df['data'].sort_index(inplace=True)

    df['ST'] = pd.DataFrame(data=df['data'].columns, columns=['symbol'])
    df['ST'] = df['ST'].merge(pd.DataFrame(data=[float(x[-4:]) for x in df['data'].columns], columns=['strike']),
                              left_index=True, right_index=True)
    df['ST'] = df['ST'].merge(pd.DataFrame(data=[x[-7:][:3] for x in df['data'].columns], columns=['swapPeriodSymbol']),
                              left_index=True, right_index=True)
    df['ST'] = df['ST'].merge(
        pd.DataFrame(data=list(map(periodToFractionOfYear, df['ST']['swapPeriodSymbol'])), columns=['swapPeriod']),
        left_index=True, right_index=True)
    df['ST'] = df['ST'].merge(
        pd.DataFrame(data=[x[-11:][:3] for x in df['data'].columns], columns=['optionPeriodSymbol']), left_index=True,
        right_index=True)
    df['ST'] = df['ST'].merge(
        pd.DataFrame(data=list(map(periodToFractionOfYear, df['ST']['optionPeriodSymbol'])), columns=['optionPeriod']),
        left_index=True, right_index=True)
    df['ST'].set_index('symbol', inplace=True)
    df['ST'] = df['ST'].sort_index()
    return df


def get_encoder_name(n_bottleneck, noise_stddev):
    return ENCODER_BASE_NAME + "_noiseStdev" + str(noise_stddev) + "_bl" + str(n_bottleneck)


def plot_swaption_volas(dfInput, trade_date):
    df = pd.DataFrame(data=dfInput['data'].loc[dfInput['data'].index == trade_date].values[0], columns=['value'],
                      index=dfInput['data'].loc[dfInput['data'].index == trade_date].columns)
    df = df.merge(dfInput['ST'], left_index=True, right_index=True)
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    for idxPlot, iOptionPeriodSymbol in enumerate(df.optionPeriodSymbol.unique()):
        for iSwapPeriodSymbol in df.swapPeriodSymbol.unique():
            iRow = int(idxPlot / 3)
            iColumn = idxPlot % 3
            label = iOptionPeriodSymbol + '_' + iSwapPeriodSymbol
            df.loc[df.optionPeriodSymbol == iOptionPeriodSymbol][df.swapPeriodSymbol == iSwapPeriodSymbol].plot(
                ax=axs[iRow, iColumn], x='strike', y='value', style='*-', label=label, sharex=True, sharey=True)
    save_fig(str(ENCODER_BASE_NAME) + '_swaption_volas_' + str(trade_date))


def plot_swaption_volas2(dfInput, trade_dates, labels, optionPeriodSymbol, swapPeriodSymbol):
    dfMerged = dfInput['ST']
    for trade_date in trade_dates:
        dfMerged = dfMerged.merge(
            pd.DataFrame(data=dfInput['data'].loc[dfInput['data'].index == trade_date].values[0], columns=[trade_date],
                         index=dfInput['data'].loc[dfInput['data'].index == trade_date].columns), left_index=True,
            right_index=True)
    fig, axs = plt.subplots(figsize=(10, 5))
    for trade_date, label in zip(trade_dates, labels):
        dfMerged.sort_values(by='strike').loc[dfMerged.optionPeriodSymbol == optionPeriodSymbol][
            dfMerged.swapPeriodSymbol == swapPeriodSymbol].plot(ax=axs, x='strike', y=trade_date, style='*-',
                                                                label=label, sharex=True, sharey=True)
    save_fig(str(ENCODER_BASE_NAME) + '_swaption_volas2_' + str(optionPeriodSymbol) + '_' + str(swapPeriodSymbol))


def plot_swaption_volas3(dfInput, dfResults, n_bottlenecks, noise_stddevs, trade_date, optionPeriodSymbol,
                         swapPeriodSymbol):
    dfMerged = dfInput['ST']
    dfMerged = dfMerged.merge(
        pd.DataFrame(data=dfInput['data'].loc[dfInput['data'].index == trade_date].values[0], columns=['input'],
                     index=dfInput['data'].loc[dfInput['data'].index == trade_date].columns), left_index=True,
        right_index=True)
    title = 'option maturity: ' + str(optionPeriodSymbol) + ', swap maturity: ' + str(swapPeriodSymbol)

    for n_bottleneck, noise_stddev in zip(n_bottlenecks, noise_stddevs):
        encoder_name = get_encoder_name(n_bottleneck, noise_stddev)
        dfMerged = dfMerged.merge(pd.DataFrame(
            data=dfResults[encoder_name]['data'].loc[dfResults[encoder_name]['data'].index == trade_date].values[0],
            columns=['bl:' + str(n_bottleneck) + ', stdev:' + str(noise_stddev)],
            index=dfResults[encoder_name]['data'].loc[dfResults[encoder_name]['data'].index == trade_date].columns),
            left_index=True, right_index=True)

    fig, axs = plt.subplots(figsize=(10, 5))
    dfMerged.sort_values(by='strike').loc[dfMerged.optionPeriodSymbol == optionPeriodSymbol][
        dfMerged.swapPeriodSymbol == swapPeriodSymbol].plot(ax=axs, x='strike', y='input', style='*-', sharex=True,
                                                            sharey=True, title=title)
    for n_bottleneck, noise_stddev in zip(n_bottlenecks, noise_stddevs):
        dfMerged.sort_values(by='strike').loc[dfMerged.optionPeriodSymbol == optionPeriodSymbol][
            dfMerged.swapPeriodSymbol == swapPeriodSymbol].plot(ax=axs, x='strike',
                                                                y='bl:' + str(n_bottleneck) + ', stdev:' + str(
                                                                    noise_stddev), style='*-', sharex=True, sharey=True)
    save_fig(str(ENCODER_BASE_NAME) + '_swaption_volas3_' + str(optionPeriodSymbol) + '_' + str(
        swapPeriodSymbol) + '_' + str(trade_date))


def plot_swaption_volas4(dfInput, dfResults, n_bottleneck, noise_stddev, trade_date):
    encoder_name = get_encoder_name(n_bottleneck, noise_stddev)
    dfInputMerged = pd.DataFrame(data=dfInput['data'].loc[dfInput['data'].index == trade_date].values[0],
                                 columns=['valueInput'],
                                 index=dfInput['data'].loc[dfInput['data'].index == trade_date].columns)
    dfInputMerged = dfInputMerged.merge(dfInput['ST'], left_index=True, right_index=True)
    dfInputMerged = dfInputMerged.merge(pd.DataFrame(
        data=dfResults[encoder_name]['data'].loc[dfResults[encoder_name]['data'].index == trade_date].values[0],
        columns=['valuePrediction'],
        index=dfResults[encoder_name]['data'].loc[dfResults[encoder_name]['data'].index == trade_date].columns),
        left_index=True, right_index=True)
    nRows = len(dfInputMerged.optionPeriodSymbol.unique())
    nColumns = len(dfInputMerged.swapPeriodSymbol.unique())
    fig, axs = plt.subplots(nRows, nColumns, figsize=(30, 40))
    for iRow, iOptionPeriodSymbol in enumerate(dfInputMerged.optionPeriodSymbol.unique()):
        for iColumn, iSwapPeriodSymbol in enumerate(dfInputMerged.swapPeriodSymbol.unique()):
            label = iOptionPeriodSymbol + '_' + iSwapPeriodSymbol
            dfInputMerged.loc[dfInputMerged.optionPeriodSymbol == iOptionPeriodSymbol][
                dfInputMerged.swapPeriodSymbol == iSwapPeriodSymbol].plot(ax=axs[iRow, iColumn], x='strike',
                                                                          y='valueInput', style='*-',
                                                                          label='In_' + label, sharex=True, sharey=True)
            dfInputMerged.loc[dfInputMerged.optionPeriodSymbol == iOptionPeriodSymbol][
                dfInputMerged.swapPeriodSymbol == iSwapPeriodSymbol].plot(ax=axs[iRow, iColumn], x='strike',
                                                                          y='valuePrediction', style='--',
                                                                          label='Pred_' + label, sharex=True,
                                                                          sharey=True)
    save_fig(str(ENCODER_BASE_NAME) + '_swaption_volas4_' + str(encoder_name) + '_' + str(trade_date))


def plot_loss(n_bottleneck, noise_stddev):
    encoder_name = get_encoder_name(n_bottleneck, noise_stddev)
    df_loss_train_test_epoch = pd.read_csv(os.path.join(os.getcwd(), PATH_OUTPUT, encoder_name + ".csv"), sep=';')
    df_loss_train_test_epoch.plot(x='iEpoch', y=['loss_train', 'loss_test'], title='reconstruction loss')
    save_fig(str(ENCODER_BASE_NAME) + '_loss_' + str(encoder_name))


def plot_hist(dfResults, index_train, index_test, n_bottleneck, noise_stddev):
    encoder_name = get_encoder_name(n_bottleneck, noise_stddev)
    fig, axs = plt.subplots(figsize=(10, 5))
    dfResults[encoder_name]['zValue'].loc[index_train].plot.hist(y='zValue', label='train', ax=axs, log=True,
                                                                 title='bl: ' + str(n_bottleneck) + ', stdev: ' + str(
                                                                     noise_stddev))
    dfResults[encoder_name]['zValue'].loc[index_test].plot.hist(y='zValue', label='test', ax=axs, log=True)
    dfResults[encoder_name]['zValue'].loc[dfResults[encoder_name]['zValue'].index < '1900'].plot.hist(y='zValue',
                                                                                                      label='injected points',
                                                                                                      ax=axs, log=True)
    axs.set(xlabel="z-value")
    save_fig(str(ENCODER_BASE_NAME) + '_hist_' + str(encoder_name))


def plot_feature_vector(dfInput, trade_date):
    from helper import save_fig
    pd.DataFrame(dfInput['data'].loc[dfInput['data'].index == '20181228'].reset_index().T.iloc[1:].values).plot(
        legend=False)
    save_fig(str(ENCODER_BASE_NAME) + '_features_' + str(trade_date))
