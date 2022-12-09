def make_plot(history,metric,name):
    """
    Accuracy plot of model 
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    acc, val_acc = history.history[metric], history.history[f'val_{metric}']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(20, 13))
    plt.plot(epochs, acc, label='Training accuracy', marker='o')
    plt.plot(epochs, val_acc, label='Validation accuracy', marker='o')
    plt.legend()
    plt.title('Dokładność trenowania i walidacji')
    plt.xlabel('Epochs')
    plt.ylabel(f'{metric}')
    plt.savefig(f'plots/{metric[i]}.png)




def make_stacked_plot(history,metrics,name):
    """
    Accuracy plot of model 
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    acc1, val_acc1 = history.history[metrics[0]], history.history[f'val_{metrics[0]}']
    acc2, val_acc2 = history.history[metrics[1]], history.history[f'val_{metrics[1]}']
    acc3, val_acc3 = history.history[metrics[2]], history.history[f'val_{metrics[2]}']

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(20, 13))
    plt.plot(epochs, acc1, label='Training accuracy', marker='o')
    plt.plot(epochs, val_acc1, label='Validation accuracy', marker='o')
    plt.plot(epochs, acc2, label='Training accuracy', marker='o')
    plt.plot(epochs, val_acc2, label='Validation accuracy', marker='o')
    plt.plot(epochs, acc3, label='Training accuracy', marker='o')
    plt.plot(epochs, val_acc3, label='Validation accuracy', marker='o')
    plt.legend()
    plt.title('Dokładność trenowania i walidacji')
    plt.xlabel('Epochs')
    plt.ylabel(f'{metics}')
    plt.savefig(f'plots/{name}.png)
