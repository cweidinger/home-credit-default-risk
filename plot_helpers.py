import matplotlib.pyplot as plt
import seaborn as sns


def plot_pred_target(pred, target):
    repayment = pred[target == 0]
    default = pred[target == 1]
    print('repayment', len(repayment))
    print('default', len(default))
    # plt.hist([repayment, default], color=['b','r'], alpha=0.5)
    sns.distplot(repayment, color='b')
    sns.distplot(default, color='r')

