import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os


def plot_curve(log_path, experiment_name, output):
    fp = open(log_path, 'r')

    train_iterations = []
    curve_loss = []
    curve_acc = []

    test_epoch = []
    curve_mAP = []
    curve_CMC_rank_1 = []
    curve_CMC_rank_5 = []
    curve_CMC_rank_3 = []

    for ln in fp:
        # get train_iterations and train_loss
        if 'Epoch[' in ln and 'Iteration[' in ln and 'Loss: ' in ln:
            element = ln.split(' ')
            epoch = element[4][6:][:-5]
            iteration = element[5][10:][:-5]
            train_iteration = (int(epoch) - 1) * 180 + int(iteration)
            loss = float(element[7][:-1])
            acc = float(element[9][:-1])

            train_iterations.append(train_iteration)
            curve_loss.append(loss)
            curve_acc.append(acc)

        if 'Validation ' in ln:
            element = ln.split(' ')
            epoch = int(element[-1])
            test_epoch.append(epoch)

        if 'mAP: ' in ln:
            element = ln.split(' ')
            ap = float(element[-1].strip()[:-1]) / 100.0
            curve_mAP.append(ap)

        if 'Rank-' in ln:
            element = ln.split(' ')
            rank = element[6][5:].strip()
            if rank is '1':
                rank_1 = float(element[-1].strip()[1:][:-1]) / 100.0
                curve_CMC_rank_1.append(rank_1)
            elif rank is '5':
                rank_5 = float(element[-1].strip()[1:][:-1]) / 100.0
                curve_CMC_rank_5.append(rank_5)
            elif rank is '3':
                rank_3 = float(element[-1].strip()[1:][:-1]) / 100.0
                curve_CMC_rank_3.append(rank_3)

    fp.close()

    plt.subplot(3, 2, 1)
    plt.plot(train_iterations, curve_loss, 'b')
    plt.xlabel('iterations')
    plt.ylabel('train total loss')

    plt.subplot(3, 2, 2)
    plt.plot(train_iterations, curve_acc, 'r')
    plt.xlabel('iterations')
    plt.ylabel('val accuracy')

    plt.subplot(3, 2, 3)
    plt.plot(test_epoch, curve_mAP[0::4], 'm')
    plt.xlabel('epochs')
    plt.ylabel('val mAP')

    plt.subplot(3, 2, 4)
    plt.plot(test_epoch, curve_CMC_rank_1[3::4], 'k')
    plt.xlabel('epochs')
    plt.ylabel('rank_1')

    plt.subplot(3, 2, 5)
    plt.plot(test_epoch, curve_CMC_rank_5[3::4], 'g')
    plt.xlabel('epochs')
    plt.ylabel('rank_5')

    plt.subplot(3, 2, 6)
    plt.plot(test_epoch, curve_CMC_rank_3[3::4], 'y')
    plt.xlabel('epochs')
    plt.ylabel('rank_3')

    # plt.draw()
    fig_path = os.path.join(output, experiment_name + '.png')
    plt.savefig(fig_path)
    print('max rank-1 for mean fusion:{}'.format(max(curve_CMC_rank_1)))


def main():
    parser = argparse.ArgumentParser(description="plot training curves")
    parser.add_argument("--log_file", help="log file", type=str)
    parser.add_argument("--name", help="name of figure", type=str)
    parser.add_argument("--output_path", help="output path", type=str)
    args = parser.parse_args()

    plot_curve(args.log_file, args.name, args.output_path)


if __name__ == '__main__':
    main()
