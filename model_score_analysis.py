import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

def model_discriminate(complex_ho, complex_ot, gabor_ho, gabor_ot, fname,
        plot_title):
    """
    Generate a plot and classification accuracy of HO/OT given
    model scores.
    """
    # generate the discriminability index
    ho_diffs = (complex_ho - gabor_ho) / (complex_ho + gabor_ho)
    ot_diffs = (complex_ot - gabor_ot) / (complex_ot + gabor_ot)

    # first linear classification
    all_diffs = np.concatenate([ot_diffs, ho_diffs])
    all_clss = np.concatenate([np.zeros((len(ot_diffs),)), np.ones((len(ho_diffs),))])

    model = LogisticRegression(C=1e5) # essentially, no regularization
    model = model.fit(all_diffs[:, np.newaxis], all_clss[:, np.newaxis])
    performance = model.score(all_diffs[:, np.newaxis], all_clss[:, np.newaxis])

    print(f'baseline for {fname} is {np.mean(all_clss)}')
    print(f'performance for {fname} is {performance}')

    # now make the plot
    plt.figure()
    _, bins, _ = plt.hist(all_diffs, density=True, bins=30, alpha=0.0) # shared bins
    plt.hist(ot_diffs, bins=bins, density=True, alpha=0.6, color='blue',
            label=f'OT neurons, mean {np.round(ot_diffs.mean(), 3)}')
    plt.hist(ho_diffs, bins=bins, density=True, alpha=0.6, color='red',
            label=f'HO neurons, mean {np.round(ho_diffs.mean(), 3)}')
    plt.legend()
    plt.xlabel('(Complex - Gabor) / Complex + Gabor)')
    plt.ylabel('neuron density')
    plt.title(plot_title)
    plt.savefig(fname)

# generate the plots
if __name__ == '__main__':
    for monkey in ['A', 'E']:
        fname = f'{monkey}_Fig_10.png'
        plot_title = f'Monkey {monkey}'

        complex_ho = np.load(f'saved_performance/{monkey}_FKCNN_corr_HO.npy')
        complex_ot = np.load(f'saved_performance/{monkey}_FKCNN_corr_OT.npy')

        gabor_ho = np.load(f'saved_performance/{monkey}_GCNN_corr_HO.npy')
        gabor_ot = np.load(f'saved_performance/{monkey}_GCNN_corr_OT.npy')

        model_discriminate(complex_ho, complex_ot, gabor_ho, gabor_ot,
                fname, plot_title)
