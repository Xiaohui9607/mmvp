import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
import glob
import os

""" short time fourier transform of audio signal """


def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))
    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(int(np.floor(frameSize / 2.0))), sig)

    # cols for windowing
    cols = np.ceil((len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))

    frames = stride_tricks.as_strided(samples, shape=(int(cols), frameSize),
                                      strides=(samples.strides[0] * hopSize, samples.strides[0])).copy()
    frames *= win
    return np.fft.rfft(frames)


""" scale frequency axis logarithmically """


def logscale_spec(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins - 1) / max(scale)
    scale = np.unique(np.round(scale))

    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):
        if i == len(scale) - 1:
            newspec[:, i] = np.sum(spec[:, int(scale[i]):], axis=1)
        else:
            newspec[:, i] = np.sum(spec[:, int(scale[i]):int(scale[i + 1])], axis=1)

    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins * 2, 1. / sr)[:freqbins + 1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale) - 1:
            freqs += [np.mean(allfreqs[int(scale[i]):])]
        else:
            freqs += [np.mean(allfreqs[int(scale[i]):int(scale[i + 1])])]

    return newspec, freqs


""" plot spectrogram"""


def plotstft(audio_path, binsize=2 ** 10, plotpath=None, colormap="jet", fig_name="spectrogram"):
    samplerate, samples = wav.read(audio_path)
    samples = np.delete(samples, 1, 1)

    s = stft(samples, binsize)

    sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)
    ims = 20. * np.log10(np.abs(sshow) / 10e-6)  # amplitude to decibel
    timebins, freqbins = np.shape(ims)
    print("timebins: ", timebins)
    print("freqbins: ", freqbins)

    '''
    plt.figure(figsize=(15, 7.5))
    plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
    plt.colorbar()

    plt.xlabel("time (s)")
    plt.ylabel("frequency (hz)")
    plt.xlim([0, timebins-1])
    plt.ylim([0, freqbins])

    xlocs = np.float32(np.linspace(0, timebins-1, 5))
    plt.xticks(xlocs, ["%.02f" % l for l in ((xlocs*len(samples)/timebins)+(0.5*binsize))/samplerate])
    ylocs = np.int16(np.round(np.linspace(0, freqbins-1, 10)))
    plt.yticks(ylocs, ["%.02f" % freq[i] for i in ylocs])

    if plotpath:
        plt.savefig("../../data/k",bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    plt.clf()
    '''
    return ims


if __name__ == '__main__':
    filepath = "../../data/CY101/rc_data/tupperware_coffee_beans/trial_1/exec_1/shake/hearing/1301275317325020.wav"
    ims = plotstft(filepath)

    # TODO a folder does't have: "exec_5", thus temporary excluded from all folders
    '''
    executions = ["exec_1", "exec_2", "exec_3", "exec_4"]
    behaviors = ["shake", "crush", "grasp", "hold", "lift_slow", "low_drop", "poke", "push", "shake", "tap"]
    modality = "hearing"  # audio

    category_object_list = os.listdir(data_path_root_folder_non_vision)

    # removing hidden file, e.g. .DS Store
    for item in category_object_list:
        if item.startswith("."):
            category_object_list.remove(item)

    for category_object in category_object_list:
        print(category_object)
        for execution in executions:
            for behavior in behaviors:
                folder = data_path_root_folder_non_vision + category_object + "/trial_1/" + \
                         execution + "/" + behavior + "/" + modality + "/"
                os.chdir(folder)
                for file in glob.glob("*.wav"):
                    filepath = folder + file
                    plotpath = plot_path_root_folder + category_object + "/trial_1/" + execution \
                               + "/" + behavior + "/" + modality + "/"
                    spectrogram_name = category_object + "_trial_1" + "_" + execution + "_" + behavior + "_" + modality
                    if not os.path.exists(plotpath):
                        os.makedirs(plotpath)
                    ims = plotstft(filepath, plotpath=plotpath, fig_name=spectrogram_name)
    '''