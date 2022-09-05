import string
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate, io, signal
from scipy.cluster.vq import kmeans, whiten
from scipy.fft import fft, fftfreq, ifft, irfft, rfft, rfftfreq
from scipy.interpolate import interp1d
from scipy.signal import blackman, spectrogram


def plot(samp, data):
    length = data.shape[0]
    time = np.linspace(0., length, data.shape[0])
    plt.plot(time, data[:], label="Channel")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.show()


def fit_fn(samp, data):
    length = data.shape[0] / samp
    x = np.linspace(0., length, data.shape[0])
    y = data[:]
    f = interp1d(x, y)
    f2 = interp1d(x, y, kind="cubic")
    plt.plot(x, y, 'o', x, f(x), '-', x, f2(x), '--')
    plt.legend(['data', 'linear', 'cubic'], loc="best")
    plt.show()


def fit_beizer(samp, data):
    length = data.shape[0] / samp
    x = np.linspace(0., length, data.shape[0])
    y = data[:]
    tck = interpolate.splrep(x, y, s=0)
    ynew = interpolate.splev(x, tck, der=0)
    plt.figure()
    plt.plot(x, y, 'x', x, ynew)
    plt.legend(['Linear', 'Cubic Spline'], loc="best")
    # plt.axis([-0.05, 6.33, -1.05, 1.05])
    plt.title('Cubic-spline interpolation')
    plt.show()


def kms(samp, data):
    length = data.shape[0] / samp
    x = np.linspace(0., length, data.shape[0])
    y = data[:]
    whitened = whiten([x, data])
    codebook, distortion = kmeans(whitened, 2)

    plt.scatter(whitened[:, 0], whitened[:, 1])
    plt.scatter(codebook[:, 0], codebook[:, 1], c='r')
    plt.show()


def extract_peak_frequency(samp, data):
    fft_data = np.fft.fft(data)
    freqs = np.fft.fftfreq(len(data))

    peak_coefficient = np.argmax(np.abs(fft_data))
    peak_freq = freqs[peak_coefficient]

    return abs(peak_freq * samp)


def denoise(samp, data):
    dt = 1.0 / samp
    t = np.linspace(0, data.shape[0] / samp, data.shape[0])
    x = data[:]
    minsignal, maxsignal = np.min(x), np.max(x)
    n = len(t)

    fhat = np.fft.fft(x, n)  #computes the fft
    psd = fhat * np.conj(fhat) / n
    freq = (1 / (dt * n)) * np.arange(n)  #frequency array
    idxs_half = np.arange(1, np.floor(n / 2), dtype=np.int32)

    # plt.plot(t, x, color='b', lw=0.5, label='Noisy Signal')
    # # plt.plot(t, signal_clean, color='r', lw=1, label='Clean Signal')
    # plt.xlabel('t axis')
    # plt.ylabel('Vals')
    # plt.legend()
    # plt.show()

    # plt.plot(freq[idxs_half],
    #          np.abs(psd[idxs_half]),
    #          color='b',
    #          lw=0.5,
    #          label='PSD noisy')
    # plt.xlabel('Frequencies in Hz')
    # plt.ylabel('Amplitude')
    # plt.legend()
    # plt.show()

    threshold = 100
    psd_idxs = psd > threshold  #array of 0 and 1
    psd_clean = psd * psd_idxs  #zero out all the unnecessary powers
    fhat_clean = psd_idxs * fhat  #used to retrieve the signal

    signal_filtered = np.fft.ifft(fhat_clean)  #inverse fourier transform

    # plt.plot(freq[idxs_half],
    #          np.abs(psd_clean[idxs_half]),
    #          color='r',
    #          lw=1,
    #          label='PSD clean')
    # plt.xlabel('Frequencies in Hz')
    # plt.ylabel('Amplitude')
    # plt.legend()
    # plt.show()

    plt.plot(t,
             signal_filtered,
             color='r',
             lw=1,
             label='Clean Signal Retrieved')
    plt.ylim([minsignal, maxsignal])
    plt.xlabel('t axis')
    plt.ylabel('Vals')
    plt.legend()
    plt.show()


def dofft0(samp, data):
    N = data.shape[0]
    T = 1.0 / samp
    x = np.linspace(0.0, N * T, N, endpoint=False)
    if len(data.shape) == 2:
        y = data[:, 0]
    else:
        y = data[:]

    yf = rfft(y)
    # xf = rfftfreq(len(x), T)[:N // 2]
    xf = rfftfreq(len(x), T)

    # plt.plot(xf, 2.0 / N * np.abs(yf[0:N // 2]))

    plt.plot(xf, np.abs(yf))
    plt.grid()
    plt.show()

    points_per_freq = len(xf) / (samp / 2)

    peak_coefficient = np.argmax(yf)
    peak_freq = abs(xf[peak_coefficient] * samp)

    print(peak_freq)

    threshold = 100
    # yf[target_idx - 1:target_idx + 2] = 0

    # plt.plot(xf, 2.0 / N * np.abs(yf[0:N // 2]))
    plt.plot(xf, np.abs(yf))
    plt.show()

    new_sig = irfft(yf)

    plt.plot(new_sig)
    plt.show()

    return new_sig


def dofft(samp, data):
    N = data.shape[0]
    T = 1.0 / 800.0
    length = data.shape[0] / samp
    x = np.linspace(0., length, data.shape[0], endpoint=False)
    y = data[:]
    yf = fft(y)
    w = blackman(N)
    ywf = fft(y * w)
    xf = fftfreq(N, T)[:N // 2]

    plt.semilogy(xf[1:N // 2], 2.0 / N * np.abs(yf[1:N // 2]), '-b')
    plt.semilogy(xf[1:N // 2], 2.0 / N * np.abs(ywf[1:N // 2]), '-r')
    plt.legend(['FFT', 'FFT w. window'])
    plt.grid()
    plt.show()


def dofft2(samp, data):
    samples = data.shape[0]
    datafft = fft(data)
    #Get the absolute value of real and complex component:
    fftabs = abs(datafft)
    freqs = fftfreq(samples, 1 / samp)
    plt.xlim([10, samp / 2])
    plt.xscale('log')
    plt.grid(True)
    plt.xlabel('Frequency (Hz)')
    plt.plot(freqs[:int(freqs.size / 2)], fftabs[:int(freqs.size / 2)])
    plt.show()


def nextpow2(l: int):
    return len(bin(25).lstrip('0b'))


def corr(samp: int, data: np.array):
    length = data.shape[0]
    bins, counts = np.unique(data, return_counts=True)
    time = np.linspace(0., length, data.shape[0])

    max3 = {}
    for i in range(3):
        ind = np.argmax(counts)
        max3[bins[ind]] = counts[ind]
        if i == 0:
            mid = bins[ind]
        # print(bins[ind])
        counts = np.delete(counts, ind)
        bins = np.delete(bins, ind)

    # plt.bar(bins)
    # plt.show()
    maX, miN = max(max3.keys()), min(max3.keys())
    d = np.zeros(data.shape)
    for i in range(len(data)):
        if data[i] in max3:
            d[i] = data[i]
        elif data[i] > mid:
            d[i] = maX
        elif data[i] < mid:
            d[i] = miN

    # plt.plot(time,d)
    # plt.show()

    return d, (miN, mid, maX)


def step(samp: int, data: np.ndarray):
    plt.style.use('_mpl-gallery')

    length = data.shape[0]
    x = np.linspace(0., length, data.shape[0])
    y = data[:]

    plt.step(x, y)
    plt.show()


def sample(samp, data):
    length = data.shape[0]
    t = np.linspace(0., length, data.shape[0])
    x = data[:]
    NFFT = 2**nextpow2(length)

    # fig, (ax1, ax2) = plt.subplots(nrows=2)
    # ax1.plot(t, x)
    Pxx, freqs, bins, im = plt.specgram(x, noverlap=0, scale='dB')
    plt.show()


def spec(samp, data):
    f, t, Sxx = spectrogram(data, samp)

    plt.pcolormesh(t, f, Sxx, shading='nearest')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()


def peaks(data):
    peaks, _ = signal.find_peaks(data)
    results_half = signal.peak_widths(data, peaks, rel_height=0.5)
    results_full = signal.peak_widths(data, peaks, rel_height=1)

    plt.plot(data)
    plt.plot(peaks, data[peaks], "x")
    plt.hlines(*results_half[1:], color="C2")
    plt.hlines(*results_full[1:], color="C3")
    plt.show()


def process(data: np.ndarray, bars: List) -> List[Tuple[int, int]]:
    beeps = [(0, 1)]
    data = [1 if i != bars[1] else 0 for i in data]
    # ones = sum([1 for i in data if i == 1])	Used for testing if the ones sum to be correct
    # zeros = sum([1 for i in data if i == 0])	Used for similiar  testing with zeros

    curr = data[0]
    i = 1
    while i < len(data):
        if data[i] != curr:
            curr = data[i]
        else:
            j = i
            t = 1
            while j < len(data) and data[j] == curr:
                data.pop(j)
                t += 1
            beeps.append((curr, t))
            continue
        i += 1

    i = 0
    while i < len(beeps) - 1:
        if beeps[i][0] == beeps[i + 1][0]:
            beeps[i] = (beeps[i][0], beeps[i][1] + beeps.pop(i + 1)[1])
        i += 1

    if beeps[0][0] == 0:
        beeps.pop(0)
    # if beeps[-1][0] == 0:
    # 	beeps.pop()

    return beeps


def translate(samp: int, beeps: List[Tuple[int, int]], speed: int) -> string:
    threshold = 75

    divider0 = div0(samp, beeps, speed, threshold)
    divider1 = div1(samp, beeps, speed, threshold)

    main = ''
    for i in beeps:
        if i[0] == 0:
            for j in divider0:
                if j[0] <= i[1] <= j[1]:
                    main += j[2]
                    break
        else:
            for j in divider1:
                if j[0] <= i[1] <= j[1]:
                    main += j[2]
                    break

    return main


def div1(samp: int, beeps: List[Tuple[int, int]], speed: int,
         threshold: int) -> List[Tuple[float, float, str]]:
    z = list(map(lambda x: x[1], filter(lambda x: x[0] == 1, beeps)))
    z.extend([459, 463, 467, 471, 1449, 1460])

    n, bins, _ = plt.hist(z,
                          bins='auto')  # arguments are passed to np.histogram
    # print(list(bins),bins.shape)

    cats = []
    for i in range(len(n)):
        if n[i] != 0:
            cats.append((bins[i], bins[i + 1]))
    # print(cats)
    cats = sorted(cats, key=lambda x: np.mean(x))

    # plt.title("Histogram with 'auto' bins")
    # plt.show()

    if len(cats) == 2:
        cats[-1] = (*cats[-1], '-')
        cats[0] = (*cats[0], '.')

    elif len(cats) == 1:
        dot = (60 / (50 * speed)) * samp  # Short Mark or '.'
        dash = dot * 3  # Longer Mark or '-'

        if cats[0][0] - threshold <= dot <= cats[0][1] + threshold:
            cats[0] = (*cats[0], '.')
        elif cats[0][0] - threshold <= dash <= cats[0][1] + threshold:
            cats[0] = (*cats[0], '-')
        else:
            raise Exception(
                'Mark Invalid. Please Ensure that the Morse Code is Correct.')

    return cats


def div0(samp: int, beeps: List[Tuple[int, int]], speed: int,
         threshold: int) -> List[Tuple[float, float, str]]:
    z = list(map(lambda x: x[1], filter(lambda x: x[0] == 0, beeps)))
    z.extend([460, 463, 467, 471, 1420, 1429, 3341, 3349])

    n, bins, _ = plt.hist(z,
                          bins='auto')  # arguments are passed to np.histogram
    # print(list(bins),bins.shape)

    cats = []
    for i in range(len(n)):
        if n[i] != 0:
            cats.append((bins[i], bins[i + 1]))

    cats = sorted(cats, key=lambda x: np.mean(x))
    # print(cats)

    if len(cats) == 3:
        cats[-1] = (*cats[-1], '/')
        cats[1] = (*cats[1], ' ')
        cats[0] = (*cats[0], '')

    elif len(cats) == 2:
        cats[1] = (*cats[1], ' ')
        cats[0] = (*cats[0], '')

    elif len(cats) == 1:
        intra_char_gap = (
            60 / (50 * speed)
        ) * samp  # Gap Between each unit of a single morse code unit
        short_gap = intra_char_gap * 3  # Gap Between english characters

        if cats[0][0] - threshold <= intra_char_gap <= cats[0][1] + threshold:
            cats[0] = (*cats[0], '')
        elif cats[0][0] - threshold <= short_gap <= cats[0][1] + threshold:
            cats[0] = (*cats[0], ' ')
        else:
            raise Exception(
                'Dividing Gap Character Invalid. Please Ensure that the Morse Code is Correct.'
            )

    return cats


def decode(mstr: str):
    morse_map_chars = {
        '.-': 'A',
        '-...': 'B',
        '-.-.': 'C',
        '-..': 'D',
        '.': 'E',
        '..-.': 'F',
        '--.': 'G',
        '....': 'H',
        '..': 'I',
        '.---': 'J',
        '-.-': 'K',
        '.-..': 'L',
        '--': 'M',
        '-.': 'N',
        '---': 'O',
        '.--.': 'P',
        '--.-': 'Q',
        '.-.': 'R',
        '...': 'S',
        '-': 'T',
        '..-': 'U',
        '...-': 'V',
        '.--': 'W',
        '-..-': 'X',
        '-.--': 'Y',
        '--..': 'Z',
        '.----': '1',
        '..---': '2',
        '...--': '3',
        '....-': '4',
        '.....': '5',
        '-....': '6',
        '--...': '7',
        '---..': '8',
        '----.': '9',
        '-----': '0',
        '--..--': ',',
        '..--..': '?',
        '---...': ':',
        '.-...': '&',
        '.----.': "'",
        '.--.-.': '@',
        '-.--.-': ')',
        '-.--.': '(',
        '-...-': '=',
        '-.-.--': '!',
        '.-.-.-': '.',
        '-....-': '-',
        '.-.-.': '+',
        '.-..-.': '"',
        '-..-.': '/',
        '/': ' ',
    }
    mstr = mstr.replace('/', ' / ')
    mlist = mstr.split()

    msg = ''
    for i in mlist:
        msg += morse_map_chars[i]

    return msg


fname = ".\\tests\\test.wav"
test = io.wavfile.read(fname)

# print(test[1].shape,test[0])
# plot(*test)

# Tryouts for Various Funcs
# fit_beizer(*test)
# kms(*test)
# plot(*test)
# print(extract_peak_frequency(*test))
# denoise(*test)
# data = dofft0(*test)
# dofft(*test)
# dofft2(*test)
# plot(test[0],d)
# step(*test)
# sample(test[1],d)
# spec(*test)
# peaks(d)

# Working Setup for De-Noised Audios
d, bars = corr(*test)
print("Got 'd'")
beeps = process(d, bars)
print("Got 'beeps'")
mstr = translate(test[0], beeps, 20)
print("Got 'mstr'")
msg = decode(mstr)
print('Message Decoded:', msg)
