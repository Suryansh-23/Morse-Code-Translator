import os
from typing import List, Tuple, Any
import matplotlib.pyplot as plt
import numpy as np
from scipy import io


class MorseWav2Text:

    def __init__(self, fname: str) -> None:
        if os.path.exists(fname):
            self.sample_rate, self.wav_data = io.wavfile.read(fname)
        else:
            raise FileNotFoundError(f"The WAV File {fname} doesn't exist.")

    def __correct__(
        self, wav_data: np.ndarray[Any, np.dtype]
    ) -> Tuple[np.ndarray[np.float64], Tuple[np.float64, np.float64,
                                             np.float64]]:
        '''It removes any irregularities present in the wav_data and flattens it out with binning them into the highest, lowest and mean amplitudes of the wave plot.
    
        Parameters
        ----------
        wav_data : numpy ndarray
            Data read from a WAV File is used. It could be of any data-type. Also, one could pass a denoised data-point array of the wav file.


        Returns
        -------
        d : numpy ndarray
            Corrected and binned data-points array with data-points having only the maximum, minimum and the mean amplitude.
        (miN, mid, maX) : tuple of float64
            These are the minimum, mean and maximum amplitudes available from the amplitude vs time plot of the wav file.
        '''
        length = wav_data.shape[0]

        bins, counts = np.unique(wav_data, return_counts=True)

        max3 = {}
        for i in range(3):
            ind = np.argmax(counts)
            max3[bins[ind]] = counts[ind]
            if i == 0:
                mid = bins[ind]
            # print(bins[ind])
            counts = np.delete(counts, ind)
            bins = np.delete(bins, ind)

        maX, miN = max(max3.keys()), min(max3.keys())
        d = np.zeros(wav_data.shape)
        for i in range(len(wav_data)):
            if wav_data[i] in max3:
                d[i] = wav_data[i]
            elif wav_data[i] > mid:
                d[i] = maX
            elif wav_data[i] < mid:
                d[i] = miN

        return d, (miN, mid, maX)

    def process(
        wav_data: np.ndarray,
        bars: Tuple[np.float64, np.float64,
                    np.float64]) -> List[Tuple[int, int]]:
        beeps = [(0, 1)]

        data = [1 if i != bars[1] else 0 for i in wav_data]
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


if __name__ == "__main__":
    MorseWav2Text(".\\tests\\tested.wav")