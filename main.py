import os
from smtplib import SMTPServerDisconnected
from typing import List, Tuple, Any
import matplotlib.pyplot as plt
import numpy as np
from scipy import io


class MorseWav2Text:
    """
    Class to decode the morse code from a wav file

    Parameters
    ----------
    fpath : str
        File Path for the wav file which is to b decoded.

    speed : int, optional
        Speed or Proficiency in Morse Code of the sender in wpm (words per minute).(default: 20)

    threshold : int, optional
        Threshold or the Tolerance that the binning process can accept.

    Functions
    ---------
    `main`: Return the string obtained by decoding the wav file morse code.

    `decode`: Returns the decoded string obtained if the morse code string is passed.

    Notes
    -----
    You will need to provide a wav file which is denoised to just leave the frequency corresponding to the Morse Code. Or you can use wav sample which has a very minimal noise in the signal.
    """

    def __init__(self,
                 fpath: str,
                 speed: int = 20,
                 threshold: int = 75) -> None:
        if os.path.exists(fpath):
            self.sample_rate, self.wav_data = io.wavfile.read(fpath)
        else:
            raise FileNotFoundError(f"The WAV File {fpath} doesn't exist.")

        self.speed = speed
        self.threshold = threshold

    def __correct__(
        self, wav_data: np.ndarray[Any, np.dtype]
    ) -> Tuple[np.ndarray[np.float64, Any], Tuple[np.float64, np.float64,
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

    def __process__(
        self, corr_data: np.ndarray,
        bars: Tuple[np.float64, np.float64,
                    np.float64]) -> List[Tuple[int, int]]:
        '''It returns an array of tuples that represent a beep or a gap in the morse code along with its length.
    
        Parameters
        ----------
        corr_data : numpy ndarray
            Data obtained from the __correct__ method. It is an array with binned data points of the wav file.

        bars : tuple of 3 numpy float64
            It contains the miN, mid and maX bins that the wav_data has been binned in.

        Returns
        -------
        beeps : List of Tuples of length 2
            Contains the continuously grouped dits, dashes and gaps with their respective lengths for easier analysis.
        '''
        beeps = [(0, 1)]

        data = [1 if i != bars[1] else 0 for i in corr_data]
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

    def __div1__(self, sample_rate: int, beeps: List[Tuple[int, int]],
                 speed: int, threshold: int) -> List[Tuple[float, float, str]]:
        '''It returns an array containing categories of beeps with there range i.e. minimum length and maximum length for grouping purpose and their morse code representation.
    
        Parameters
        ----------
        sample_rate : int
            Sample rate of the wav file.

        beeps : List of Tuples of length 2
            Contains the continuously grouped dits, dashes and gaps with their respective lengths.

        speed : int
            Speed or Proficiency in Morse Code of the sender in wpm (words per minute).(default: 20)

        threshold : int
            Threshold or the Tolerance that the binning process can accept.

        Returns
        -------
        beeps : List of Tuples of length 2
            It is an array of tuples with bins' minimum, maximum and their morse representation for types of 1. This is calculated by automatic binning available through numpy.
        '''

        lengths_of_1 = list(
            map(lambda x: x[1], filter(lambda x: x[0] == 1, beeps)))
        # lengths_of_1.extend([459, 463, 467, 471, 1449, 1460])

        n, bins, _ = plt.hist(
            lengths_of_1, bins='auto')  # arguments are passed to np.histogram
        # print(list(bins),bins.shape)

        types_of_1 = []
        for i in range(len(n)):
            if n[i] != 0:
                types_of_1.append((bins[i], bins[i + 1]))
        # print(cats)
        types_of_1 = sorted(types_of_1, key=lambda x: np.mean(x))

        # plt.title("Histogram with 'auto' bins")
        # plt.show()

        if len(types_of_1) == 2:
            types_of_1[-1] = (*types_of_1[-1], '-')
            types_of_1[0] = (*types_of_1[0], '.')

        elif len(types_of_1) == 1:
            dot = (60 / (50 * speed)) * sample_rate  # Short Mark or '.'
            dash = dot * 3  # Longer Mark or '-'

            if types_of_1[0][0] - threshold <= dot <= types_of_1[0][
                    1] + threshold:
                types_of_1[0] = (*types_of_1[0], '.')
            elif types_of_1[0][0] - threshold <= dash <= types_of_1[0][
                    1] + threshold:
                types_of_1[0] = (*types_of_1[0], '-')
            else:
                raise Exception(
                    'Mark Invalid. Please Ensure that the Morse Code is Correct.'
                )

        return types_of_1

    def __div0__(self, sample_rate: int, beeps: List[Tuple[int, int]],
                 speed: int, threshold: int) -> List[Tuple[float, float, str]]:
        '''It returns an array containing categories of intra-character, short and long gap with there range i.e. minimum length and maximum length for grouping purpose and their morse code representation.
    
        Parameters
        ----------
        sample_rate : int
            Sample rate of the wav file.

        beeps : List of Tuples of length 2
            Contains the continuously grouped dits, dashes and gaps with their respective lengths.

        speed : int
            Speed or Proficiency in Morse Code of the sender in wpm (words per minute).
        
        threshold : int
            Threshold or the Tolerance that the binning process can accept.

        Returns
        -------
        beeps : List of Tuples of length 2
            It is an array of tuples with bins' minimum, maximum and their morse representation for types of 0. This is calculated by automatic binning available through numpy.
        '''

        lengths_of_0 = list(
            map(lambda x: x[1], filter(lambda x: x[0] == 0, beeps)))
        # lengths_of_0.extend([460, 463, 467, 471, 1420, 1429, 3341, 3349])

        n, bins, _ = plt.hist(
            lengths_of_0, bins='auto')  # arguments are passed to np.histogram
        # print(list(bins),bins.shape)

        types_of_0 = []
        for i in range(len(n)):
            if n[i] != 0:
                types_of_0.append((bins[i], bins[i + 1]))

        types_of_0 = sorted(types_of_0, key=lambda x: np.mean(x))
        # print(cats)

        if len(types_of_0) == 3:
            types_of_0[-1] = (*types_of_0[-1], '/')
            types_of_0[1] = (*types_of_0[1], ' ')
            types_of_0[0] = (*types_of_0[0], '')

        elif len(types_of_0) == 2:
            types_of_0[1] = (*types_of_0[1], ' ')
            types_of_0[0] = (*types_of_0[0], '')

        elif len(types_of_0) == 1:
            intra_char_gap = (
                60 / (50 * speed)
            ) * sample_rate  # Gap Between each unit of a single morse code unit
            short_gap = intra_char_gap * 3  # Gap Between english characters

            if types_of_0[0][0] - threshold <= intra_char_gap <= types_of_0[0][
                    1] + threshold:
                types_of_0[0] = (*types_of_0[0], '')
            elif types_of_0[0][0] - threshold <= short_gap <= types_of_0[0][
                    1] + threshold:
                types_of_0[0] = (*types_of_0[0], ' ')
            else:
                raise Exception(
                    'Dividing Gap Character Invalid. Please Ensure that the Morse Code is Correct.'
                )

        return types_of_0

    def __translate__(self, sample_rate: int, beeps: List[Tuple[int, int]],
                      speed: int, threshold: int) -> str:
        ''''It returns a string containing the morse code obtained from the wav file.
    
        Parameters
        ----------
        sample_rate : int
            Sample rate of the wav file.

        beeps : List of Tuples of length 2
            Contains the continuously grouped dits, dashes and gaps with their respective lengths.

        speed : int
            Speed or Proficiency in Morse Code of the sender in wpm (words per minute).
        
        threshold : int
            Threshold or the Tolerance that the binning process can accept.

        Returns
        -------
        morse_str : str
            It is the morse code string of the wav data obtained from the wav file. It is produced by mapping the beeps to the bin string obtained from the __div0__ and __div1__ methods.
        '''

        divider0 = self.__div0__(sample_rate, beeps, speed, threshold)
        divider1 = self.__div1__(sample_rate, beeps, speed, threshold)

        morse_str = ''
        for i in beeps:
            if i[0] == 0:
                for j in divider0:
                    if j[0] <= i[1] <= j[1]:
                        morse_str += j[2]
                        break
            else:
                for j in divider1:
                    if j[0] <= i[1] <= j[1]:
                        morse_str += j[2]
                        break

        return morse_str

    def decode(self, mstr: str) -> str:
        '''It converts the morse code string into the message. It follows the English International Morse Code for Standard Conversion.
    
        Parameters
        ----------
        mstr : str
            It is the string containing the morse code.

        Returns
        -------
        msg : str
            It is the message obtained after decoding the morse code string.

        Note
        ----
        It doesn't support any sort of Morse Code Abbreviations.
        '''
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

    def main(self):
        '''It sequentially performs the required routines to produce the decoded string of the morse code available in the wav file.
        It then returns the string obtained.

        Returns
        -------
        mstr : str
            It is the decoded string format of the wav file containing morse code. The result returned is in Capital Letters.
        '''

        d, bars = self.__correct__(self.wav_data)
        # print("Got 'd'")
        beeps = self.__process__(d, bars)
        # print("Got 'beeps'")
        mstr = self.__translate__(self.sample_rate, beeps, self.speed,
                                  self.threshold)
        # print("Got 'mstr'")
        return self.decode(mstr)


if __name__ == "__main__":
    decoder = MorseWav2Text(".\\tests\\test.wav")
    print('Message Decoded:', decoder.main())