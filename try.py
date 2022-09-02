from itertools import count
import statistics as stat
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import io, interpolate
from scipy.interpolate import interp1d
from scipy.cluster.vq import vq, whiten, kmeans
from scipy.fft import fft, fftfreq
from scipy.signal import blackman
from collections import Counter

def plot(samp, data):
	length = data.shape[0] 
	time = np.linspace(0., length, data.shape[0])
	plt.scatter(time, data[:], label="Channel")
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
	plt.plot(x, y, 'x',x,ynew)
	plt.legend(['Linear', 'Cubic Spline'], loc="best")
	# plt.axis([-0.05, 6.33, -1.05, 1.05])
	plt.title('Cubic-spline interpolation')
	plt.show()


def kms(samp,data):
	length = data.shape[0] / samp
	x = np.linspace(0., length, data.shape[0])
	y = data[:]
	whitened = whiten([x,data])
	codebook, distortion = kmeans(whitened,2)

	plt.scatter(whitened[:,0], whitened[:,1])
	plt.scatter(codebook[:,0], codebook[:,1],c='r')
	plt.show()

def dofft(samp,data):
	N = data.shape[0]
	T = 1.0 / 800.0
	length = data.shape[0] / samp
	x = np.linspace(0., length, data.shape[0],endpoint=False)
	y = data[:]
	yf = fft(y)
	w = blackman(N)
	ywf = fft(y*w)
	xf = fftfreq(N,T)[:N//2]
	
	plt.semilogy(xf[1:N//2], 2.0/N * np.abs(yf[1:N//2]), '-b')
	plt.semilogy(xf[1:N//2], 2.0/N * np.abs(ywf[1:N//2]), '-r')
	plt.legend(['FFT', 'FFT w. window'])
	plt.grid()
	plt.show()

def dofft2(samp,data):
	samples = data.shape[0]
	datafft = fft(data)
	#Get the absolute value of real and complex component:
	fftabs = abs(datafft)
	freqs = fftfreq(samples,1/samp)
	plt.xlim( [10, samp/2] )
	plt.xscale( 'log' )
	plt.grid( True )
	plt.xlabel( 'Frequency (Hz)' )
	# plt.plot(freqs[:int(freqs.size/2)],fftabs[:int(freqs.size/2)])
	# plt.show()

	x, y = freqs[:int(freqs.size/2)],fftabs[:int(freqs.size/2)]

	print(y)

def corr(samp:int ,data: np.ndarray):
	length = data.shape[0] 
	bins,counts = np.unique(data,return_counts=True)
	time = np.linspace(0., length, data.shape[0])

	max3 = {}
	for i in range(3):
		ind = np.argmax(counts)
		max3[bins[ind]] = counts[ind]
		if i == 0:
			mid = bins[ind]
		# print(bins[ind])
		counts = np.delete(counts,ind)
		bins = np.delete(bins,ind)

	# plt.bar(bins)
	# plt.show()
	maX, miN = max(max3.keys()), min(max3.keys())
	d = list(map(lambda x: x if x in max3 else 0, data))
	for i in range(len(data)):
		if data[i] in max3:
			d[i] = data[i]
		elif data[i] > mid:
			d[i] = maX
		elif data[i] < mid:
			d[i] = miN

	plt.plot(time,d)
	plt.show()
	
	# print(tmp.shape)


def step(samp:int, data: np.ndarray):
	plt.style.use('_mpl-gallery')
	
	length = data.shape[0]
	x = np.linspace(0., length, data.shape[0])
	y = data[:]

	plt.step(x,y)
	plt.show()

def sample(samp, data):
	length = data.shape[0] / samp
	t = np.linspace(0., length, data.shape[0])
	x = data[:]
	NFFT = 1024
	Fs = samp

	fig, (ax1, ax2) = plt.subplots(nrows=2)
	ax1.plot(t, x)
	ax2.specgram(x, Fs=Fs)
	plt.show()

test = io.wavfile.read(".\\tests\\a.wav")

# print(test[1].shape,test[0])

# plot(*test)
# fit_beizer(*test)
# kms(*test)
# dofft(*test)
dofft2(*test)
# corr(*test)
# step(*test)
# sample(*test)