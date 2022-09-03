from itertools import count
import statistics as stat
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import io, interpolate
from scipy.interpolate import interp1d
from scipy.cluster.vq import vq, whiten, kmeans
from scipy.fft import fft, fftfreq
from scipy.signal import blackman, spectrogram
from scipy import signal
from collections import Counter

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
	plt.plot(freqs[:int(freqs.size/2)],fftabs[:int(freqs.size/2)])
	plt.show()

def nextpow2(l:int):
	return len(bin(25).lstrip('0b'))

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
	
	return d, (miN,mid,maX)
	


def step(samp:int, data: np.ndarray):
	plt.style.use('_mpl-gallery')
	
	length = data.shape[0]
	x = np.linspace(0., length, data.shape[0])
	y = data[:]

	plt.step(x,y)
	plt.show()

def sample(samp, data):
	length = data.shape[0]
	t = np.linspace(0., length, data.shape[0])
	x = data[:]
	NFFT = 2**nextpow2(length)

	# fig, (ax1, ax2) = plt.subplots(nrows=2)
	# ax1.plot(t, x)
	Pxx, freqs, bins, im = plt.specgram(x,noverlap=0,scale='dB')
	plt.show()

def spec(samp,data):
	f, t, Sxx = spectrogram(data,samp)

	plt.pcolormesh(t, f, Sxx,shading='nearest')
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

def process(data,bars):
	beeps = [(0,1)]
	data = [1 if i != bars[1] else 0 for i in data]
	ones = sum([1 for i in data if i == 1])
	zeros = sum([1 for i in data if i == 0])
	
	curr = data[0]
	i = 1
	while i < len(data):
		if data[i] != curr:
			curr = data[i]				
		else:
			j = i
			t = 1
			while j < len (data) and data[j] == curr:
				data.pop(j)
				t += 1
			beeps.append((curr,t))
			continue
		i += 1	
	
	i = 0
	while i < len(beeps) - 1:
		if beeps[i][0] == beeps[i+1][0]:
			beeps[i] = (beeps[i][0],beeps[i][1] + beeps.pop(i+1)[1])
		i += 1
	
	if beeps[0][0] == 0:
		beeps.pop(0)
	# if beeps[-1][0] == 0:
	# 	beeps.pop()
	
	return beeps

def translate(beeps,speed):
	
	div0(beeps)

def div1(beeps):
	z = list(map(lambda x: x[1],filter(lambda x: x[0] == 1,beeps)))
	z.extend([459,463,467,471,1449,1460])

	n, bins, _  = plt.hist(z, bins='auto')  # arguments are passed to np.histogram
	# print(list(bins),bins.shape)

	cats = []
	for i in range(len(n)):
		if n[i] != 0:
			cats.append((bins[i],bins[i+1],(bins[i]+bins[i+1])/2))
	# print(cats)
	cats = sorted(cats,key=lambda x: np.mean(x))
	
	# plt.title("Histogram with 'auto' bins")
	# plt.show()

	if len(cats) == 2:
		cats[-1] = (cats[-1],'-')
		cats[0] = (cats[0],'.')
	
	elif len(cats) == 1:
		# todo
		pass

	return list(map(lambda x: (*x[0][:2],x[1]),cats))


def div0(beeps):
	div_types = []
	z = list(map(lambda x: x[1],filter(lambda x: x[0] == 0,beeps)))
	z.extend([460,463,467,471,1420,1429,3341,3349])

	n, bins, _  = plt.hist(z, bins='auto')  # arguments are passed to np.histogram
	# print(list(bins),bins.shape)

	cats = []
	for i in range(len(n)):
		if n[i] != 0:
			cats.append((bins[i],bins[i+1]))
	
	cats = sorted(cats,key=lambda x: np.mean(x))
	print(cats)

	if len(cats) == 3:
		cats[-1] = (cats[-1],'/')
		cats[1] = (cats[1],' ')
		cats[0] = (cats[0],'')
	# print(cats)
	
	# plt.title("Histogram with 'auto' bins")
	# plt.show()

	# hist, _ = np.histogram(z,bins='auto')
	# print(hist)


fname = ".\\tests\\hello world.wav"
test = io.wavfile.read(fname)

# print(test[1].shape,test[0])

# plot(*test)
# fit_beizer(*test)
# kms(*test)
# dofft(*test)
# dofft2(*test,fr)
d, bars = corr(*test)
print("Got 'd'")
beeps = process(d,bars)
print("Got 'beeps'")
translate(beeps,20)
# plot(test[0],d)
# step(*test)
# sample(test[1],d)
# spec(*test)
# peaks(d)