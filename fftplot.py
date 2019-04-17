import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Plot an FFT file.')
parser.add_argument('filename', metavar='filename', type=str, help='filename of the FFT')
args = parser.parse_args()

filename = args.filename

c = np.fromfile(filename, 'F')
c_conj = c.conj()

print(c[0:4])
print(c_conj[0:4])

powers = (c*c_conj).astype(np.float32)

print(powers[0:4])

# Now use the evenly-spaced time bins to make the frequency bins

fsamp = 1./120. # Sampling rate of time series, in Hz
N = 2*len(powers) # Number of samples

fbin_resolution = fsamp/N # in units of Hz per bin
freqs = np.arange(N)*fbin_resolution # these are the frequencies
freqs = freqs[range(int(N/2))]

powers = powers[range(int(N/2))]

plt.figure()
#plt.scatter(freqs[1:], powers[1:], marker='o', s=1.5, color='k', alpha=0.5)
plt.plot(freqs[1:], powers[1:])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Normalized Power')
plt.title('FFT Plot for file '+filename)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.xlim(0, freqs[-1])

#plt.figure()
#plt.hist(powers[1:], bins=100)
#plt.title('Histogram of Powers for file ', filename)
#plt.xlabel('Power')
#plt.ylabel('n')

plt.show()



