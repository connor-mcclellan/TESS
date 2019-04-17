import pickle
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Plot a pickled matplotlib figure.')
parser.add_argument('filename', metavar='filename', type=str, help='filename of the pickle')
args = parser.parse_args()

filename = args.filename

p = pickle.load(open(filename, 'rb'))
plt.show()
