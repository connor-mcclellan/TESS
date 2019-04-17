from data_class import Star
import argparse

parser = argparse.ArgumentParser(description='Turn a 2-minute cadence time series into .dat and .inf files, ready for PRESTO.')
parser.add_argument('filename', metavar='filename', type=str, help='filename of the time series data')
args = parser.parse_args()

fname = args.filename
star = Star(fname)

star.prepare()
star.export()




