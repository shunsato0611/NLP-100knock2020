import argparse
from itertools import islice

parser = argparse.ArgumentParser(description='Print the first NUM lines of FILE to standard output.')
parser.add_argument('file', metavar='FILE', type=argparse.FileType('r'),
                    help='file to process')
parser.add_argument('-n', metavar='NUM', type=int, default=10,
                    help='number of lines to print (default: 10)')

args = parser.parse_args()

for line in islice(args.file, args.n):
    print(line, end='')
