#!/usr/bin/env python3
from os import system
from glob import glob
from sys import argv

E = 0
if len(argv) < 2:
	print('You must specify an Energy.')
	quit()
else:
	E = int(argv[-1])

base = open('HistoBase.gnu', 'r').read()
command = base.replace('??', str(E)).replace('\n', '; ').replace('"', '\\"') + 'plot '
for i, filename in enumerate(sort(glob('mass_E{}_*.txt'.format(E)))):
	if i > 0:
		command += ', '
	command += '\\"' + filename + '\\" title \\"Distribuzione ' + filename.split('_')[-1][:-5] + '\\" with boxes';

system('gnuplot -e "' + command + '"')