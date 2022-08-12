# IT CANNOT FUSE MORE THAN 10 DIFFERENT FILES!


import matplotlib.pyplot as plt
import os
import argparse as ap
from xy_rotor import *


ps = ap.ArgumentParser(description="Gathers Lines from specified files and plots them together")
ps.add_argument("fnames", nargs="+", help="Absolute path to filenames")
ps.add_argument("dest", help="Where to put out plot")
ps.add_argument("title", help="Title")
ps.add_argument("--fnamedir", default="", help="Path to filenames (could also be included under fnames)")
ps.add_argument("--loc", default=os.getcwd(), help="Path to ddmmyy")


args = ps.parse_args()
print(args.dest)

if args.fnamedir != "":
    args.fnamedir += "/"

figtog, axtog = plt.subplots()
colors=['r','g','b','c']

axes = [getrawdata(args.fnamedir+fname) for fname in args.fnames]
tits = [ax.get_title() for ax in axes]

try:
    tshuf = [[tits[i][j] for i in range(len(tits))] for j in range(len(tits[0]))]
except:
    tshuf = "<Name error>"




titout = []
for ch in tshuf:
    if ch.count(ch[0]) == len(ch):
        titout.append(ch[0])
titout=''.join(titout)

q=[]

for i in range(len(axes)):
    x_list = []
    lower_list = []
    upper_list = []
    for l in axes[i].lines:
        print(l)
        l.remove()
        l.axes = axtog
        l.set_transform(axtog.transData) 
        l.set_color(colors[i])


        print(i)
        q+=list(np.array([l.get_paths() for l in axes[i].collections]).flatten())

        print(q)


    axtog.add_line(l)

handles, labels = axtog.get_legend_handles_labels()
axtog.legend(handles[::-1], labels[::-1])
axtog.autoscale()


## Room to set labels, title, etc.


axtog.set_xlabel(axes[0].get_xlabel())
axtog.set_ylabel(axes[0].get_ylabel())
axtog.set_title(str(titout))


plotout(figtog, args.title+" - "+list(LABS.keys())[list(LABS.values()).index(axes[0].get_ylabel())]+" - "+titout, args.dest, args.loc)
