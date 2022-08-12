import argparse as ap
import os
import matplotlib as mpl
from xy_rotor import *


ps = ap.ArgumentParser(description="Plots averaged properties of the system (in equilibrium). It needs to be under a folder with the name <ddmmyy> of today: --loc/<ddmmyy>/destfilepath")
ps.add_argument("destfilepath", help="Destination file path after ddmmyy/")
ps.add_argument("--calcfilepath", nargs='+', help="Absolute Path to the raw data corr to this")
ps.add_argument("--ykey", nargs='+', help="Key of the plot on the y-axis (<temp>, <en>, <magn>, <staggmagn>, <heatcap>, <numconn>, <accrat_spins>, <accrat_conns>)")
ps.add_argument("--title", default="<No title>", help="Title to give the plot")
ps.add_argument("--loc", default=os.getcwd(), help="Filepath before ddmmyy")
ps.add_argument("--xkey", default="temp", help="Var to plot on the x-axis")
ps.add_argument("--errbars", default="y", help="To include error bars")
ps.add_argument("--plotimg", default="y", help="Plot to .png file")
ps.add_argument("--plotaxfile", default="n", help="Write ax to file")

args = ps.parse_args()

for i in range(len(args.calcfilepath)):
    for key in args.ykey:
        plotprop(getrawdata(args.calcfilepath[i]), key, title='['+str(i)+'] '+args.title, savedir=args.destfilepath, loc=args.loc, errbars=boolpars(args.errbars),writeplot=[boolpars(args.plotimg),boolpars(args.plotaxfile)])
