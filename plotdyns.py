import argparse as ap
from xy_rotor import *

ps = ap.ArgumentParser(description="Plots dynamics of the system. From: locget/sharedfilepath/calcfilepath. To: locsave/<ddmmyy>/sharedfilepath/destfilename.")
ps.add_argument("calcfilepath", help="Absolute path to the raw data")
ps.add_argument("destdir", help="Destination direc")
ps.add_argument("--ykey", nargs="+", help="Key or keys of the plot on the y-axis (<temp>, <en>, <magn>, <staggmagn>, <heatcap>, <numconn>, <accrat_spins>, <accrat_conns>)")
ps.add_argument("--title", default="<No title>", help="Title to give the plot")
ps.add_argument("--locsave", default=os.getcwd(), help="Filepath before ddmmyy to save")
ps.add_argument("--sharedfilepath", default=[""], nargs="*", help="Destination file path between ddmmyy and destfname")
ps.add_argument("--locget", default="", help="Filepath before ddmmyy to read")
ps.add_argument("--plotimg", default="y", help="Plot to .png file")
ps.add_argument("--plotaxfile", default="n", help="Write ax to file")
ps.add_argument("--fourtransfed", default="n", help="Fourier-transforms the output")

ps.add_argument("--minsweep", type=int, default=0, help="Amount of sweeps to start at")
ps.add_argument("--maxsweep", type=int, default=None, help="Amount of sweeps to stop at")

def fixslash(arg):
    if arg == '' or arg[-1] == "/":
        return arg
    return arg+'/'

args = ps.parse_args()
for i in range(len(args.sharedfilepath)):
    for key in args.ykey:
        plotdynamics(getrawdata(fixslash(args.locget)+fixslash(args.sharedfilepath[i])+args.calcfilepath), key, title='['+str(i)+'] '+args.title, savedir=fixslash(args.sharedfilepath[i])+args.destdir, loc=args.locsave, writeplot=[boolpars(args.plotimg),boolpars(args.plotaxfile)], sweeplims=[args.minsweep,args.maxsweep], ftrans=boolpars(args.fourtransfed))


