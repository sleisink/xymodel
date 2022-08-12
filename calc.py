import argparse as ap
import numpy as np
import os
from xy_rotor import *

ps = ap.ArgumentParser(description="Runs a simulation, computes data and stores it under --path/<ddmmyy>/dirc/. <ddmmyy> needs to be todays date.")
ps.add_argument("dimension", type=int, help="Amount of nodes in the graph")
ps.add_argument("tstart", type=float, help="Lowest temperature")
ps.add_argument("tstop", type=float, help="Highest temperature")
ps.add_argument("tamount", type=int, help="Amount of temperatures (evenly distributed)")
ps.add_argument("sweeps", type=int, help="Amount of sweeps. When started from equilibrium, the same amount is executed without measurements to get the system in equilibrium")
ps.add_argument("dirc", help="Final path to store data")

ps.add_argument("--storedynamics", default="y", help="Should dynamics be stored? (y/n)")
ps.add_argument("--storenondyms", default="y", help="Should non-dynamic data be stored? (y/n)")
ps.add_argument("--startfromeq", default="y", help="Should it start from equilibrium? (y/n)")
ps.add_argument("--fixedconn", default="n", help="Fixed lattice for all temperatures? (y/n)")

ps.add_argument("--snaps", default=[], type=int, nargs="*", help="A number of snapshots measured")

ps.add_argument("--path", default=os.getcwd(), help="Path to store data")

ps.add_argument("--initconnstruc", default=["rand", 0], nargs="+", help="Structure of the initial graph. Supported options: <rand>. If <rand> extra argument: Fraction of all nodes each node is connected to. ONLY VALS 0,1 WORK!")

distrtext = "It is either <constant [val]> or <uniform [width]> or <uniform> (where width=pi is implied)"
ps.add_argument("--j_distr", default=["constant", 1], nargs="+", help="Distribution of J_ij. "+distrtext)
ps.add_argument("--alpha_distr", default=["constant", 0], nargs="+", help="Distribution of alpha_ij. "+distrtext)
ps.add_argument("--spininit_distr", default=["uniform", 2*np.pi], nargs="+", help="Distribution of sigma_i. "+distrtext)

ps.add_argument("--fname", default="", help="Result folder name")
ps.add_argument("--maxconn", default=["global", np.infty], nargs="*", help="global/local, Maximum number of connections a node may have")
ps.add_argument("--conn", default="nonfixed", help="<nonfixed> does not put restrictions on network density, <fixed> preserves overall number of connections, <fixed2> preserves degree of each node (initialising a regular graph).")

args = ps.parse_args()


assert direxists(args.path, args.dirc)





def matrdistr(arg, dim,symm=False):
    if arg[0] == "constant":

        return [float(arg[1])*np.ones((args.dimension,)*dim),0]
    elif arg[0] == "uniform":
        if len(arg) == 1:
            out = [np.random.rand(*((args.dimension,)*dim))*2*np.pi, 2*np.pi]

        else:
            out = [np.random.rand(*((args.dimension,)*dim))*float(arg[1]), float(arg[1])]

        if symm:
            out[0] = np.tril(out[0]) + np.tril(out[0], -1).T
        return out
    return None


def seconds():
    now = dt.datetime.now()
    return 3600*now.hour+60*now.minute+now.second+now.microsecond/1e6


args.maxconn[1]=float(args.maxconn[1]) if args.maxconn[1] != '-1' else np.infty


if args.initconnstruc[0] == "rand":
    initconnstruc = rand(args.dimension, float(args.initconnstruc[1]),maxnum=float(args.maxconn[1]))
    #add more options later?

alph = matrdistr(args.alpha_distr,2, symm=True)
spin = matrdistr(args.spininit_distr, 1)[0]




then = seconds()
metadata = "Calculation started on "+dt.datetime.today().strftime('%d%m%y')+" at "+dt.datetime.now().strftime("%H:%M")

fname = args.fname+"_"+str(i)
c, snaps, dyn = runcalculation(neigh=initconnstruc, j=matrdistr(args.j_distr, 2, symm=True)[0], ts=np.arange(args.tstart, args.tstop, (args.tstop-args.tstart)/args.tamount), fixedconn=boolpars(args.fixedconn), sweeps=args.sweeps, savenums=args.snaps, savedynamics=boolpars(args.storedynamics), equilibrium=boolpars(args.startfromeq), alpha=deepcopy(alph)[0], alphwidth=deepcopy(alph)[1], spin=deepcopy(spin), maxconn=args.maxconn, conn=args.conn)

#pos=None may be added as optfion later


h = lambda x,tab=2: "\n"+x+":"+"\t"*tab
metadata+=". It took "+str(round(seconds()-then,4))+" seconds.\n\n#####Settings#####\n##################"+h("dimension")+str(args.dimension)+h("tstart",3)+str(args.tstart)+h("tstop",3)+str(args.tstop)+h("tamount")+str(args.tamount)+h("sweeps",3)+str(args.sweeps)+h("dirc",3)+args.dirc+"\n"+h("--storedynamics",1)+args.storedynamics+h("--storenondyms")+args.storenondyms+h("--startfromeq")+args.startfromeq+h("--fixedconn")+args.fixedconn+"\n"+h("--snaps")+str(args.snaps)+h("--path",3)+args.path+h("--initconnstruc",1)+str(args.initconnstruc)+"\n"+h("--j_distr")+str(args.j_distr)+h("--alpha_distr")+str(args.alpha_distr)+h("--spininit_distr",1)+str(args.spininit_distr)+h("--fname")+str(args.fname)+h("--maxconn")+str(args.maxconn)+h("--conn")+str(args.conn)    metadata+="\n\n\n#######Help#######\n##################\n\n"+ps.format_help()

time=dt.datetime.now().strftime("[%H:%M]")+"_"+fname
if fname == "":
time = dt.datetime.now().strftime("[%H:%M]")
else: time = fname

os.mkdir(args.path+"/"+dirtoday()+args.dirc+"/"+time)
with open(args.path+"/"+dirtoday()+args.dirc+"/"+time+"/metadata.txt", "wt") as f:
print(metadata, file=f)


if boolpars(args.storenondyms):
saverawdata(c,args.dirc,args.path, time+"/nondynams_raw.txt")
if boolpars(args.storedynamics):
saverawdata(dyn,args.dirc,args.path, time+"/dynams_raw.txt")
if args.snaps != []:
saverawdata(snaps,args.dirc,args.path, time+"/snaps_raw.txt")
