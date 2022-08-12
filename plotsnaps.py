import argparse as ap
import os
from xy_rotor import *


ps = ap.ArgumentParser(description="Plots images of the network at certain moments in time")
ps.add_argument("calcfilepath", help="Absolute path to the raw data")
ps.add_argument("destfilepath", help="Destination file path after ddmmyy/")
ps.add_argument("--title", default="<No title>", help="Title to give the plot")
ps.add_argument("--loc", default=os.getcwd(), help="Filepath before ddmmyy")
ps.add_argument("--nodekey", default=["spins"], nargs="+", help="What to plot on the nodes. <spins>, <frustr>. Or: <no>")
ps.add_argument("--edgekey", default=[None], nargs="+", help="What to plot on the edges. <frustr>")
ps.add_argument("--fixedscaling", default=None, type=int, help="How to scale the colorbar. Automatic in general but for comparison you want it fixed. Unscaled: scaled such that the most frustrated spin has 1.")
ps.add_argument("--addgraphs", default=[], nargs="*", help="Some graphs to may add for a plot")
ps.add_argument("--pos", default="default", help="<default> or <frustr>")

args = ps.parse_args()
if args.nodekey[0]=="no":
    plotgraph(getrawdata(args.calcfilepath), title=args.title, savedir=args.destfilepath,loc=args.loc,addgraphs=args.addgraphs[0])

else:
    for nkey, edgkey in zip(args.nodekey, args.edgekey):
        plotsnaps(getrawdata(args.calcfilepath), title=args.title, savedir=args.destfilepath,loc=args.loc, nodekey=nkey, edgekey=edgkey, addgraphs=args.addgraphs, pos=args.pos)
