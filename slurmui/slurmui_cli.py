from slurmui import run_ui
from argparse import ArgumentParser

def slurmui_cli():
    # adding arguments later
    parser = ArgumentParser("SLURM UI")
    parser.add_argument("-v", "--verbose", action='store_true')
    parser.add_argument("-c", "--cluster", help="Specify the name of the cluster")
    parser.add_argument("-i", "--interval", help="Specify the interval in seconds to refresh the UI", type=int, default=5)
    parser.add_argument("-r", "--history-range", help="Specify the time range of history jobs to load", type=str, default="1 week")
    args = parser.parse_args()
    run_ui(verbose=args.verbose, cluster=args.cluster, interval=args.interval)

