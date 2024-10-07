# SlurmUI
Simple terminal UI for slurm. Credit to [Norman Müller](https://github.com/SirWyver/slurmui).

<div align="center"> 
  <img src="demo.png">
</div>


## Install
```shell
pip install git+https://github.com/ShenhanQian/slurmui.git
```

## Usage

```shell
slurmui
```

Due to the difference between slurm settings, the program can fail when parsing results from `sinfo`. Therefore, you need to specify the cluster name for special cases.

| Cluster | Argument | Comment |
|-|-|-|
| [TUM CVG](https://cvg.cit.tum.de/) | | Works with the default setting. |
| [TUM VCG](https://www.niessnerlab.org/) | `--cluster tum_vcg` | Format of GRES is different. |
| [LRZ AI](https://doku.lrz.de/lrz-ai-systems-11484278.html) | `--cluster lrz_ai` | Need to filter out irrelavant partitions. |

For other clusters, you can first try running `slurmui` without arguments. If failed, you will need to add if-else sentences for your special cases. 

## Control
| Shortcut | Description               |
|----------|---------------------------|
|d | delete job (Confirm with <<Enter>>) |
|l | display log|
|g | display gpus |
|r | refresh |
|s | sort by column |
|q / <Escape> | Abort|
|Arrow keys | cursor up/down |
|<Ctrl+C> | Quit |

## Troubleshooting
- ### `AttributeError: 'NoneType' object has no attribute 'groups'`

    Search in the repository for places where `sinfo` is called with the output format specifies like `-O 'Partition:25,NodeHost,Gres:80,GresUsed:80,StateCompact,FreeMem,CPUsState'`. The width for some items(s) may not be enough, causing a problem to splitting the string into columns.