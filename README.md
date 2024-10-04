# SlurmUI
Simple terminal UI for slurm

## Usage
```
pip install slurmui
```
To start, use type `slurmui`

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