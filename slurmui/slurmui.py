import io
import importlib.metadata
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Header, Footer, Static, RichLog, DataTable
from textual.containers import Container, Horizontal
from rich.text import Text
import subprocess
import pandas as pd
import re
import os
import threading
from functools import wraps
from datetime import datetime


DEBUG = False
if DEBUG:
    from slurmui.debug_strings import SINFO_DEBUG, SQUEUE_DEBUG


def run_in_thread(func):
    """Decorator to run a function in a separate thread"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=func, args=args, kwargs=kwargs)
        thread.start()
    return wrapper

def handle_error(func):
    """Decorator to wrap action methods with try-except logic"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            self.info_log.write(str(e))
    return wrapper


class SlurmUI(App):
    cluster = None
    interval = 10
    STAGE = {
        "action": "job",
        "job": {
            "sort_column": 0,
            "sort_ascending": True
        },
        "node": {},
    }
    stats = {}
    selected_jobid = []
    show_all_nodes = False

    theme = "textual-dark"
    selected_text_style = "bold on orange3"
    border_type = "solid"
    border_color = "white"

    CSS_PATH = "slurmui.tcss"
    TITLE = f"SlurmUI (v{importlib.metadata.version('slurmui')})"

    BINDINGS = [
        Binding("q", "abort_quit", "Quit"),
        Binding("space", "select", "Select"),
        Binding("v", "select_inverse", "Inverse"),
        Binding("r", "refresh", "Refresh"),
        Binding("enter", "confirm", "Confirm", priority=True, key_display='enter'),
        Binding("s", "sort", "Sort"),
        Binding("d", "delete", "Delete"),
        Binding("G", "print_gpustat", "GPU"),
        Binding("g", "display_node_list", "Nodes"),
        Binding("l", "display_job_log", "Log"),
    ]

    def compose(self) -> ComposeResult:
        self.header = Header()
        self.footer = Footer()

        self.njobs = Static("Jobs: ")
        self.ngpus = Static("GPUs:")
        self.timestamp = Static(":::")
        self.status_bar = Horizontal(self.njobs, self.ngpus, self.timestamp, classes="status_bar")

        self.node_table = DataTable(id="node_table")
        # self.node_table.zebra_stripes = True
        self.job_table = DataTable(id="job_table")
        # self.job_table.zebra_stripes = True
        self.active_table = self.job_table

        self.info_log = RichLog(wrap=True, highlight=True, id="info_log", auto_scroll=True)
        self.info_log.can_focus = False
        self.info_log.border_title = "Info"
        
        self.job_log = RichLog(wrap=True, highlight=True, id="job_log", auto_scroll=False)
        self.job_log_position = None
        
        yield self.header
        yield self.status_bar
        yield Container(self.job_table, self.node_table, self.info_log, self.job_log)
        yield self.footer

    def on_mount(self):
        self.init_job_table()
        self.init_node_table()
        self.switch_display("job")

    def on_ready(self) -> None:
        if self.interval > 0:
            self.set_interval(self.interval, self.auto_refresh)
    
    def check_action(self, action: str, parameters):  
        """Check if an action may run."""
        # self.info_log.write(f"Action: {action}")

        if action == "abort_quit":
            pass
        elif action == "select" and self.STAGE['action'] not in ['job', 'select']:
            return False
        elif action == "select_inverse" and self.STAGE['action'] not in ['job', 'select']:
            return False
        elif action == "refresh" and self.STAGE['action'] not in ['job', 'node', 'log']:
            return False
        elif action == "confirm" and self.STAGE['action'] != 'delete':
            return False
        elif action == "sort" and self.STAGE['action'] not in ['job', 'node']:
            return False
        elif action == "delete" and self.STAGE['action'] not in ['job', 'select']:
            return False
        elif action == "print_gpustat" and self.STAGE['action'] != 'job':
            return False
        elif action == "display_node_list" and self.STAGE['action'] not in ['job', 'node']:
            return False
        elif action == "display_job_log" and self.STAGE['action'] != 'job':
            return False
        return True

    @handle_error
    def action_abort_quit(self):
        if self.STAGE["action"] == "job":
            self.exit(0)
        else:
            self.action_abort()

    @handle_error
    def action_abort(self):
        if self.STAGE["action"] == "delete":
            self.info_log.write("Delete: aborted")
            self.selected_jobid = []
        elif self.STAGE["action"] == "log":
            self.job_log_position = None
        elif self.STAGE["action"] == "select":
            self.info_log.write("Select: none")
            self.selected_jobid = []
        self.STAGE['action'] = "job"
        self.update_job_table()
        self.switch_display("job")
        self.refresh_bindings()

    @handle_error
    def action_select(self):
        if (self.STAGE["action"] == "job" and not self.selected_jobid) or self.STAGE["action"] == "select":
            i = self.active_table.cursor_coordinate[0]
            value = str(self.active_table.get_cell_at((i, 0)))

            job_id, _ = self._get_selected_job()
            if job_id in self.selected_jobid:
                self.selected_jobid.remove(job_id)
                self.active_table.update_cell_at((i, 0), value)
            else:
                self.selected_jobid.append(job_id)
                self.active_table.update_cell_at((i, 0), Text(str(value), style=self.selected_text_style))
            
            if self.selected_jobid:
                self.STAGE["action"] = "select"
                self.info_log.write(f"Select: {' '.join(self.selected_jobid)}")
            else:
                self.STAGE["action"] = "job"
                self.info_log.write(f"Select: none")
            self.active_table.action_cursor_down()
        self.refresh_bindings()  
   
    @handle_error
    def action_select_inverse(self):
        assert self.STAGE["action"] in ["job", "select"]
        for i in range(len(self.active_table.rows)):
            job_id = str(self.active_table.get_cell_at((i, 0)))
            
            if job_id in self.selected_jobid:
                self.selected_jobid.remove(job_id)
                self.active_table.update_cell_at((i, 0), job_id)
            else:
                self.selected_jobid.append(job_id)
                self.active_table.update_cell_at((i, 0), Text(str(job_id), style=self.selected_text_style))
                self.active_table.move_cursor(row=i, column=0)
        if self.selected_jobid:
            self.STAGE["action"] = "select"
            self.info_log.write(f"Select: {' '.join(self.selected_jobid)}")
        else:
            self.STAGE["action"] = "job"
            self.info_log.write(f"Select: none")
        self.refresh_bindings()  
    
    @handle_error
    def auto_refresh(self):
        self.action_refresh()

    @run_in_thread
    @handle_error
    def action_refresh(self):
        if self.STAGE["action"] == "job":
            self.update_job_table()
        elif self.STAGE["action"] == "log":
            self.update_log(self.STAGE["job_id"])
        elif self.STAGE["action"] == "node":
            self.update_node_table()
        self.update_status()
    
    @handle_error
    def action_confirm(self):
        # job to delete
        if self.STAGE["action"] == "delete":
            perform_scancel(self.STAGE['job_id'])
            self.info_log.write(f"Delete: {self.STAGE['job_id']}? succeeded")
            self.selected_jobid = []
            self.update_job_table()
            self.STAGE["action"] = "job"
        self.refresh_bindings()

    @handle_error
    def action_sort(self):
        sort_column = self.active_table.cursor_column
        if sort_column != self.STAGE[self.STAGE["action"]].get("sort_column"):
            self.STAGE[self.STAGE["action"]]["sort_ascending"] = False
        else:
            self.STAGE[self.STAGE["action"]]["sort_ascending"] = not self.STAGE[self.STAGE["action"]].get("sort_ascending", True)
        self.STAGE[self.STAGE["action"]]['sort_column'] = sort_column
        if self.STAGE["action"] == "job":
            self.update_job_table()
            self.switch_display("job")
        elif self.STAGE["action"] == "node":
            self.update_node_table()
            self.switch_display("node")
        self.active_table.move_cursor(row=0, column=sort_column)
    
    @handle_error
    def action_delete(self):
        if self.STAGE["action"] == "job":
            job_id, job_name = self._get_selected_job()
            self.info_log.write(f"Delete: {job_id}? press <<ENTER>> to confirm")
            self.STAGE.update({"action": "delete", "job_id": job_id, "job_name": job_name})
        elif self.STAGE["action"] == "select":
            self.info_log.write(f"Delete: {' '.join(self.selected_jobid)}? press <<ENTER>> to confirm")
            self.STAGE.update({"action": "delete", "job_id": ' '.join(self.selected_jobid)})
        self.refresh_bindings()  

    @handle_error
    def action_print_gpustat(self):
        if self.STAGE["action"] == "job":
            job_id, _ = self._get_selected_job()
            gpustat = subprocess.check_output(f"""srun --jobid {job_id} gpustat""", shell=True, timeout=3).decode("utf-8").rstrip()
            self.info_log.write(gpustat)

    @handle_error
    def action_display_node_list(self):
        if self.STAGE["action"] == "job":
            self.STAGE.update({"action": "node"})
            self.update_node_table()
            self.switch_display("node")
            self.refresh_bindings()
        elif self.STAGE["action"] == "node":
            self.show_all_nodes = not self.show_all_nodes
            self.update_node_table()
    
    @handle_error
    def action_display_job_log(self):
        if self.STAGE["action"] == "job":
            job_id, job_name = self._get_selected_job()
            self.STAGE.update({"action": "log", "job_id": job_id, "job_name": job_name})
            self.switch_display("job_log")
            self.update_log(job_id)
        self.refresh_bindings()  

    def switch_display(self, action):
        if action == "job":
            self.job_table.styles.height = "80%"
            self.active_table = self.job_table
            self.active_table.focus()

            self.node_table.styles.height = "0%"

            self.info_log.styles.height="20%"
            self.info_log.styles.border = (self.border_type, self.border_color)

            self.job_log.styles.height="0%"
            self.job_log.styles.border = ("none", self.border_color)
            self.job_log.clear()
        elif action == "node":
            self.job_table.styles.height = "0%"

            self.node_table.styles.height = "100%"
            self.active_table = self.node_table
            self.active_table.focus()
        elif action == "job_log":
            self.job_table.styles.height="0%"
        
            self.info_log.styles.height="0%"
            self.info_log.styles.border = ("none", self.border_color)
            
            self.job_log.styles.height="100%"
            self.job_log.styles.border = (self.border_type, self.border_color)
            self.job_log.focus()
        else:
            raise ValueError(f"Invalid action: {action}")

    @run_in_thread
    def update_status(self):
        ngpus_avail = self.stats.get("ngpus_avail", 0)
        ngpus = self.stats.get("ngpus", 0)
        njobs = self.stats.get("njobs", 0)
        njobs_running = self.stats.get("njobs_running", 0)

        self.njobs.update(f"Jobs: {njobs_running}/{njobs}")
        self.ngpus.update(f"GPUs: {ngpus_avail}/{ngpus}")
        self.timestamp.update(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def query_squeue(self, sort_column=None, sort_ascending=True):
        squeue_df = get_squeue() 
        if sort_column is not None:
            squeue_df = squeue_df.sort_values(squeue_df.columns[sort_column], ascending=sort_ascending)

        self.stats['njobs'] = len(squeue_df)
        self.stats['njobs_running'] = sum(1 for row in squeue_df.iterrows() if row[1]['STATE'] == 'RUNNING')
        return squeue_df

    @run_in_thread
    def init_job_table(self, sort_column=None, sort_ascending=True):
        if 'sort_column' in self.STAGE[self.STAGE["action"]]:
            sort_column = self.STAGE[self.STAGE["action"]]['sort_column']
        if 'sort_ascending' in self.STAGE[self.STAGE["action"]]:
            sort_ascending = self.STAGE[self.STAGE["action"]]['sort_ascending']

        squeue_df = self.query_squeue(sort_column=sort_column, sort_ascending=sort_ascending)
        self.update_status()

        self.job_table.clear()
        self.job_table.add_columns(*squeue_df.columns)
        for _, row in squeue_df.iterrows():
            table_row = [str(row[col]) for col in squeue_df.columns]
            self.job_table.add_row(*table_row)

    @run_in_thread
    def update_job_table(self, sort_column=None, sort_ascending=True):
        if 'sort_column' in self.STAGE[self.STAGE["action"]]:
            sort_column = self.STAGE[self.STAGE["action"]]['sort_column']
        if 'sort_ascending' in self.STAGE[self.STAGE["action"]]:
            sort_ascending = self.STAGE[self.STAGE["action"]]['sort_ascending']
            
        squeue_df = self.query_squeue(sort_column=sort_column, sort_ascending=sort_ascending)
        self.update_status()
        
        # If the table is empty, initialize it
        if not self.job_table.columns:
            self.job_table.add_columns(*squeue_df.columns)
            for _, row in squeue_df.iterrows():
                table_row = [str(row[col]) for col in squeue_df.columns]
                self.job_table.add_row(*table_row)
            return
        
        # Update only the cells that have changed
        for row_index, (_, row) in enumerate(squeue_df.iterrows()):
            table_row = [str(row[col]) for col in squeue_df.columns]
            if row_index < len(self.job_table.rows):
                for col_index, cell in enumerate(table_row):

                    if self.job_table.get_cell_at((row_index, col_index)) != cell:
                        self.job_table.update_cell_at((row_index, col_index), cell)
            else:
                self.job_table.add_row(*table_row)
        
        # Remove any extra rows
        while len(self.job_table.rows) > len(squeue_df):
            row_key, _ = self.job_table.coordinate_to_cell_key((len(self.job_table.rows) - 1, 0))
            self.job_table.remove_row(row_key)

    def _get_selected_job(self):
        row_idx = self.active_table.cursor_row
        row = self.active_table.get_row_at(row_idx)
        job_id = str(row[0])
        job_name = str(row[2])
        return job_id, job_name

    @handle_error
    def update_log(self, job_id):
        log_fn = get_log_fn(job_id)
        self.job_log.border_title = f"{log_fn}"
        self.job_log.border_subtitle = f"{log_fn}"
        current_scroll_y = self.job_log.scroll_offset[1]

        if not self.job_log_position:
            with open(log_fn, 'r') as f:
                self.job_log_position = max(sum(len(line) for line in f) - 2**16, 0)  # read the last 64KB
            
            with open(log_fn, 'r') as log_file:
                log_file.seek(self.job_log_position)
                new_lines = log_file.readlines()[1:]  # drop the first line because it can be incomplete
                self.job_log_position = log_file.tell()
        else:
            with open(log_fn, 'r') as log_file:
                log_file.seek(self.job_log_position)
                new_lines = log_file.readlines()
                self.job_log_position = log_file.tell()

        update_scroll = current_scroll_y == self.job_log.max_scroll_y
        
        for line in new_lines:
            self.job_log.write(line)

        if update_scroll:
            self.job_log.scroll_end(animate=False)

    @run_in_thread
    def init_node_table(self, sort_column=None, sort_ascending=True):
        if 'sort_column' in self.STAGE[self.STAGE["action"]]:
            sort_column = self.STAGE[self.STAGE["action"]]['sort_column']
        if 'sort_ascending' in self.STAGE[self.STAGE["action"]]:
            sort_ascending = self.STAGE[self.STAGE["action"]]['sort_ascending']

        overview_df = self.query_gpus(sort_column=sort_column, sort_ascending=sort_ascending)
        self.update_status()

        self.node_table.clear()
        self.node_table.add_columns(*overview_df.columns)
        for _, row in overview_df.iterrows():
            table_row = [str(row[col]) for col in overview_df.columns]
            self.node_table.add_row(*table_row)
    
    @run_in_thread
    def update_node_table(self, sort_column=None, sort_ascending=True):
        if 'sort_column' in self.STAGE[self.STAGE["action"]]:
            sort_column = self.STAGE[self.STAGE["action"]]['sort_column']
        if 'sort_ascending' in self.STAGE[self.STAGE["action"]]:
            sort_ascending = self.STAGE[self.STAGE["action"]]['sort_ascending']

        overview_df = self.query_gpus(sort_column=sort_column, sort_ascending=sort_ascending)
        
        # If the table is empty, initialize it
        if not self.node_table.columns:
            self.node_table.add_columns(*overview_df.columns)
            for _, row in overview_df.iterrows():
                table_row = [str(row[col]) for col in overview_df.columns]
                self.node_table.add_row(*table_row)
            return
        
        # Update only the cells that have changed
        for row_index, (_, row) in enumerate(overview_df.iterrows()):
            table_row = [str(row[col]) for col in overview_df.columns]
            if row_index < len(self.node_table.rows):
                for col_index, cell in enumerate(table_row):
                    if self.node_table.get_cell_at((row_index, col_index)) != cell:
                        self.node_table.update_cell_at((row_index, col_index), cell)
            else:
                self.node_table.add_row(*table_row)
        
        # Remove any extra rows
        while len(self.node_table.rows) > len(overview_df):
            row_key, _ = self.node_table.coordinate_to_cell_key((len(self.node_table.rows) - 1, 0))
            self.node_table.remove_row(row_key)
        
        self.update_status()

    def query_gpus(self,  sort_column=None, sort_ascending=True):
        overview_df = get_sinfo(self.cluster)
        if sort_column is not None:
            overview_df = overview_df.sort_values(overview_df.columns[sort_column],ascending=sort_ascending)
        
        self.stats['ngpus'] = overview_df["#Total"].sum()
        self.stats['ngpus_avail'] = overview_df["#Avail"].sum()

        if not self.show_all_nodes:
            # filter out nodes with no available GPUs
            overview_df = overview_df[overview_df["#Avail"] > 0]
        return overview_df

def perform_scancel(job_id):
    os.system(f"""scancel {job_id}""")

def parse_gres_used(gres_used_str, num_total, cluster=None):
    try:
        device = ""
        alloc_str = "N/A"
        if cluster == "lrz_ai":
            try:
                _, device, num_gpus, alloc_str = re.match("(.*):(.*):(.*)\\(IDX:(.*)\\).*", gres_used_str).groups()
            except:
                _, num_gpus = re.match("(.*):(.*)", gres_used_str).groups()
        elif cluster == "tum_vcg":
            _, device, num_gpus, alloc_str = re.match("(.*):(.*):(.*)\\(IDX:(.*)\\),.*", gres_used_str).groups()
        else:
            _, device, num_gpus, alloc_str = re.match("(.*):(.*):(.*)\\(IDX:(.*)\\).*", gres_used_str).groups()
        
        num_gpus = int(num_gpus)
    except Exception as e:
        print(e)
        raise ValueError(f"Error parsing gres_used: \n\t{gres_used_str}\nCheck if the string matches the expected format")

    alloc_gpus = []
    for gpu_ids in alloc_str.split(","):
        if "-" in gpu_ids:
            start, end = gpu_ids.split("-")
            for i in range(int(start), int(end)+1):
                alloc_gpus.append(i)
        else:
            if gpu_ids == "N/A":
                pass
            else:
                alloc_gpus.append(int(gpu_ids))
            
    return {"Device": device,
            "#Alloc": num_gpus,
            "Free IDX": [idx for idx in range(num_total) if idx not in alloc_gpus]}

def parse_gres(gres_str, cluster=None):
    try:
        device = ""
        if cluster == "tum_vcg":
            _, device, num_gpus = re.match("(.*):(.*):(.*),.*", gres_str).groups()
        elif cluster == "lrz_ai":
            try:
                _, num_gpus, _ = re.match("(.*):(.*)\\(S:(.*)\\)", gres_str).groups()
            except:
                _, num_gpus = re.match("(.*):(.*)", gres_str).groups()
        else:
            _, num_gpus, _ = re.match("(.*):(.*)\\(S:(.*)\\)", gres_str).groups()

        num_gpus = int(num_gpus)
    except Exception as e:
        print(e)
        raise ValueError(f"Error parsing gres: \n\t{gres_str}\nCheck if the string matches the expected format")
    
    return {"Device": device,
            "#Total": num_gpus}

def remove_first_line(input_string):
    lines = input_string.split('\n')
    return '\n'.join(lines[1:])

def get_sinfo(cluster):
    if DEBUG:
        response_string = SINFO_DEBUG
    else:
        if cluster == 'lrz_ai':
            response_string = ""
            partitions = [
                # "",  # only keep this line if you want to see all partitions
                "-p 'mcml-dgx-a100-40x8'",
                "-p 'mcml-hgx-a100-80x4'",
                "-p 'mcml-hgx-h100-92x4'",
                "-p 'lrz-dgx-a100-80x8'", 
                "-p 'lrz-hgx-a100-80x4'", 
                "-p 'lrz-hgx-h100-92x4'",
                # "-p 'lrz-dgx-1-v100x8'", 
                # "-p 'lrz-dgx-1-p100x8'", 
                # "-p 'lrz-hpe-p100x4'", 
                # "-p 'lrz-v100x2'",
            ]
            for p in partitions:
                s = subprocess.check_output(f"""sinfo --Node {p} -O 'Partition:25,NodeHost,Gres:80,GresUsed:80,StateCompact,FreeMem,CPUsState'""", shell=True).decode("utf-8")  # WARNING: insufficient width for any item can crash the prgram
                if len(response_string) > 0:
                    response_string += remove_first_line(s)
                else:
                    response_string += s
        else:
            response_string = subprocess.check_output(f"""sinfo -O 'Partition:25,NodeHost,Gres:500,GresUsed:500,StateCompact,FreeMem,CPUsState'""", shell=True).decode("utf-8")

    formatted_string = re.sub(' +', ' ', response_string)
    data = io.StringIO(formatted_string)
    df = pd.read_csv(data, sep=" ")
    overview_df = [ ]# pd.DataFrame(columns=['Host', "Device", "#Avail", "#Total", "Free IDX"])
    for row in df.iterrows():
        node_available = row[1]["STATE"] in ["mix", "idle", "alloc"]

        # if "mcml" not in row[1]["HOSTNAMES"]:
        #     continue

        if row[1]['GRES'] != "(null)":
            host_info = parse_gres(row[1]['GRES'], cluster)
        else:
            continue

        host_avail_info = parse_gres_used(row[1]['GRES_USED'], host_info["#Total"], cluster)
        host_info.update(host_avail_info)
        if not node_available:
            host_info["#Avail"] = 0
            host_info["Free IDX"] = []
        else:
            host_info["#Avail"] = host_info['#Total'] - host_info["#Alloc"]
        try:
            host_info['Mem (GB)'] = int(row[1]["FREE_MEM"]) // 1024
        except:
            host_info['Mem (GB)'] = row[1]["FREE_MEM"]

        cpu_info = row[1]["CPUS(A/I/O/T)"].split("/")
        host_info['#CPUs Idle'] = cpu_info[1]
        host_info['#CPUs Alloc'] = cpu_info[0]
        host_info['Host'] = str(row[1]["HOSTNAMES"])
        host_info['Partition'] = str(row[1]["PARTITION"])
        host_info['State'] = str(row[1]["STATE"])

        overview_df.append(host_info)
    overview_df = pd.DataFrame.from_records(overview_df).drop_duplicates("Host")
    overview_df = overview_df[['Partition', 'Host', "Device", "State", "#Avail", "#Total", "Free IDX", "Mem (GB)", "#CPUs Idle", "#CPUs Alloc"]]
    return overview_df

def get_squeue():
    if DEBUG:
        response_string = SQUEUE_DEBUG
    else:
        sep = "|"
        response_string = subprocess.check_output(f"""squeue --format="%.18i{sep}%Q{sep}%.20P{sep}%.100j{sep}%.10u{sep}%.8T{sep}%.10M{sep}%.6l{sep}%.S{sep}%.4D{sep}%R" --me -S T""", shell=True).decode("utf-8")
    formatted_string = re.sub(' +', ' ', response_string)
    data = io.StringIO(formatted_string)
    df = pd.read_csv(data, sep=sep)

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Strip whitespace from each string element in the DataFrame
    for col in df.select_dtypes(['object']).columns:
        df[col] = df[col].str.strip()
    
    # Add allocated GPU IDs
    df["GPU_IDS"] = "N/A"
    mask = (df["PARTITION"] != "in") & (df["STATE"] == "RUNNING")
    if mask.any():
        df.loc[mask, "GPU_IDS"] = df.loc[mask, "JOBID"].apply(lambda x: get_job_gpu_ids(x))
    return df 

def get_job_gpu_ids(job_id):
    try:
        response_string = subprocess.check_output(f"""scontrol show jobid -dd {job_id} | grep GRES""", shell=True).decode("utf-8")
        formatted_string = response_string.split(":")[-1].strip()[:-1]
    except:
        return "N/A"
    return formatted_string

def get_log_fn(job_id):
    response_string = subprocess.check_output(f"""scontrol show job {job_id} | grep StdOut""", shell=True).decode("utf-8")
    formatted_string = response_string.split("=")[-1].strip()
    return formatted_string

def read_log(fn, num_lines=100):
    with open(os.path.expanduser(fn), 'r') as f:
        txt_lines = list(f.readlines()[-num_lines:])
    
    return txt_lines

def run_ui(debug=False, cluster=None, interval=10):
    if debug:
        # global for quick debugging
        global DEBUG
        DEBUG = True
    app = SlurmUI()
    app.cluster = cluster
    app.interval = interval
    app.run()


if __name__ == "__main__":
    run_ui()
