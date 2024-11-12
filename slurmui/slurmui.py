import io
import importlib.metadata
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import DataTable
from textual.widgets import Header, Footer, RichLog
from textual.containers import Container
import subprocess
import pandas as pd
import re
import os
import threading
from functools import wraps
from datetime import datetime
import pyperclip


DEBUG = False
if DEBUG:
    from slurmui.debug_strings import SINFO_DEBUG, SQUEUE_DEBUG


def run_in_thread(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=func, args=args, kwargs=kwargs)
        thread.start()
    return wrapper


class SlurmUI(App):
    cluster = None
    STAGE = {
        "action": "monitor",
        "monitor": {
            "sort_column": 0,
            "sort_ascending": True
        },
        "gpu": {},
    }
    stats = {}

    BINDINGS = [
        Binding("g", "display_gpu", "GPU"),
        Binding("l", "display_log", "Log"),
        Binding("d", "stage_delete", "Delete job"),
        Binding("r", "refresh", "Refresh"),
        Binding("s", "sort", "Sort"),
        Binding("i", "copy_jobid", "Copy JobID"),
        Binding("q", "abort_quit", "Quit"),
        Binding("enter", "confirm", "Confirm", priority=True),
        Binding("escape", "abort_quit", "Abort"),
    ]

    def action_copy_jobid(self):
        job_id, _ = self._get_selected_job()
        
        pyperclip.copy(job_id)  # Copies text to clipboard
        clipboard_text = pyperclip.paste()
        if clipboard_text == job_id:
            self.txt_log.write(f"Copied {clipboard_text} to clipboard")
        else:
            self.txt_log.write(f"Error copying JOBID to clipboard.")

    def compose(self) -> ComposeResult:
        self.header = Header()
        self.footer = Footer()
        self.gpu_table = DataTable(id="gpu_table")
        self.squeue_table = DataTable(id="squeue_table")
        self.active_table = self.squeue_table
        self.gpu_table.zebra_stripes = True
        self.squeue_table.zebra_stripes = True
        self.txt_log = RichLog(wrap=True, highlight=True, id="info", auto_scroll=False)
        self.log_position = None
        yield self.header
        yield Container(self.gpu_table, self.squeue_table, self.txt_log)
        yield self.footer

        if self.interval > 0:
            self.set_interval(self.interval, self.auto_refresh)

    def on_mount(self):
        self._minimize_output_panel()

    def on_ready(self) -> None:
        self.init_gpu_table()
        self.switch_table_display("monitor")
        self.init_squeue_table()

    def auto_refresh(self):
        self.action_refresh()
    
    @run_in_thread
    def action_refresh(self):
        if self.STAGE["action"] == "monitor":
            try:
                self.update_squeue_table()
            except Exception as e:
                self.txt_log.write(str(e))
        elif self.STAGE["action"] == "log":
            self.update_log(self.STAGE["job_id"])
        elif self.STAGE["action"] == "gpu":
            try:
                self.update_gpu_table()
            except Exception as e:
                raise ValueError(str(e))
        self.update_title()
    
    def update_title(self):
        ngpus_avail = self.stats.get("ngpus_avail", 0)
        ngpus = self.stats.get("ngpus", 0)
        njobs = self.stats.get("njobs", 0)
        njobs_running = self.stats.get("njobs_running", 0)

        self.title = f"SlurmUI (v{importlib.metadata.version('slurmui')}) \t[Jobs: {njobs_running}/{njobs} | GPUs: {ngpus_avail}/{ngpus}]"

    def query_squeue(self, sort_column=None, sort_ascending=True):
        squeue_df = get_squeue() 
        if sort_column is not None:
            squeue_df = squeue_df.sort_values(squeue_df.columns[sort_column], ascending=sort_ascending)

        self.stats['njobs'] = len(squeue_df)
        self.stats['njobs_running'] = sum(1 for row in squeue_df.iterrows() if row[1]['STATE'] == 'RUNNING')
        return squeue_df

    @run_in_thread
    def init_squeue_table(self, sort_column=None, sort_ascending=True):
        if 'sort_column' in self.STAGE[self.STAGE["action"]]:
            sort_column = self.STAGE[self.STAGE["action"]]['sort_column']
        if 'sort_ascending' in self.STAGE[self.STAGE["action"]]:
            sort_ascending = self.STAGE[self.STAGE["action"]]['sort_ascending']

        squeue_df = self.query_squeue(sort_column=sort_column, sort_ascending=sort_ascending)

        self.squeue_table.clear()
        self.squeue_table.add_columns(*squeue_df.columns)
        for _, row in squeue_df.iterrows():
            table_row = [str(row[col]) for col in squeue_df.columns]
            self.squeue_table.add_row(*table_row)
        
        self.update_title()
        self.squeue_table.focus()

    @run_in_thread
    def update_squeue_table(self, sort_column=None, sort_ascending=True):
        if 'sort_column' in self.STAGE[self.STAGE["action"]]:
            sort_column = self.STAGE[self.STAGE["action"]]['sort_column']
        if 'sort_ascending' in self.STAGE[self.STAGE["action"]]:
            sort_ascending = self.STAGE[self.STAGE["action"]]['sort_ascending']
            
        squeue_df = self.query_squeue(sort_column=sort_column, sort_ascending=sort_ascending)
        
        # If the table is empty, initialize it
        if not self.squeue_table.columns:
            self.squeue_table.add_columns(*squeue_df.columns)
            for _, row in squeue_df.iterrows():
                table_row = [str(row[col]) for col in squeue_df.columns]
                self.squeue_table.add_row(*table_row)
            return
        
        # Update only the cells that have changed
        for row_index, (_, row) in enumerate(squeue_df.iterrows()):
            table_row = [str(row[col]) for col in squeue_df.columns]
            if row_index < len(self.squeue_table.rows):
                for col_index, cell in enumerate(table_row):

                    if self.squeue_table.get_cell_at((row_index, col_index)) != cell:
                        self.squeue_table.update_cell_at((row_index, col_index), cell)
            else:
                self.squeue_table.add_row(*table_row)
        
        # Remove any extra rows
        while len(self.squeue_table.rows) > len(squeue_df):
            row_key, _ = self.squeue_table.coordinate_to_cell_key((len(self.squeue_table.rows) - 1, 0))
            self.squeue_table.remove_row(row_key)
        
        self.update_title()
        self.squeue_table.focus()

    def _get_selected_job(self):
        row_idx = self.active_table.cursor_row
        row = self.active_table.get_row_at(row_idx)
        job_id = row[0]
        job_name = row[2]
        return job_id, job_name

    def action_stage_delete(self):
        if self.STAGE['action'] == "monitor":
            try:
                job_id, job_name = self._get_selected_job()
                self.txt_log.clear()
                self.txt_log.write(f"Delete: {job_id} - {job_name}? Press <<ENTER>> to confirm")
                self.STAGE.update({"action": "delete", "job_id": job_id, "job_name": job_name})
            except Exception as e:
                self.txt_log.clear()
                self.txt_log.write(str(e))

    def action_display_log(self):
        try:
            if self.STAGE["action"] == "monitor":
                job_id, job_name = self._get_selected_job()
                self.STAGE.update({"action": "log", "job_id": job_id, "job_name": job_name})
                self._maximize_output_panel()
                self.update_log(job_id)
        except Exception as e:
            self.txt_log.clear()
            self.txt_log.write(str(e))

    def update_log(self, job_id):
        try:
            log_fn = get_log_fn(job_id)
            current_scroll_y = self.txt_log.scroll_offset[1]

            if not self.log_position:
                with open(log_fn, 'r') as f:
                    self.log_position = max(sum(len(line) for line in f) - 2**16, 0)  # read the last 64KB
                
                with open(log_fn, 'r') as log_file:
                    log_file.seek(self.log_position)
                    new_lines = log_file.readlines()[1:]  # drop the first line because it can be incomplete
                    self.log_position = log_file.tell()
            else:
                with open(log_fn, 'r') as log_file:
                    log_file.seek(self.log_position)
                    new_lines = log_file.readlines()
                    self.log_position = log_file.tell()
        except Exception as e:
            current_scroll_y = self.txt_log.max_scroll_y
            new_lines = [f"[{datetime.now()}] {str(e)}"]

        update_scroll = current_scroll_y == self.txt_log.max_scroll_y
        
        for line in new_lines:
            self.txt_log.write(line)

        if update_scroll:
            self.txt_log.scroll_end(animate=False)

    def _minimize_output_panel(self):
        self.log_position = None
        self.squeue_table.styles.height="80%"
        self.txt_log.styles.max_height="20%"
        self.txt_log.can_focus = False
        self.txt_log.styles.border = ("heavy","grey")

    def _maximize_output_panel(self):
        self.squeue_table.styles.height="0%"
        self.txt_log.styles.max_height="100%"
        self.txt_log.can_focus = True
        self.txt_log.styles.border = ("heavy","white")

    def action_display_gpu(self):
        self.STAGE.update({"action": "gpu"})
        try:
            self.update_gpu_table()
            self.switch_table_display("gpu")
        except Exception as e:
            self.txt_log.clear()
            self.txt_log.write(str(e))

    @run_in_thread
    def init_gpu_table(self, sort_column=None, sort_ascending=True):
        if 'sort_column' in self.STAGE[self.STAGE["action"]]:
            sort_column = self.STAGE[self.STAGE["action"]]['sort_column']
        if 'sort_ascending' in self.STAGE[self.STAGE["action"]]:
            sort_ascending = self.STAGE[self.STAGE["action"]]['sort_ascending']

        overview_df = self.query_gpus(sort_column=sort_column, sort_ascending=sort_ascending)

        self.gpu_table.clear()
        self.gpu_table.add_columns(*overview_df.columns)
        for _, row in overview_df.iterrows():
            table_row = [str(row[col]) for col in overview_df.columns]
            self.gpu_table.add_row(*table_row)

        self.gpu_table.focus()
        self.update_title()
    
    @run_in_thread
    def update_gpu_table(self, sort_column=None, sort_ascending=True):
        if 'sort_column' in self.STAGE[self.STAGE["action"]]:
            sort_column = self.STAGE[self.STAGE["action"]]['sort_column']
        if 'sort_ascending' in self.STAGE[self.STAGE["action"]]:
            sort_ascending = self.STAGE[self.STAGE["action"]]['sort_ascending']

        overview_df = self.query_gpus(sort_column=sort_column, sort_ascending=sort_ascending)
        
        # If the table is empty, initialize it
        if not self.gpu_table.columns:
            self.gpu_table.add_columns(*overview_df.columns)
            for _, row in overview_df.iterrows():
                table_row = [str(row[col]) for col in overview_df.columns]
                self.gpu_table.add_row(*table_row)
            return
        
        # Update only the cells that have changed
        for row_index, (_, row) in enumerate(overview_df.iterrows()):
            table_row = [str(row[col]) for col in overview_df.columns]
            if row_index < len(self.gpu_table.rows):
                for col_index, cell in enumerate(table_row):
                    if self.gpu_table.get_cell_at((row_index, col_index)) != cell:
                        self.gpu_table.update_cell_at((row_index, col_index), cell)
            else:
                self.gpu_table.add_row(*table_row)
        
        # Remove any extra rows
        while len(self.gpu_table.rows) > len(overview_df):
            self.gpu_table.remove_row(len(self.gpu_table.rows) - 1)
        
        self.update_title()
        self.gpu_table.focus()

    def query_gpus(self,  sort_column=None, sort_ascending=True):
        overview_df = get_sinfo(self.cluster)
        if sort_column is not None:
            overview_df = overview_df.sort_values(overview_df.columns[sort_column],ascending=sort_ascending)
        
        self.stats['ngpus'] = overview_df["#Total"].sum()
        self.stats['ngpus_avail'] = overview_df["#Avail"].sum()
        return overview_df

    def switch_table_display(self, action):
        if action == "gpu":
            self.gpu_table.styles.height = "100%"
            self.squeue_table.styles.height = "0%"
            self.active_table = self.gpu_table
        elif action == "monitor":
            self.gpu_table.styles.height = "0%"
            self.squeue_table.styles.height = "80%"
            self.active_table = self.squeue_table
        else:
            raise ValueError(f"Invalid action: {action}")

    def action_sort(self):
        sort_column = self.active_table.cursor_column
        if sort_column != self.STAGE[self.STAGE["action"]].get("sort_column"):
            self.STAGE[self.STAGE["action"]]["sort_ascending"] = False
        else:
            self.STAGE[self.STAGE["action"]]["sort_ascending"] = not self.STAGE[self.STAGE["action"]].get("sort_ascending", True)
        self.STAGE[self.STAGE["action"]]['sort_column'] = sort_column
        if self.STAGE["action"] == "monitor":
            self.update_squeue_table()
            self.switch_table_display("monitor")
        elif self.STAGE["action"] == "gpu":
            self.update_gpu_table()
            self.switch_table_display("gpu")
        self.active_table.cursor_coordinate = (0, sort_column)
        
    def action_confirm(self):
        if self.STAGE["action"] == "monitor":
            pass
        else:
            self.txt_log.clear()
            # job to delete
            if self.STAGE["action"] == "delete":
                perform_scancel(self.STAGE['job_id'])
                self.txt_log.write(f"{self.STAGE['job_id']} - {self.STAGE['job_name']} deleted")
                self.update_squeue_table()
                self.STAGE["action"] = "monitor"

    def action_abort(self):
        if self.STAGE["action"] == "log":
            self._minimize_output_panel()

        self.STAGE['action'] = "monitor"
        self.update_squeue_table()
        self.switch_table_display("monitor")
        self.txt_log.clear()
        
    def action_abort_quit(self):
        if self.STAGE["action"] == "monitor":
            self.exit(0)
        else:
            self.action_abort()

def perform_scancel(job_id):
    os.system(f"""scancel {job_id}""")

def parse_gres_used(gres_used_str, num_total, cluster=None):
    device = ""
    alloc_str = "N/A"
    try:
        if cluster == "lrz_ai":
            try:
                try:
                    _, device, num_gpus, alloc_str = re.match("(.*):(.*):(.*)\\(IDX:(.*)\\).*", gres_used_str).groups()
                except:
                    _, num_gpus = re.match("(.*):(.*)", gres_used_str).groups()
            except:
                raise ValueError(f"DEBUG: {gres_used_str}")
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
    device = ""
    try:
        if cluster == "tum_vcg":
            _, device, num_gpus = re.match("(.*):(.*):(.*),.*", gres_str).groups()
        elif cluster == "lrz_ai":
            try:
                _, num_gpus, _ = re.match("(.*):(.*)\\(S:(.*)\\)", gres_str).groups()
            except:
                try:
                    _, num_gpus = re.match("(.*):(.*)", gres_str).groups()
                except:
                    raise ValueError(f"DEBUG: {gres_str}")
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
                "-p 'lrz-hgx-h100-92x4'"
                "-p 'lrz-dgx-1-v100x8'", 
                "-p 'lrz-dgx-1-p100x8'", 
                "-p 'lrz-hpe-p100x4'", 
                "-p 'lrz-v100x2'",
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
        df["GPU_IDS"][mask] = df[mask]["JOBID"].apply(lambda x: get_job_gpu_ids(x))
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
