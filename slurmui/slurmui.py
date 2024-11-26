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
from datetime import datetime, timedelta


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
    # configuration that can be set from the command line
    cluster = None
    interval = 10
    history_range = "1 week"  # Default history range

    # internal states
    STAGE = {
        "action": "job",
        "job": {
            "sort_column": 0,
            "sort_ascending": True,
        },
    }
    stats = {}
    selected_jobid = []
    show_all_nodes = False
    show_all_history = False

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
        Binding("L", "open_with_less", "Open with less"),
        Binding("h", "display_history_jobs", "History"),
        Binding("H", "toggle_history_range", "Range"),
    ]

    def compose(self) -> ComposeResult:
        self.header = Header()
        self.footer = Footer()

        self.njobs = Static("Jobs: ")
        self.ngpus = Static("GPUs:")
        self.timestamp = Static(":::")
        self.status_bar = Horizontal(self.njobs, self.ngpus, self.timestamp, classes="status_bar")

        self.node_table = DataTable(id="node_table")
        self.job_table = DataTable(id="job_table")
        self.history_table = DataTable(id="history_table")

        self.info_log = RichLog(wrap=True, highlight=True, id="info_log", auto_scroll=True)
        self.info_log.can_focus = False
        self.info_log.border_title = "Info"
        
        self.job_log = RichLog(wrap=True, highlight=True, id="job_log", auto_scroll=False)
        self.job_log_position = None
        
        self.tables = {
            "job": self.job_table,
            "node": self.node_table,
            "history": self.history_table
        }
        self.active_table = "job"
        
        yield self.header
        yield self.status_bar
        yield Container(self.job_table, self.node_table, self.history_table, self.info_log, self.job_log)
        yield self.footer

    def on_mount(self):
        self.rewrite_table("job")
        self.rewrite_table("node")
        self.rewrite_table("history")
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
        elif action == "refresh" and self.STAGE['action'] not in ['job', 'history', 'node', 'job_log']:
            return False
        elif action == "confirm" and self.STAGE['action'] != 'delete':
            return False
        elif action == "sort" and self.STAGE['action'] not in ['job', 'history', 'node']:
            return False
        elif action == "delete" and self.STAGE['action'] not in ['job', 'select']:
            return False
        elif action == "print_gpustat" and self.STAGE['action'] != 'job':
            return False
        elif action == "display_node_list" and self.STAGE['action'] not in ['job', 'node']:
            return False
        elif action == "display_job_log" and self.STAGE['action'] not in ['job', 'history']:
            return False
        elif action == "open_with_less" and self.STAGE['action'] not in ['job', 'job_log', 'history']:
            return False
        elif action == "display_history_jobs" and self.STAGE['action'] not in ['job', 'history']:
            return False
        elif action == "toggle_history_range" and self.STAGE['action'] != 'history':
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
        elif self.STAGE["action"] == "job_log":
            self.job_log_position = None
        elif self.STAGE["action"] == "select":
            self.info_log.write("Select: none")
            self.selected_jobid = []
        self.STAGE['action'] = "job"
        self.update_table("job")
        self.switch_display("job")
        self.refresh()

    @handle_error
    def action_select(self):
        if (self.STAGE["action"] == "job" and not self.selected_jobid) or self.STAGE["action"] == "select":
            i = self.tables[self.active_table].cursor_coordinate[0]
            value = str(self.tables[self.active_table].get_cell_at((i, 0)))

            job_id, _ = self._get_selected_job()
            if job_id in self.selected_jobid:
                self.selected_jobid.remove(job_id)
                self.tables[self.active_table].update_cell_at((i, 0), value)
            else:
                self.selected_jobid.append(job_id)
                self.tables[self.active_table].update_cell_at((i, 0), Text(str(value), style=self.selected_text_style))
            
            if self.selected_jobid:
                self.STAGE["action"] = "select"
                self.info_log.write(f"Select: {' '.join(self.selected_jobid)}")
            else:
                self.STAGE["action"] = "job"
                self.info_log.write(f"Select: none")
            self.tables[self.active_table].action_cursor_down()
        self.refresh_bindings()  
   
    @handle_error
    def action_select_inverse(self):
        assert self.STAGE["action"] in ["job", "select"]
        for i in range(len(self.tables[self.active_table].rows)):
            job_id = str(self.tables[self.active_table].get_cell_at((i, 0)))
            
            if job_id in self.selected_jobid:
                self.selected_jobid.remove(job_id)
                self.tables[self.active_table].update_cell_at((i, 0), job_id)
            else:
                self.selected_jobid.append(job_id)
                self.tables[self.active_table].update_cell_at((i, 0), Text(str(job_id), style=self.selected_text_style))
                self.tables[self.active_table].move_cursor(row=i, column=0)
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
            self.update_table("job")
        elif self.STAGE["action"] == "job_log":
            self.update_log(self.STAGE["log_fn"])
        elif self.STAGE["action"] == "node":
            self.update_table("node")
        elif self.STAGE["action"] == "history":
            self.update_table("history")
        self.update_status()
    
    @handle_error
    def action_confirm(self):
        # job to delete
        if self.STAGE["action"] == "delete":
            perform_scancel(self.STAGE['job_id'])
            self.info_log.write(f"Delete: {self.STAGE['job_id']}? succeeded")
            self.selected_jobid = []
            self.update_table("job")
            self.STAGE["action"] = "job"
        self.refresh_bindings()

    @handle_error
    def action_sort(self):
        sort_column = self.tables[self.active_table].cursor_column
        if sort_column != self.STAGE[self.STAGE["action"]].get("sort_column"):
            self.STAGE[self.STAGE["action"]]["sort_ascending"] = False
        else:
            self.STAGE[self.STAGE["action"]]["sort_ascending"] = not self.STAGE[self.STAGE["action"]].get("sort_ascending", True)
        self.STAGE[self.STAGE["action"]]['sort_column'] = sort_column

        self.rewrite_table(self.active_table, keep_state=True)
        self.tables[self.active_table].move_cursor(row=0, column=sort_column)
    
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
            self.update_table("node")
            self.switch_display("node")
            self.refresh_bindings()
        elif self.STAGE["action"] == "node":
            self.show_all_nodes = not self.show_all_nodes
            self.update_table("node")
            self.print_node_prompt()
    
    @handle_error
    def action_display_job_log(self):
        if self.STAGE["action"] in ["job", "history"]:
            job_id, job_name = self._get_selected_job()
            log_fn = self._get_log_fn(job_id)
            assert os.path.exists(log_fn), f"Log file not found: {log_fn}"
            self.STAGE.update({"action": "job_log", "job_id": job_id, "log_fn": log_fn, "job_name": job_name})
            self.update_log(log_fn)
            self.switch_display("job_log")
        self.refresh_bindings()
    
    @handle_error
    def action_open_with_less(self):
        if self.STAGE["action"] in ["job", "job_log", "history"]:
            job_id, _ = self._get_selected_job()
            log_fn = self._get_log_fn(job_id)
            assert os.path.exists(log_fn), f"Log file not found: {log_fn}"
            with self.suspend():
                subprocess.run(['less', log_fn])
            self.refresh()

    @handle_error
    def action_display_history_jobs(self):
        if self.STAGE["action"] == "job":
            self.STAGE.update({"action": "history"})
            self.update_table("history")
            self.switch_display("history")
            self.refresh_bindings()
        elif self.STAGE["action"] == "history":
            self.show_all_history = not self.show_all_history
            self.update_table("history")
            self.print_history_prompt()

    @handle_error
    def action_toggle_history_range(self):
        if self.history_range == "1 week":
            self.history_range = "1 month"
        elif self.history_range == "1 month":
            self.history_range = "4 months"
        elif self.history_range == "4 months":
            self.history_range = "1 year"
        else:
            self.history_range = "1 week"
        self.update_table("history")
        self.print_history_prompt()
    
    def print_node_prompt(self):
        info = f"Press 'g' to toggle nodes: {'All' if self.show_all_nodes else 'Available'}"
        self.info_log.clear()
        self.info_log.write(info)

    def print_history_prompt(self):
        info = f"Press 'H' to toggle history range: {self.history_range}\n" \
        + f"Press 'h' to toggle job state: {'All' if self.show_all_history else 'Completed'}"
        self.info_log.clear()
        self.info_log.write(info)


    def switch_display(self, action):
        if action == "job":
            self.job_table.styles.height = "80%"
            self.active_table = action
            self.tables[self.active_table].focus()
            self.info_log.styles.border = (self.border_type, self.border_color)
            self.info_log.styles.height="20%"
            self.info_log.clear()
            self.info_log.focus()

            self.history_table.styles.height = "0%"
            self.node_table.styles.height = "0%"
            self.job_log.styles.border = (self.border_type, self.border_color)
            self.job_log.styles.height="0%"
            self.job_log.clear()
        elif action == "node":
            self.node_table.styles.height = "80%"
            self.active_table = action
            self.tables[self.active_table].focus()
            self.info_log.styles.height="20%"
            self.info_log.styles.border = (self.border_type, self.border_color)
            self.info_log.clear()
            self.print_node_prompt()
        
            self.job_table.styles.height = "0%"
            self.history_table.styles.height = "0%"
            self.job_log.styles.height="0%"
            self.job_log.styles.border = (self.border_type, self.border_color)
            self.job_log.clear()
        elif action == "job_log":
            self.job_log.styles.height="100%"
            self.job_log.styles.border = (self.border_type, self.border_color)
            self.job_log.focus()
            
            self.job_table.styles.height="0%"
            self.node_table.styles.height="0%"
            self.history_table.styles.height="0%"
            self.info_log.styles.height="0%"
            self.info_log.styles.border = ("none", self.border_color)
        elif action == "history":
            self.history_table.styles.height = "80%"
            self.active_table = action
            self.tables[self.active_table].focus()
            self.info_log.styles.height="20%"
            self.info_log.styles.border = (self.border_type, self.border_color)
            self.info_log.clear()
            self.print_history_prompt()

            self.job_table.styles.height = "0%"
            self.node_table.styles.height = "0%"
            self.job_log.styles.height="0%"
            self.job_log.styles.border = (self.border_type, self.border_color)
            self.job_log.clear()
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

    def query_jobs(self, sort_column=None, sort_ascending=True):
        squeue_df = get_squeue() 
        if sort_column is not None:
            squeue_df = squeue_df.sort_values(squeue_df.columns[sort_column], ascending=sort_ascending)

        self.stats['njobs'] = len(squeue_df)
        self.stats['njobs_running'] = sum(1 for row in squeue_df.iterrows() if row[1]['STATE'] == 'RUNNING')
        return squeue_df

    @run_in_thread
    @handle_error
    def rewrite_table(self, table_type, keep_state=False):
        if table_type not in self.STAGE:
            self.STAGE[table_type] = {}
        if 'sort_column' in self.STAGE[table_type]:
            sort_column = self.STAGE[table_type]['sort_column']
        else:
            sort_column = None
        if 'sort_ascending' in self.STAGE[table_type]:
            sort_ascending = self.STAGE[table_type]['sort_ascending']
        else:
            sort_ascending = True

        df = self.query_table_data(table_type, sort_column, sort_ascending)
        self.update_status()

        table = self.tables[table_type]
        if keep_state:
            cursor_column = table.cursor_column
            table.clear()
            table.move_cursor(row=0, column=cursor_column)
        else:
            table.clear(columns=True)
            table.add_columns(*df.columns)
        
        for _, row in df.iterrows():
            table_row = [str(row[col]) for col in df.columns]
            table.add_row(*table_row)

    @run_in_thread
    @handle_error
    def update_table(self, table_type):
        if 'sort_column' in self.STAGE[table_type]:
            sort_column = self.STAGE[table_type]['sort_column']
        else:
            sort_column = None
        if 'sort_ascending' in self.STAGE[table_type]:
            sort_ascending = self.STAGE[table_type]['sort_ascending']
        else:
            sort_ascending = True

        df = self.query_table_data(table_type, sort_column, sort_ascending)
        self.update_status()

        table = self.tables[table_type]
        if not table.columns:
            table.add_columns(*df.columns)
            for _, row in df.iterrows():
                table_row = [str(row[col]) for col in df.columns]
                table.add_row(*table_row)
            return

        for row_index, (_, row) in enumerate(df.iterrows()):
            table_row = [str(row[col]) for col in df.columns]
            if row_index < len(table.rows):
                for col_index, cell in enumerate(table_row):
                    if table.get_cell_at((row_index, col_index)) != cell:
                        table.update_cell_at((row_index, col_index), cell)
            else:
                table.add_row(*table_row)

        while len(table.rows) > len(df):
            row_key, _ = table.coordinate_to_cell_key((len(table.rows) - 1, 0))
            table.remove_row(row_key)

    def query_table_data(self, table_type, sort_column=None, sort_ascending=True):
        if table_type == "job":
            return self.query_jobs(sort_column, sort_ascending)
        elif table_type == "node":
            return self.query_gpus(sort_column, sort_ascending)
        elif table_type == "history":
            return self.query_history(sort_column, sort_ascending)
        else:
            raise ValueError(f"Invalid table type: {table_type}")

    def _get_selected_job(self):
        row_idx = self.tables[self.active_table].cursor_row
        row = self.tables[self.active_table].get_row_at(row_idx)
        job_id = str(row[0])
        job_name = str(row[2])
        return job_id, job_name

    @handle_error
    def update_log(self, log_fn):
        self.job_log.border_title = f"{log_fn}"
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
    
    @handle_error
    def _get_log_fn(self, job_id):
        if self.STAGE["action"] == "history":
            response_string = subprocess.check_output(f"""sacct -j {job_id} --format=StdOut -P""", shell=True).decode("utf-8")
            formatted_string = response_string.split("\n")[1].strip()
            formatted_string = formatted_string.replace("%j", job_id)
        elif self.STAGE["action"] in ["job", "job_log"]:
            response_string = subprocess.check_output(f"""scontrol show job {job_id} | grep StdOut""", shell=True).decode("utf-8")
            formatted_string = response_string.split("=")[-1].strip()
        else:
            raise ValueError(f"Cannot get log file for action: {self.STAGE['action']}")
        return formatted_string

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

    def query_history(self, sort_column=None, sort_ascending=True):
        starttime = self.get_history_starttime()
        sacct_df = get_sacct(starttime=starttime)
        if sort_column is not None:
            sacct_df = sacct_df.sort_values(sacct_df.columns[sort_column], ascending=sort_ascending)
        
        if not self.show_all_history:
            sacct_df = sacct_df[sacct_df["State"] == "COMPLETED"]
        return sacct_df

    def get_history_starttime(self):
        if self.history_range == "1 week":
            return (datetime.now() - timedelta(weeks=1)).strftime('%Y-%m-%d')
        elif self.history_range == "1 month":
            return (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        elif self.history_range == "4 months":
            return (datetime.now() - timedelta(days=120)).strftime('%Y-%m-%d')
        elif self.history_range == "1 year":
            return (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        else:
            return "2024-11-26"

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
        response_string = subprocess.check_output(f"""squeue --format="%.18i{sep}%.20P{sep}%.200j{sep}%.8T{sep}%.10M{sep}%.6l{sep}%.S{sep}%.10u{sep}%Q{sep}%.4D{sep}%R" --me -S T""", shell=True).decode("utf-8")
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

def get_sacct(starttime="2024-11-26", endtime="now"):
    sep = "|"
    response_string = subprocess.check_output(
        f"""sacct --format="JobID,JobName,State,Start,Elapsed,NodeList,Partition,StdOut" -P --starttime={starttime} --endtime={endtime}""",
        shell=True
    ).decode("utf-8")
    data = io.StringIO(response_string)
    df = pd.read_csv(data, sep=sep)

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Strip whitespace from each string element in the DataFrame
    for col in df.select_dtypes(['object']).columns:
        df[col] = df[col].str.strip()
    
    # Filter to keep only the main job entries
    df = df[~df['JobID'].str.contains(r'\.')]

    # Filter out entries where StdOut is NaN (interactive jobs)
    df = df[df['StdOut'].notna()]

    return df

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
