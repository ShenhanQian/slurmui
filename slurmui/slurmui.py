import io
import importlib.metadata
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Header, Footer, RichLog, DataTable, Tabs, Tab
from textual.containers import Container
from rich.text import Text
import subprocess
import pandas as pd
import re
import os
import threading
from functools import wraps
from datetime import datetime, timedelta
import time
import socket
import signal


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
            os.system(f"echo {e}")
    return wrapper


class SlurmUI(App):
    # configuration that can be set from the command line
    verbose = False
    cluster = None
    interval = 10
    history_range = "1 week"  # Default history range

    # internal states
    STAGE = {
        "action": "job",
        "job": {
            # "sort_column": 0,
            # "sort_ascending": True,
        },
        "history": {
            "sort_column": 0,
            "sort_ascending": False,
        },
        "node": {
            # "sort_column": 0,
            # "sort_ascending": True,
        },
    }
    stats = {}
    selected_jobid = []
    show_all_nodes = False
    show_all_history = True
    show_all_jobs = False

    theme = "textual-dark"
    selected_text_style = "bold on orange3"
    border_type = "solid"
    border_color = "white"

    CSS_PATH = "slurmui.tcss"
    TITLE = f"SlurmUI (v{importlib.metadata.version('slurmui')})"

    BINDINGS = [
        Binding("g", "display_nodes", "GPUs"),
        Binding("h", "display_history_jobs", "History"),
        Binding("j", "display_jobs", "Jobs"),
        Binding("q", "abort", "Abort"),
        Binding("space", "select", "Select"),
        Binding("v", "select_inverse", "Inverse"),
        Binding("r", "refresh", "Refresh"),
        Binding("enter", "confirm", "Confirm", priority=True, key_display='enter'),
        Binding("s", "sort", "Sort"),
        Binding("d", "delete", "Delete"),
        Binding("G", "print_gpustat", "GPU"),
        Binding("l", "display_job_log", "Log"),
        Binding("L", "open_with_less", "Open with less"),
        Binding("H", "toggle_history_range", "Range"),
    ]

    def compose(self) -> ComposeResult:
        self.header = Header()
        self.footer = Footer()

        self.tab_nodes = Tab("GPUs", id="node")
        self.tab_history = Tab("History", id="history")
        self.tab_jobs = Tab("Jobs", id="job")
        self.tab_time = Tab("Time", id="time", disabled=True, classes="time-tab")
        self.tabs = Tabs(self.tab_nodes, self.tab_history, self.tab_jobs, self.tab_time, id="tabs")
        self.tabs.can_focus = False

        self.node_table = DataTable(id="node_table")
        self.job_table = DataTable(id="job_table")
        self.history_table = DataTable(id="history_table")
        self.tables = {
            "job": self.job_table,
            "node": self.node_table,
            "history": self.history_table
        }
        self.active_table = "job"

        self.info_log = RichLog(wrap=True, highlight=True, id="info_log", auto_scroll=True)
        self.info_log.can_focus = False
        self.info_log.border_title = "Info"
        
        self.job_log = RichLog(wrap=True, highlight=True, id="job_log", auto_scroll=False)
        self.job_log_position = None
        
        
        yield self.header
        yield self.tabs
        yield Container(self.job_table, self.node_table, self.history_table, self.info_log, self.job_log)
        yield self.footer

    def on_mount(self):
        pass

    def on_ready(self) -> None:
        self.rewrite_table("job")
        self.rewrite_table("node")
        self.rewrite_table("history")
        self.tabs.active = 'job'
        if self.interval > 0:
            self.set_interval(self.interval, self.auto_refresh)

    def on_tabs_tab_activated(self, message):
        tab_id = message.tab.id
        if self.verbose:
            self.info_log.write(f"Tab activated: {tab_id}")

        self.STAGE.update({"action": tab_id})
        self.update_table(tab_id)
        self.switch_display(tab_id)
        self.refresh_bindings()
        
    @handle_error
    def check_action(self, action: str, parameters):  
        """Check if an action may run."""
        if action == "display_nodes" and self.STAGE['action'] not in ['node', 'history', 'job']:
            return False
        elif action == "display_history_jobs" and self.STAGE['action'] not in ['node', 'history', 'job']:
            return False
        elif action == "display_jobs" and self.STAGE['action'] not in ['node', 'history', 'job']:
            return False
        elif action == "abort":
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
        elif action == "display_job_log" and self.STAGE['action'] not in ['job', 'history']:
            return False
        elif action == "open_with_less" and self.STAGE['action'] not in ['job', 'job_log', 'history']:
            return False
        elif action == "toggle_history_range" and self.STAGE['action'] != 'history':
            return False
        return True

    @handle_error
    def action_display_nodes(self):
        if self.STAGE[self.active_table]['updating']:
            return
        if self.STAGE["action"] in ["history", "job"]:
            self.STAGE.update({"action": "node"})
            self.update_table("node")
            self.refresh_bindings()
            self.tabs.active = "node"
        elif self.STAGE["action"] == "node":
            self.show_all_nodes = not self.show_all_nodes
            self.rewrite_table("node", keep_state=True)
            self.switch_display("node")

    @handle_error
    def action_display_history_jobs(self):
        if self.STAGE[self.active_table]['updating']:
            return
        if self.STAGE["action"] in ["node", "job"]:
            self.STAGE.update({"action": "history"})
            self.update_table("history")
            self.refresh_bindings()
            self.tabs.active = "history"
        elif self.STAGE["action"] == "history":
            self.show_all_history = not self.show_all_history
            self.rewrite_table("history", keep_state=True)
            self.switch_display("history")
    
    @handle_error
    def action_display_jobs(self):
        if self.STAGE[self.active_table]['updating']:
            return
        if self.STAGE["action"] in ["node", "history"]:
            self.STAGE.update({"action": "job"})
            self.update_table("job")
            self.refresh_bindings()
            self.tabs.active = "job"
        elif self.STAGE["action"] == "job":
            self.show_all_jobs = not self.show_all_jobs
            self.rewrite_table("job", keep_state=True)
            self.refresh_bindings()
            self.switch_display("job")

    @handle_error
    def action_abort(self):
        if self.STAGE["action"] == "delete":
            self.info_log.write("Delete: aborted")
            self.selected_jobid = []
            self.STAGE.pop("job_id", None)
            self.STAGE.pop("job_name", None)
        elif self.STAGE["action"] == "job_log":
            self.job_log_position = None
            self.STAGE.pop("job_id", None)
            self.STAGE.pop("job_name", None)
            self.STAGE.pop("log_fn", None)
        elif self.STAGE["action"] == "select":
            self.info_log.write("Select: none")
            self.selected_jobid = []
        elif self.STAGE["action"] in ["node", "history"]:
            self.tabs.active = "job"
        action = self.tabs.active
        self.STAGE['action'] = action
        self.update_table(action)
        self.switch_display(action)
        self.refresh()

    @handle_error
    def action_select(self):
        if (self.STAGE["action"] == "job" and not self.selected_jobid) or self.STAGE["action"] == "select":
            i = self.tables[self.active_table].cursor_coordinate[0]
            value = str(self.tables[self.active_table].get_cell_at((i, 0)))

            job_id = self._get_selected_job()
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
    
    @run_in_thread
    @handle_error
    def auto_refresh(self):
        if self.STAGE["action"] == "job":
            self.update_table("job")
        elif self.STAGE["action"] == "job_log":
            self.update_log(self.STAGE["log_fn"])
        elif self.STAGE["action"] == "node":
            self.update_table("node")
        # elif self.STAGE["action"] == "history":
        #     self.update_table("history")
        self.update_status()

    @handle_error
    def action_refresh(self):
        if self.STAGE["action"] == "job":
            self.rewrite_table("job", keep_state=True)
        elif self.STAGE["action"] == "job_log":
            self.update_log(self.STAGE["log_fn"])
        elif self.STAGE["action"] == "node":
            self.rewrite_table("node", keep_state=True)
        elif self.STAGE["action"] == "history":
            self.rewrite_table("history", keep_state=True)
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
            job_id = self._get_selected_job()
            self.info_log.write(f"Delete: {job_id}? press <<ENTER>> to confirm")
            self.STAGE.update({"action": "delete", "job_id": job_id})
        elif self.STAGE["action"] == "select":
            self.info_log.write(f"Delete: {' '.join(self.selected_jobid)}? press <<ENTER>> to confirm")
            self.STAGE.update({"action": "delete", "job_id": ' '.join(self.selected_jobid)})
        self.refresh_bindings()  

    @handle_error
    def action_print_gpustat(self):
        if self.STAGE["action"] == "job":
            job_id = self._get_selected_job()
            gpustat = subprocess.check_output(f"""srun --jobid {job_id} gpustat""", shell=True, timeout=3).decode("utf-8").rstrip()
            self.info_log.write(gpustat)
    
    @handle_error
    def action_display_job_log(self):
        if self.STAGE["action"] in ["job", "history"]:
            job_id = self._get_selected_job()
            log_fn = self._get_log_fn(job_id)
            assert os.path.exists(log_fn), f"Log file not found: {log_fn}"
            self.STAGE.update({"action": "job_log", "log_fn": log_fn})
            self.update_log(log_fn)
            self.switch_display("job_log")
        self.refresh_bindings()
    
    @handle_error
    def action_open_with_less(self):
        if self.STAGE["action"] in ["job", "job_log", "history"]:
            if 'log_fn' not in self.STAGE:
                job_id = self._get_selected_job()
                log_fn = self._get_log_fn(job_id)
            else:
                log_fn = self.STAGE['log_fn']
            assert os.path.exists(log_fn), f"Log file not found: {log_fn}"
            with self.suspend():
                # Save the current SIGINT handler
                original_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
                try:
                    subprocess.run(['less', '+G', log_fn])
                finally:
                    # Restore the original SIGINT handler
                    signal.signal(signal.SIGINT, original_sigint)
            self.refresh()

    @handle_error
    def action_toggle_history_range(self):
        if self.STAGE[self.active_table]['updating']:
            return
        if self.history_range == "1 week":
            self.history_range = "1 month"
        elif self.history_range == "1 month":
            self.history_range = "4 months"
        elif self.history_range == "4 months":
            self.history_range = "1 year"
        else:
            self.history_range = "1 week"
        self.rewrite_table("history", keep_state=True)
        self.switch_display("history")
    
    def print_tab_prompt(self, tab_id):
        if not self.verbose:
            self.info_log.clear()

        if tab_id == "node":
            info = f"Press 'g' to toggle nodes: {'All' if self.show_all_nodes else 'Available'}"
        elif tab_id == "history":
            info = f"Press 'h' to toggle job states: {'All' if self.show_all_history else 'Complete'}\t| " \
            + f"Press 'H' to toggle history range: {self.history_range}"
        elif tab_id == "job":
            info = f"Press 'j' to toggle users: {'All' if self.show_all_jobs else 'Me'}"
        self.info_log.write(info)

    def switch_display(self, action):
        if self.verbose:
            self.info_log.write(f"Switch display: {action}")
        if action == "node":
            self.node_table.styles.height = "80%"
            self.active_table = action
            self.tables[self.active_table].focus()
            self.info_log.styles.height="20%"
            self.info_log.styles.border = (self.border_type, self.border_color)
            self.print_tab_prompt(action)
        
            self.job_table.styles.height = "0%"
            self.history_table.styles.height = "0%"
            self.job_log.styles.height="0%"
            self.job_log.styles.border = (self.border_type, self.border_color)
            self.job_log.clear()
        elif action == "history":
            self.history_table.styles.height = "80%"
            self.active_table = action
            self.tables[self.active_table].focus()
            self.info_log.styles.height="20%"
            self.info_log.styles.border = (self.border_type, self.border_color)
            self.print_tab_prompt(action)

            self.job_table.styles.height = "0%"
            self.node_table.styles.height = "0%"
            self.job_log.styles.height="0%"
            self.job_log.styles.border = (self.border_type, self.border_color)
            self.job_log.clear()
        elif action == "job":
            self.job_table.styles.height = "80%"
            self.active_table = action
            self.tables[self.active_table].focus()
            self.info_log.styles.border = (self.border_type, self.border_color)
            self.info_log.styles.height="20%"
            self.print_tab_prompt(action)

            self.history_table.styles.height = "0%"
            self.node_table.styles.height = "0%"
            self.job_log.styles.border = (self.border_type, self.border_color)
            self.job_log.styles.height="0%"
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
        else:
            raise ValueError(f"Invalid action: {action}")

    @run_in_thread
    @handle_error
    def update_status(self):
        self.title = f"SlurmUI (v{importlib.metadata.version('slurmui')})"

        njobs = self.stats.get("njobs", 0)
        njobs_running = self.stats.get("njobs_running", 0)
        self.tab_jobs.label = f"Jobs: {njobs_running}/{njobs}"

        ngpus_avail = self.stats.get("ngpus_avail", 0)
        ngpus = self.stats.get("ngpus", 0)
        self.tab_nodes.label = f"GPUs: {ngpus_avail}/{ngpus}"

        nhistory = self.stats.get("nhistory", 0)
        nhistory_completed = self.stats.get("nhistory_completed", 0)
        self.tab_history.label = f"History: {nhistory_completed}/{nhistory}"
        
        self.tab_time.label = f"{socket.gethostname()} | {datetime.now().strftime('%H:%M:%S')}"

    @handle_error
    def query_jobs(self, sort_column=None, sort_ascending=True):
        squeue_df = get_squeue(self.cluster, self.show_all_jobs) 
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
        if self.STAGE[table_type].get('updating', False):
            return
        self.STAGE[table_type]['updating'] = True

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

        time.sleep(0.1)
        self.STAGE[table_type]['updating'] = False

    @run_in_thread
    @handle_error
    def update_table(self, table_type):
        if self.STAGE[table_type].get('updating', False):
            return
        self.STAGE[table_type]['updating'] = True
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
        self.STAGE[table_type]['updating'] = False

    @handle_error
    def query_table_data(self, table_type, sort_column=None, sort_ascending=True):
        if table_type == "job":
            return self.query_jobs(sort_column, sort_ascending)
        elif table_type == "node":
            return self.query_gpus(sort_column, sort_ascending)
        elif table_type == "history":
            return self.query_history(sort_column, sort_ascending)
        else:
            raise ValueError(f"Invalid table type: {table_type}")

    @handle_error
    def _get_selected_job(self):
        row_idx = self.tables[self.active_table].cursor_row
        row = self.tables[self.active_table].get_row_at(row_idx)
        job_id = str(row[0])
        return job_id

    @handle_error
    def update_log(self, log_fn):
        self.job_log.border_title = f"{log_fn}"
        current_scroll_y = self.job_log.scroll_offset[1]

        if not self.job_log_position:
            with open(log_fn, 'r') as f:
                self.job_log_position = max(sum(len(line) for line in f) - 2**12, 0)  # read the last 4KB
            
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

    @handle_error
    def query_gpus(self,  sort_column=None, sort_ascending=True):
        overview_df = get_sinfo(self.cluster)
        self.stats['ngpus'] = overview_df["GPUs (Total)"].sum()
        self.stats['ngpus_avail'] = overview_df["GPUs (Avail)"].sum()
        if not self.show_all_nodes:
            # filter out nodes with no available GPUs
            overview_df = overview_df[overview_df["GPUs (Avail)"] > 0]
        overview_df = overview_df[['Partition', 'Host', "Device", "State", "Mem (GB)", "CPUs", "GPUs", "Free IDX"]]
        if sort_column is not None:
            overview_df = overview_df.sort_values(overview_df.columns[sort_column],ascending=sort_ascending)
        return overview_df

    @handle_error
    def query_history(self, sort_column=None, sort_ascending=True):
        starttime = self.get_history_starttime()
        sacct_df = get_sacct(starttime=starttime)
        if sort_column is not None:
            sacct_df = sacct_df.sort_values(sacct_df.columns[sort_column], ascending=sort_ascending)
        
        self.stats['nhistory'] = len(sacct_df)

        if not self.show_all_history:
            sacct_df = sacct_df[sacct_df["State"] == "COMPLETED"]

        self.stats['nhistory_completed'] = len(sacct_df)
        return sacct_df

    @handle_error
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
            "GPUs (Avail)": num_gpus,
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
            "GPUs (Total)": num_gpus}

def remove_first_line(input_string):
    lines = input_string.split('\n')
    return '\n'.join(lines[1:])

def get_sinfo(cluster):
    if DEBUG:
        response_string = SINFO_DEBUG
    else:
        response_string = subprocess.check_output(f"""sinfo -O 'Partition:25,NodeHost,Gres:500,GresUsed:500,StateCompact,FreeMem,Memory,CPUsState'""", shell=True).decode("utf-8")

    formatted_string = re.sub(' +', ' ', response_string)
    data = io.StringIO(formatted_string)
    df = pd.read_csv(data, sep=" ")
    overview_df = [ ]# pd.DataFrame(columns=['Host', "Device", "GPUs (Avail)", "GPUs (Total)", "Free IDX"])
    for row in df.iterrows():
        node_available = row[1]["STATE"] in ["mix", "idle", "alloc"]

        if row[1]['GRES'] != "(null)":
            host_info = parse_gres(row[1]['GRES'], cluster)
        else:
            continue

        host_avail_info = parse_gres_used(row[1]['GRES_USED'], host_info["GPUs (Total)"], cluster)
        host_info.update(host_avail_info)
        if not node_available:
            host_info["GPUs (Avail)"] = 0
            host_info["Free IDX"] = []
        else:
            host_info["GPUs (Avail)"] = host_info['GPUs (Total)'] - host_info["GPUs (Avail)"]
        host_info["GPUs"] = f"{host_info['GPUs (Avail)']}/{host_info['GPUs (Total)']}"
        
        try:
            host_info['Mem (Avail)'] = int(row[1]["FREE_MEM"]) // 1024
        except:
            host_info['Mem (Avail)'] = row[1]["FREE_MEM"]
        try:
            host_info['Mem (Total)'] = int(row[1]["MEMORY"]) // 1024
        except:
            host_info['Mem (Total)'] = row[1]["MEMORY"]
        host_info['Mem (GB)'] = f"{host_info['Mem (Avail)']}/{host_info['Mem (Total)']}"

        cpu_info = row[1]["CPUS(A/I/O/T)"].split("/")
        host_info['CPUs (Avail)'] = cpu_info[1]
        host_info['CPUs (Total)'] = cpu_info[3]
        host_info['CPUs'] = f"{host_info['CPUs (Avail)']}/{host_info['CPUs (Total)']}"

        host_info['Host'] = str(row[1]["HOSTNAMES"])
        host_info['Partition'] = str(row[1]["PARTITION"])
        host_info['State'] = str(row[1]["STATE"])

        overview_df.append(host_info)
    overview_df = pd.DataFrame.from_records(overview_df).drop_duplicates("Host")
    return overview_df

def get_squeue(cluster=None, show_all_jobs=False):
    sep = "|"
    if DEBUG:
        response_string = SQUEUE_DEBUG
    else:
        if show_all_jobs:
            response_string = subprocess.check_output(f"""squeue --format="%18i{sep}%10u{sep}%20P{sep}%200j{sep}%8T{sep}%10M{sep}%S{sep}%l{sep}%.4D{sep}%R" -S T""", shell=True).decode("utf-8")
        else:
            response_string = subprocess.check_output(f"""squeue --format="%18i{sep}%10u{sep}%20P{sep}%200j{sep}%8T{sep}%10M{sep}%S{sep}%l{sep}%.4D{sep}%R" --me -S T""", shell=True).decode("utf-8")
    formatted_string = re.sub(' +', ' ', response_string)
    data = io.StringIO(formatted_string)
    df = pd.read_csv(data, sep=sep)

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Strip whitespace from each string element in the DataFrame
    for col in df.select_dtypes(['object']).columns:
        df[col] = df[col].str.strip()
    
    if "NODELIST(REASON)" in df.columns:
        df.rename(columns={"NODELIST(REASON)": "NODELIST"}, inplace=True)

    # right align time
    max_length = df["TIME"].str.len().max()
    df.loc[:, ["TIME"]] = df.loc[:, "TIME"].apply(lambda x: f"{x:>{max_length}}")
    
    # remove years from start time
    df.loc[:, ["START_TIME"]] = df.loc[:, "START_TIME"].apply(lambda x: simplify_start_time(x))
    
    # remove seconds from time limit
    max_length = df["TIME_LIMIT"].str.len().max()
    df.loc[:, ["TIME_LIMIT"]] = df.loc[:, "TIME_LIMIT"].apply(lambda x: f"{x[:-3]:>{max_length-3}}")
    
    # Add allocated GPU IDs
    mask = (df["PARTITION"] != "in") & (df["STATE"] == "RUNNING")
    if mask.any():
        df["GRES"] = "N/A"
        df.loc[mask, ["GRES"]] = df.loc[mask, "JOBID"].apply(lambda x: get_job_gres(x))
        # Reorder columns to make "GRES" the second to last column
        columns = list(df.columns)
        columns.remove("GRES")
        columns.insert(-2, "GRES")
        df = df[columns]
    return df 

def simplify_start_time(start_time):
    try:
        if start_time != "nan":
            start_time = datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S").strftime("%m-%d %H:%M")
    except Exception as e:
        pass
    return start_time

def get_job_gpu_ids(job_id):
    try:
        response_string = subprocess.check_output(f"""scontrol show jobid -dd {job_id} | grep GRES""", shell=True).decode("utf-8")
        formatted_string = response_string.split(":")[-1].strip()[:-1]
    except:
        return "N/A"
    return formatted_string

def get_job_gres(job_id):
    response_string = subprocess.check_output(f"""scontrol show jobid -dd {job_id} | grep JOB_GRES""", shell=True).decode("utf-8")
    pattern = r"gpu:([^,]+)"
    matches = re.findall(pattern, response_string)
    job_gres = ",".join(matches)  # Join extracted parts with a comma
    return job_gres

def get_sacct(starttime="2024-11-26", endtime="now"):
    response_string = subprocess.check_output(
        f"""sacct --format="JobID,JobName,State,Start,Elapsed,NodeList,Partition,StdOut" -P --starttime={starttime} --endtime={endtime}""",
        shell=True
    ).decode("utf-8")
    data = io.StringIO(response_string)
    df = pd.read_csv(data, sep='|')

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Strip whitespace from each string element in the DataFrame
    for col in df.select_dtypes(['object']).columns:
        df[col] = df[col].str.strip()
    
    # Filter out entries where Start or StdOut is NaN (interactive jobs)
    df = df.dropna(subset=['Start', 'StdOut'])
    return df

def read_log(fn, num_lines=100):
    with open(os.path.expanduser(fn), 'r') as f:
        txt_lines = list(f.readlines()[-num_lines:])
    
    return txt_lines

def run_ui(verbose=False, cluster=None, interval=10, history_range="1 week"):
    # if debug:
    #     # global for quick debugging
    #     global DEBUG
    #     DEBUG = True
    app = SlurmUI()
    app.verbose = verbose
    app.cluster = cluster
    app.interval = interval
    app.history_range = history_range
    app.run()


if __name__ == "__main__":
    run_ui()
