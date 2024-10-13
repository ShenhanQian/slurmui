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


DEBUG = False
if DEBUG:
    from slurmui.debug_strings import SINFO_DEBUG, SQUEUE_DEBUG


class SlurmUI(App):
    cluster = None

    BINDINGS = [
        Binding("d", "stage_delete", "Delete job"),
        Binding("l", "display_log", "Log"),
        Binding("g", "display_gpu", "GPU"),
        Binding("r", "refresh", "Refresh"),
        Binding("s", "sort", "Sort"),
        Binding("q", "abort_quit", "Quit"),
        Binding("enter", "confirm", "Confirm", priority=True),
        Binding("escape", "abort_quit", "Abort"),
        # ("k", "scroll_up", "Up")    
    ]
    STAGE = {"action": "monitor"}
    gpu_overview_df = None
    sqeue_df =None

    def compose(self) -> ComposeResult:
        self.header = Header()
        self.footer = Footer()
        self.table = DataTable( id="table")
        self.table.zebra_stripes = True
        self.txt_log = RichLog(wrap=True, highlight=True, id="info")
        yield self.header
        yield Container(self.table, self.txt_log)
        yield self.footer

        if self.interval > 0:
            self.set_interval(self.interval, self.auto_refresh)
    
    def query_squeue(self, sort_column=None, sort_ascending=True):
        squeue_df = get_squeue() 
        if sort_column is not None:
            squeue_df = squeue_df.sort_values(squeue_df.columns[sort_column],ascending=sort_ascending)
        self.sqeue_df = squeue_df
        return squeue_df

    def update_squeue_table(self, sort_column=None, sort_ascending=True):
        self.table.clear(columns=True)
        squeue_df = self.query_squeue(sort_column=sort_column, sort_ascending=sort_ascending)
        # add device information
        squeue_df["GPU_IDS"] = "N/A"
        real_session_mask = squeue_df["PARTITION"]!="in"
        if real_session_mask.any():
            squeue_df["GPU_IDS"][real_session_mask] =squeue_df[real_session_mask]["JOBID"].apply(lambda x: get_job_gpu_ids(x))
        # self.table.columns = []
        self.table.add_columns(*squeue_df.columns)
        for row in squeue_df.iterrows():
            table_row = [str(x) for x in row[1].values]
            self.table.add_row(*table_row)
        self.table.focus()

    def auto_refresh(self):
        self.action_refresh()
    
    def action_refresh(self):
        self.query_gpus()
        self.last_cursor_coordinate  = self.table.cursor_coordinate  # memorize cursor position

        if self.STAGE["action"] == "monitor":
            self.update_squeue_table()
        elif self.STAGE["action"] == "log":
            self.update_log(self.STAGE["job_id"])
        elif self.STAGE["action"] == "gpu":
            self.update_gpu_table()
        self.table.cursor_coordinate = self.last_cursor_coordinate
        
        self.restore_sort()

    def on_mount(self):
        self._minimize_text_log()

    def on_ready(self) -> None:
        self.update_squeue_table()
        self.query_gpus()

    # def action_modal(self):
    #     log_screen = LogScreen()
    #     #log_screen.load_log()
    #     self.push_screen(log_screen)
    #     log_screen.on_ready()

    def _get_selected_job(self):
        row_idx = self.table.cursor_row
        row = self.table.get_row_at(row_idx)
        job_id = row[0]
        job_name = row[2]
        return job_id, job_name

    def _minimize_text_log(self):
        self.table.styles.height="80%"
        self.txt_log.styles.max_height="10%"
        self.txt_log.can_focus = False
        self.txt_log.styles.border = ("heavy","grey")
    def _maximize_text_log(self):
        self.table.styles.height="0%"
        self.txt_log.styles.max_height="100%"
        self.txt_log.can_focus = True
        self.txt_log.styles.border = ("heavy","white")

    def action_stage_delete(self):
        if self.STAGE['action'] == "monitor":
            try:
                job_id, job_name = self._get_selected_job()
                self.txt_log.clear()
                self.txt_log.write(f"Delete: {job_id} - {job_name}? Press <<ENTER>> to confirm")
                self.STAGE = {"action": "delete", "job_id": job_id, "job_name": job_name}
            except Exception as e:
                self.txt_log.clear()
                self.txt_log.write(str(e))

    def action_display_log(self):
        try:
            if self.STAGE["action"] == "monitor":
                job_id, job_name = self._get_selected_job()
                self.STAGE = {"action": "log", "job_id": job_id, "job_name": job_name}
                self._maximize_text_log()
                self.update_log(job_id)
        except Exception as e:
            self.txt_log.clear()
            self.txt_log.write(str(e))

    def query_gpus(self,  sort_column=None, sort_ascending=True):
        overview_df = get_sinfo(self.cluster)
        if sort_column is not None:
            overview_df = overview_df.sort_values(overview_df.columns[sort_column],ascending=sort_ascending)
        
        # also change the title to include GPU information
        total_num_gpus = overview_df["#Total"].sum()
        total_available = overview_df["#Avail"].sum()
        self.title = f"SlurmUI --- GPU STATS: {total_available}/{total_num_gpus} -- Version: {importlib.metadata.version('slurmui')}"
        return overview_df


    def update_gpu_table(self, sort_column=None, sort_ascending=True):
        self.table.clear(columns=True)
        overview_df = self.query_gpus(sort_column=sort_column, sort_ascending=sort_ascending)
        self.table.add_columns(*overview_df.columns)
        for row in overview_df.iterrows():
            table_row = [str(x) for x in row[1].values]
            self.table.add_row(*table_row)
        self.table.focus()

    def action_display_gpu(self):
        self.STAGE = {"action": "gpu"}
        try:
            self.update_gpu_table()
        except Exception as e:
            self.txt_log.clear()
            self.txt_log.write(str(e))

    def action_sort(self):
        column_idx = self.table.cursor_column
        if column_idx != self.STAGE.get("column_idx"):
            self.STAGE["sort_ascending"] = False
        else:
            self.STAGE["sort_ascending"] = not self.STAGE.get("sort_ascending", True)
        self.STAGE['column_idx'] = column_idx
        if self.STAGE["action"] == "monitor":
            self.update_squeue_table(sort_column=column_idx, sort_ascending=self.STAGE["sort_ascending"])
        elif self.STAGE["action"] == "gpu":
            self.update_gpu_table(sort_column=column_idx, sort_ascending=self.STAGE["sort_ascending"])
        self.table.cursor_coordinate = (0, column_idx)
        
    def restore_sort(self):
        if self.STAGE.get("column_idx", None) is None or self.STAGE.get("sort_ascending", None) is None:
            return

        if self.STAGE["action"] == "monitor":
            self.update_squeue_table(sort_column=self.STAGE["column_idx"], sort_ascending=self.STAGE["sort_ascending"])
        elif self.STAGE["action"] == "gpu":
            self.update_gpu_table(sort_column=self.STAGE["column_idx"], sort_ascending=self.STAGE["sort_ascending"])
        self.table.cursor_coordinate = self.last_cursor_coordinate

    def update_log(self, job_id):
        if not DEBUG:
            try:
                log_fn = get_log_fn(job_id)
                txt_lines = read_log(log_fn)
            except:
                txt_lines = ["Log file not found"]
        else:
            txt_lines = read_log("~/ram_batch_triplane0_l1.txt")

        self.txt_log.clear()
        for text_line in txt_lines:
            self.txt_log.write(text_line)

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
            self._minimize_text_log()
        elif self.STAGE["action"] == "gpu":
            self.update_squeue_table()
        self.txt_log.clear()
        self.STAGE['action'] = "monitor"

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
        response_string = subprocess.check_output(f"""squeue --format="%.18i{sep}%Q{sep}%.20P{sep}%.40j{sep}%.10u{sep}%.8T{sep}%.10M{sep}%.6l{sep}%S{sep}%.4D{sep}%R" --me -S T""", shell=True).decode("utf-8")
    formatted_string = re.sub(' +', ' ', response_string)
    data = io.StringIO(formatted_string)
    df = pd.read_csv(data, sep=sep)
    df.columns = [x.strip() for x in df.columns]
    return df 

def get_job_gpu_ids(job_id):
    try:
        # response_string = subprocess.check_output(f"""srun --jobid {job_id} '/bin/env' | grep SLURM_STEP_GPUS""", shell=True,stderr=subprocess.STDOUT, timeout=0.3).decode("utf-8")
        # response_string = subprocess.check_output(f"""scontrol --dd show job {job_id} | grep GRES=gpu:""", shell=True).decode("utf-8")
        # gpu_ids = re.match(response_string, ".*GRES=gpu:.*\(IDX:(.*)\)").groups()[0]
        # formatted_string = gpu_ids # response_string.split("=")[-1].strip()
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
