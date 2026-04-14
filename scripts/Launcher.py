from pathlib import Path
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from utils.visualization_utils import build_patient_timeline_figure, get_full_discharge_summary, get_llm_summary

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Model configuration
MODELS = {
    "Mistral 7B": {
        "trajectory_path": PROJECT_ROOT / "data" / "patient_trajectories.csv",
        "llm_outputs_dir": PROJECT_ROOT / "llm_outputs"
    },
    "Gemma 4": {
        "trajectory_path": PROJECT_ROOT / "data" / "patient_trajectories_gemma4.csv",
        "llm_outputs_dir": PROJECT_ROOT / "llm_outputs_gemma4"
    }
}


class TrajectoryApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Patient Trajectory Viewer")
        self.root.geometry("520x350")
        self.root.configure(bg="#f0f0f0")

        self.selected_model = tk.StringVar(value="Mistral 7B")
        self.subject_ids = self.load_subject_ids()
        self.patient_df = None
        self.setup_styles()
        self.create_widgets()
        
        # Start local server to handle Plotly interaction
        self.server_port = 5050
        self.start_local_server()

    def start_local_server(self):
        """Start a simple HTTP server in a thread to receive requests from Plotly."""
        app_ref = self
        
        class RequestHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                parsed_url = urlparse(self.path)
                if parsed_url.path == '/open_ds':
                    params = parse_qs(parsed_url.query)
                    subject_id = params.get('subject_id', [None])[0]
                    hadm_id = params.get('hadm_id', [None])[0]
                    adm_idx = params.get('adm_idx', [None])[0]
                    
                    if subject_id and hadm_id and adm_idx:
                        # Safely call Tkinter from another thread
                        app_ref.root.after(0, app_ref.show_full_ds_explicit, subject_id, hadm_id, adm_idx)
                        
                        # Respond with a 1x1 transparent GIF to act as a stealthy "trigger"
                        # This allows Plotly buttons (via relayout image source) to trigger Python actions
                        self.send_response(200)
                        self.send_header('Content-type', 'image/gif')
                        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                        self.send_header('Pragma', 'no-cache')
                        self.send_header('Expires', '0')
                        self.send_header('Access-Control-Allow-Origin', '*')
                        self.end_headers()
                        
                        # 1x1 Transparent GIF pixel
                        self.wfile.write(b'GIF89a\x01\x00\x01\x00\x80\x00\x00\xff\xff\xff\x00\x00\x00!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;')
                    else:
                        self.send_error(400, "Missing parameters")
                elif parsed_url.path == '/open_summary':
                    params = parse_qs(parsed_url.query)
                    subject_id = params.get('subject_id', [None])[0]
                    hadm_id = params.get('hadm_id', [None])[0]
                    adm_idx = params.get('adm_idx', [None])[0]
                    group = params.get('group', [None])[0]
                    note_id = params.get('note_id', [None])[0]

                    if subject_id and hadm_id and adm_idx and group and note_id:
                        app_ref.root.after(0, app_ref.show_summary_explicit, subject_id, hadm_id, adm_idx, group, note_id)

                        self.send_response(200)
                        self.send_header('Content-type', 'image/gif')
                        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                        self.send_header('Pragma', 'no-cache')
                        self.send_header('Expires', '0')
                        self.send_header('Access-Control-Allow-Origin', '*')
                        self.end_headers()
                        self.wfile.write(b'GIF89a\x01\x00\x01\x00\x80\x00\x00\xff\xff\xff\x00\x00\x00!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;')
                    else:
                        self.send_error(400, "Missing parameters")
                else:
                    self.send_error(404)
            
            def log_message(self, format, *args):
                pass # Suppress logging to console

        def run_server():
            try:
                server = HTTPServer(('localhost', self.server_port), RequestHandler)
                server.serve_forever()
            except Exception as e:
                print(f"Server error: {e}")

        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()

    def load_subject_ids(self):
        """Load unique subject IDs from the trajectory data."""
        trajectory_path = MODELS[self.selected_model.get()]["trajectory_path"]
        try:
            if trajectory_path.exists():
                df = pd.read_csv(trajectory_path)
                if "subject_id" in df.columns:
                    ids = sorted(df["subject_id"].unique().astype(str).tolist())
                    return ids
        except Exception as e:
            print(f"Warning: Could not load subject IDs for dropdown: {e}")
        return []

    def on_model_change(self, *args):
        """Handle model selection change."""
        self.subject_ids = self.load_subject_ids()
        self.entry.config(values=self.subject_ids)
        if self.subject_ids:
            self.status_var.set(f"Loaded {len(self.subject_ids)} subjects for {self.selected_model.get()}")
        else:
            self.status_var.set(f"No data found for {self.selected_model.get()}")

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")  # Use a more modern theme
        style.configure("TFrame", background="#f0f0f0")
        style.configure("TLabel", background="#f0f0f0", font=("Segoe UI", 10))
        style.configure("Header.TLabel", font=("Segoe UI", 16, "bold"), foreground="#333")
        style.configure("Action.TButton", font=("Segoe UI", 10, "bold"), padding=10, background="#4D96FF", foreground="white")
        style.map("Action.TButton", background=[("active", "#3a81e6")])
        style.configure("Footer.TLabel", font=("Segoe UI", 8, "italic"), foreground="#888")

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(
            main_frame, 
            text="Patient Trajectory Viewer", 
            style="Header.TLabel"
        )
        title_label.pack(pady=(0, 10))

        # Description
        desc_label = ttk.Label(
            main_frame,
            text="Enter a Subject ID to visualise the patient's\nmedical journey through admissions.",
            justify=tk.CENTER
        )
        desc_label.pack(pady=(0, 20))

        # Model Selection
        model_frame = ttk.Frame(main_frame)
        model_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(model_frame, text="Select Model:").pack(side=tk.LEFT, padx=(0, 10))
        
        for model_name in MODELS.keys():
            ttk.Radiobutton(
                model_frame, 
                text=model_name, 
                variable=self.selected_model, 
                value=model_name,
                command=self.on_model_change
            ).pack(side=tk.LEFT, padx=10)

        # Entry/Dropdown Field
        entry_frame = ttk.Frame(main_frame)
        entry_frame.pack(fill=tk.X, pady=10)

        ttk.Label(entry_frame, text="Subject ID:").pack(side=tk.LEFT, padx=(0, 10))
        
        # Use Combobox for both dropdown and manual entry
        self.entry = ttk.Combobox(
            entry_frame, 
            values=self.subject_ids,
            font=("Segoe UI", 11)
        )
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.entry.bind("<Return>", lambda e: self.submit())

        # Submit Button
        self.submit_btn = ttk.Button(
            main_frame,
            text="Generate Timeline",
            style="Action.TButton",
            command=self.submit
        )
        self.submit_btn.pack(pady=(20, 0), fill=tk.X)

        # Status Bar
        self.status_var = tk.StringVar(value="Ready")
        
        # Footer Frame (to hold status and branding)
        footer_frame = ttk.Frame(self.root)
        footer_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        status_bar = tk.Label(
            footer_frame, 
            textvariable=self.status_var, 
            bd=1, 
            relief=tk.SUNKEN, 
            anchor=tk.W,
            font=("Segoe UI", 9)
        )
        status_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)

        footer_label = ttk.Label(
            footer_frame, 
            text="Powered by LLM-based clinical note extraction ", 
            style="Footer.TLabel",
            anchor=tk.E
        )
        footer_label.pack(side=tk.RIGHT, padx=5)

    def submit(self):
        subject_id = self.entry.get().strip()

        if not subject_id:
            messagebox.showwarning("Input Required", "Please enter a valid Subject ID.")
            return

        model_info = MODELS[self.selected_model.get()]
        trajectory_path = model_info["trajectory_path"]

        self.status_var.set("Loading data...")
        self.root.update_idletasks()

        try:
            if not trajectory_path.exists():
                messagebox.showerror("Error", f"Data file not found at:\n{trajectory_path}\n\nGemma 4 might not be processed yet.")
                return

            df = pd.read_csv(trajectory_path)
            self.patient_df = df[df["subject_id"].astype(str) == subject_id].copy()

            if self.patient_df.empty:
                messagebox.showerror("No Data", f"No records found for Subject ID: {subject_id}")
                return

            self.status_var.set("Generating timeline...")
            self.root.update_idletasks()

            fig = build_patient_timeline_figure(self.patient_df, subject_id)
            # Remove all modebar buttons except download as image
            fig.show(config={
                'modeBarButtonsToRemove': [
                    'zoom2d', 'pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 
                    'autoScale2d', 'resetScale2d', 'hoverClosestCartesian', 'hoverCompareCartesian',
                    'toggleSpikelines'
                ],
                'displaylogo': False
            })
            
            self.status_var.set(f"Displayed timeline for {subject_id}")

        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred:\n{str(e)}")
            self.status_var.set("Error occurred")
        finally:
            self.root.update_idletasks()

    def show_full_ds_explicit(self, subject_id, hadm_id, adm_idx):
        full_ds = get_full_discharge_summary(subject_id, hadm_id)
        
        # Create a new window for the DS
        ds_window = tk.Toplevel(self.root)
        ds_window.title(f"Full Clinical Note - Subject {subject_id}")
        ds_window.geometry("900x700")
        ds_window.configure(bg="#f8f9fa")
        
        # Focus the new window
        ds_window.lift()
        ds_window.attributes('-topmost', True)
        ds_window.after(1, lambda: ds_window.attributes('-topmost', False))

        # Header Frame
        header_frame = tk.Frame(ds_window, bg="#4D96FF", height=60)
        header_frame.pack(side=tk.TOP, fill=tk.X)
        
        header_label = tk.Label(
            header_frame, 
            text=f"Full Clinical Note: Admission {adm_idx}", 
            bg="#4D96FF", 
            fg="white",
            font=("Segoe UI", 14, "bold"),
            pady=10
        )
        header_label.pack()

        # Footer Frame (pinned to bottom)
        footer_frame = tk.Frame(ds_window, bg="#f8f9fa", pady=15)
        footer_frame.pack(side=tk.BOTTOM, fill=tk.X)

        close_btn = ttk.Button(footer_frame, text="Close Window", command=ds_window.destroy)
        close_btn.pack()

        # Content Frame
        content_frame = tk.Frame(ds_window, bg="#f8f9fa", padx=20, pady=10)
        content_frame.pack(expand=True, fill='both')
        
        # Add scrollable text area
        txt = scrolledtext.ScrolledText(
            content_frame, 
            undo=True, 
            wrap=tk.WORD, 
            font=("Consolas", 11),
            bg="white",
            relief=tk.FLAT,
            padx=10,
            pady=10
        )
        txt.insert(tk.INSERT, full_ds)
        txt.configure(state='disabled') # Make it read-only
        txt.pack(expand=True, fill='both')

    def show_summary_explicit(self, subject_id, hadm_id, adm_idx, group, note_id):
        llm_outputs_dir = MODELS[self.selected_model.get()]["llm_outputs_dir"]
        summary_html = get_llm_summary(group, subject_id, hadm_id, note_id, llm_outputs_dir)
        
        # Convert HTML-like summary to plain text for Tkinter
        summary_text = summary_html.replace("<b>", "").replace("</b>", "").replace("<br>", "\n")

        # Create a new window for the Summary
        sum_window = tk.Toplevel(self.root)
        sum_window.title(f"Analysis Summary - Subject {subject_id}")
        sum_window.geometry("800x700")
        sum_window.configure(bg="#f8f9fa")
        
        # Focus the new window
        sum_window.lift()
        sum_window.attributes('-topmost', True)
        sum_window.after(1, lambda: sum_window.attributes('-topmost', False))
        
        # Header Frame
        header_frame = tk.Frame(sum_window, bg="#FF6B6B", height=60)
        header_frame.pack(side=tk.TOP, fill=tk.X)
        
        header_label = tk.Label(
            header_frame, 
            text=f"AI-Generated Analysis: Admission {adm_idx}", 
            bg="#FF6B6B", 
            fg="white",
            font=("Segoe UI", 14, "bold"),
            pady=10
        )
        header_label.pack()

        # Footer Frame (pack FIRST at the bottom to ensure it's always visible)
        footer_frame = tk.Frame(sum_window, bg="#f8f9fa", pady=15)
        footer_frame.pack(side=tk.BOTTOM, fill=tk.X)

        close_btn = ttk.Button(footer_frame, text="Close Window", command=sum_window.destroy)
        close_btn.pack()

        # Content Frame (fill the remaining space)
        content_frame = tk.Frame(sum_window, bg="#f8f9fa", padx=20, pady=10)
        content_frame.pack(expand=True, fill='both')

        # Add scrollable text area
        txt = scrolledtext.ScrolledText(
            content_frame, 
            undo=True, 
            wrap=tk.WORD, 
            font=("Segoe UI", 11),
            bg="white",
            relief=tk.FLAT,
            padx=10,
            pady=10
        )
        
        # Apply some basic formatting for headers in the summary
        for line in summary_text.split('\n'):
            if line.endswith(':'):
                txt.insert(tk.END, line + '\n', 'header')
            else:
                txt.insert(tk.END, line + '\n')
        
        txt.tag_configure('header', font=("Segoe UI", 11, "bold"), foreground="#333")
        txt.configure(state='disabled') # Make it read-only
        txt.pack(expand=True, fill='both')


def launch_gui():
    root = tk.Tk()
    app = TrajectoryApp(root)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()