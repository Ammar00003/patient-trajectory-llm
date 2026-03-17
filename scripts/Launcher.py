from pathlib import Path
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
from utils.visualization_utils import build_patient_timeline_figure

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAJECTORY_PATH = PROJECT_ROOT / "data" / "patient_trajectories.csv"


class TrajectoryApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Patient Trajectory Viewer")
        self.root.geometry("500x280")
        self.root.configure(bg="#f0f0f0")

        self.subject_ids = self.load_subject_ids()
        self.setup_styles()
        self.create_widgets()

    def load_subject_ids(self):
        """Load unique subject IDs from the trajectory data."""
        try:
            if TRAJECTORY_PATH.exists():
                df = pd.read_csv(TRAJECTORY_PATH)
                if "subject_id" in df.columns:
                    ids = sorted(df["subject_id"].unique().astype(str).tolist())
                    return ids
        except Exception as e:
            print(f"Warning: Could not load subject IDs for dropdown: {e}")
        return []

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

        self.status_var.set("Loading data...")
        self.root.update_idletasks()

        try:
            if not TRAJECTORY_PATH.exists():
                messagebox.showerror("Error", f"Data file not found at:\n{TRAJECTORY_PATH}")
                return

            df = pd.read_csv(TRAJECTORY_PATH)
            patient_df = df[df["subject_id"].astype(str) == subject_id].copy()

            if patient_df.empty:
                messagebox.showerror("No Data", f"No records found for Subject ID: {subject_id}")
                return

            self.status_var.set("Generating timeline...")
            self.root.update_idletasks()

            fig = build_patient_timeline_figure(patient_df, subject_id)
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


def launch_gui():
    root = tk.Tk()
    app = TrajectoryApp(root)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()