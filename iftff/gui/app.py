"""A small Tkinter GUI over the three iftff commands (train / minimize / md).

It is a thin front-end: each tab collects the *same* options you would pass on
the command line, then runs the real command (`python -m iftff.train` or
`python -m iftff.run`) as a subprocess and streams its output into the log at
the bottom. The exact command is printed before it runs, so the GUI also
doubles as a way to learn the terminal syntax.

Launched by `iftff gui`. Tkinter ships with Python's standard library, so no
extra dependency is needed.
"""
import os
import queue
import re
import subprocess
import sys
import threading

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, scrolledtext
    _TK_ERROR = None
except Exception as exc:  # pragma: no cover - only on a Python built without Tk
    tk = None
    _TK_ERROR = exc

# Modern theme (Sun Valley / Windows-11 look). Optional: installed via the
# `gui` extra. Without it the GUI refuses to start and prints how to install it,
# rather than falling back to Tk's dated default theme.
try:
    import sv_ttk
    _SVTTK_ERROR = None
except Exception as exc:  # pragma: no cover
    sv_ttk = None
    _SVTTK_ERROR = exc

# Default force field for md/minimize: the bundled reg=0.01 reparametrised force
# field (the recommended production force field), matching the CLI default.
_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
_DEFAULT_XML = os.path.join(_DATA_DIR, 'AmberReparametrised_lambda001.xml')


# --------------------------------------------------------------------------
# Field specification. Each field mirrors one command-line flag.
#   kind: 'file' | 'dir' | 'save' | 'choice' | 'flag' | 'int' | 'float' | 'str'
#   adv:  True  -> shown under the collapsible "Advanced options" section
# --------------------------------------------------------------------------
class F:
    def __init__(self, flag, label, kind, default='', adv=False, required=False, choices=None):
        self.flag = flag            # e.g. '--n-epochs'  (None => positional/none)
        self.label = label
        self.kind = kind
        self.default = default
        self.adv = adv
        self.required = required
        self.choices = choices or []


# ---- MD / minimize share the same driver (iftff/run.py) ------------------
_SIM_COMMON = [
    F('--pdb', 'Input structure (PDB)', 'file', required=True),
    F('--xml', 'Force field (XML)', 'file', default=_DEFAULT_XML, required=True),
    F('--output-dir', 'Output folder', 'dir', default='./simulation_output'),
]
_SIM_ADV = [
    F('--device', 'Device', 'choice', default='auto', adv=True, choices=['auto', 'cuda', 'cpu']),
    F('--min-tolerance', 'Min. tolerance (kJ/mol/nm)', 'float', default='1.0', adv=True),
    F('--min-max-iterations', 'Min. max iterations', 'int', default='1000', adv=True),
]

MINIMIZE_FIELDS = _SIM_COMMON + _SIM_ADV

MD_FIELDS = _SIM_COMMON + [
    F('--n-steps', 'Number of MD steps', 'int', default='250000'),
    F('--temperature', 'Temperature (K)', 'float', default='300.0'),
    F('--traj-interval', 'Steps between saved frames', 'int', default='10000'),
    F('--friction', 'Friction (1/ps)', 'float', default='1.0', adv=True),
    F('--timestep', 'Timestep (ps)', 'float', default='0.002', adv=True),
] + _SIM_ADV

# ---- Training (iftff/train.py) -------------------------------------------
TRAIN_FIELDS = [
    F('--pdb-dir', 'Structures folder (PDBs)', 'dir', required=True),
    F('--output-xml', 'Output force field (XML)', 'save', default='./trained.xml'),
    F('--n-epochs', 'Epochs', 'int', default='10'),
    F('--reg-lambda', 'Regularization strength (toward stock AMBER)', 'float', default='0.01'),
    F('--lr', 'Learning rate', 'float', default='0.001'),
    F('--device', 'Device', 'choice', default='auto', adv=True, choices=['auto', 'cuda', 'cpu']),
    F('--checkpoint-dir', 'Checkpoint folder', 'dir', default='checkpoints', adv=True),
    F('--gradient-clip-norm', 'Gradient clip norm', 'float', default='1.0', adv=True),
    F('--save-interval', 'Save every N epochs', 'int', default='10', adv=True),
    F('--inner-min-tolerance', 'Inner min. tolerance', 'float', default='0.001', adv=True),
    F('--inner-min-max-iterations', 'Inner min. max iters', 'int', default='100000', adv=True),
    F('--cg-max-iter', 'CG max iterations', 'int', default='200', adv=True),
    F('--cg-damping', 'CG damping (mu)', 'float', default='1.0', adv=True),
    F('--cg-fallback-damping', 'CG fallback damping', 'float', default='10.0', adv=True),
    F('--cg-tolerance', 'CG tolerance', 'float', default='1e-8', adv=True),
]

# module, fields, and any always-on extra flags for each tab
TABS = [
    ('Train', 'iftff.train', TRAIN_FIELDS, []),
    ('Minimize', 'iftff.run', MINIMIZE_FIELDS, ['--minimize-only']),
    ('MD', 'iftff.run', MD_FIELDS, []),
]


def parse_run_output(task, text):
    """Pull the headline results out of a finished run's output.

    Returns a list of (label, value) pairs to show in the Result panel. The log
    still holds everything; this is just the part worth reading at a glance.
    """
    def find(pat):
        m = re.search(pat, text)
        return m.group(1).strip() if m else None

    rows = []
    if task in ('Minimize', 'MD'):
        rmin = find(r'Minimized: RMSD=([\d.]+) nm')
        emin = find(r'Minimized: RMSD=[\d.]+ nm, PE=(-?[\d.]+)')
        if rmin:
            rows.append(('Minimized RMSD (vs input)', f'{rmin} nm'))
        if emin:
            rows.append(('Minimized energy', f'{emin} kJ/mol'))
        if task == 'MD':
            rf = find(r'Final: RMSD=([\d.]+) nm')
            drift = find(r'Drift=([\d.]+) nm')
            ef = find(r'Final: RMSD=[\d.]+ nm, Drift=[\d.]+ nm, PE=(-?[\d.]+)')
            frames = find(r'Frames: (\d+)')
            if rf:
                rows.append(('Final RMSD (vs experimental)', f'{rf} nm'))
            if drift:
                rows.append(('Drift (final vs minimized)', f'{drift} nm'))
            if ef:
                rows.append(('Final energy', f'{ef} kJ/mol'))
            if frames:
                rows.append(('Trajectory frames', frames))
        for label, pat in [('Output folder', r'Output: (.+)'),
                           ('Report file', r'Report: (.+)'),
                           ('Trajectory file', r'Trajectory: (.+)')]:
            v = find(pat)
            if v:
                rows.append((label, v))
    elif task == 'Train':
        losses = re.findall(r'loss=([\d.]+) nm', text)
        if losses:
            rows.append(('Final training loss (RMSD)', f'{losses[-1]} nm'))
        nep = find(r'Epochs:\s+(\d+)')
        if nep:
            rows.append(('Epochs', nep))
        for label, pat in [('Trained force field', r'Final XML:\s+(.+)'),
                           ('Per-protein losses', r'Per-protein:\s+(.+)')]:
            v = find(pat)
            if v:
                rows.append((label, v))
    return rows


class TabForm(ttk.Frame):
    """One tab: a scrollable form of fields + a Run button, sharing the app log."""

    def __init__(self, master, app, name, module, fields, extra_flags):
        super().__init__(master, padding=12)
        self.app = app
        self.name = name
        self.module = module
        self.fields = fields
        self.extra_flags = extra_flags
        self.vars = {}
        self._build()

    def _build(self):
        self.columnconfigure(1, weight=1)
        row = 0
        adv_fields = [f for f in self.fields if f.adv]
        for f in self.fields:
            if f.adv:
                continue
            row = self._add_field(self, f, row)

        # Advanced options in a collapsible labelled frame.
        if adv_fields:
            self.adv_visible = tk.BooleanVar(value=False)
            self.adv_btn = ttk.Checkbutton(
                self, text='Advanced options', style='Toolbutton',
                variable=self.adv_visible, command=self._toggle_adv)
            self.adv_btn.grid(row=row, column=0, columnspan=3, sticky='w', pady=(10, 2))
            row += 1
            self.adv_frame = ttk.Frame(self)
            self.adv_frame.grid(row=row, column=0, columnspan=3, sticky='ew')
            self.adv_frame.columnconfigure(1, weight=1)
            self.adv_frame.grid_remove()
            arow = 0
            for f in adv_fields:
                arow = self._add_field(self.adv_frame, f, arow)
            row += 1

        self.run_btn = ttk.Button(self, text='Run ' + self.name, command=self.run,
                                  style='Accent.TButton')
        self.run_btn.grid(row=row, column=0, columnspan=3, sticky='ew', pady=(14, 0))

    def _add_field(self, parent, f, row):
        ttk.Label(parent, text=f.label + (' *' if f.required else '')).grid(
            row=row, column=0, sticky='w', padx=(0, 8), pady=3)
        if f.kind == 'flag':
            var = tk.BooleanVar(value=bool(f.default))
            ttk.Checkbutton(parent, variable=var).grid(row=row, column=1, sticky='w', pady=3)
        elif f.kind == 'choice':
            var = tk.StringVar(value=f.default)
            ttk.Combobox(parent, textvariable=var, values=f.choices, state='readonly',
                         width=12).grid(row=row, column=1, sticky='w', pady=3)
        else:
            var = tk.StringVar(value=f.default)
            ttk.Entry(parent, textvariable=var).grid(row=row, column=1, sticky='ew', pady=3)
            if f.kind in ('file', 'dir', 'save'):
                ttk.Button(parent, text='Browse…', width=9,
                           command=lambda ff=f, v=var: self._browse(ff, v)).grid(
                    row=row, column=2, sticky='w', padx=(6, 0), pady=3)
        self.vars[f.flag] = var
        return row + 1

    def _browse(self, f, var):
        if f.kind == 'dir':
            path = filedialog.askdirectory(title=f.label)
        elif f.kind == 'save':
            path = filedialog.asksaveasfilename(title=f.label, defaultextension='.xml',
                                                filetypes=[('XML', '*.xml'), ('All', '*')])
        else:
            path = filedialog.askopenfilename(title=f.label,
                                              filetypes=[('PDB', '*.pdb'), ('XML', '*.xml'), ('All', '*')])
        if path:
            var.set(path)

    def _toggle_adv(self):
        if self.adv_visible.get():
            self.adv_frame.grid()
        else:
            self.adv_frame.grid_remove()

    def build_argv(self):
        """Turn the form into a CLI argument list, validating as we go."""
        argv = list(self.extra_flags)
        for f in self.fields:
            var = self.vars[f.flag]
            if f.kind == 'flag':
                if var.get():
                    argv.append(f.flag)
                continue
            val = str(var.get()).strip()
            if not val:
                if f.required:
                    raise ValueError(f'"{f.label}" is required.')
                continue
            if f.kind == 'int':
                int(val)                      # raises ValueError with a clear message
            elif f.kind == 'float':
                float(val)
            argv += [f.flag, val]
        return argv

    def run(self):
        try:
            argv = self.build_argv()
        except ValueError as e:
            self.app.log(f'\n[input error] {e}\n')
            return
        self.app.start_run(self.name, self.module, argv)


class App:
    def __init__(self, root):
        self.root = root
        self.q = queue.Queue()
        self.proc = None
        self._task = None          # name of the running task
        self._out_lines = []       # accumulated output of the current run
        root.title('iftff')
        root.geometry('780x680')
        root.minsize(680, 560)

        # A small accent palette on top of the sv-ttk light theme.
        self.ACCENT = '#0067c0'     # primary blue
        self.SUCCESS = '#107c10'    # green
        self.ERROR = '#b00020'      # red
        style = ttk.Style(root)
        style.configure('Value.TLabel', foreground=self.ACCENT,
                        font=('TkDefaultFont', 11, 'bold'))
        style.configure('Result.TLabelframe.Label', foreground=self.ACCENT,
                        font=('TkDefaultFont', 10, 'bold'))
        style.configure('Log.TLabelframe.Label', foreground='#9a9a9a')

        outer = ttk.Frame(root, padding=10)
        outer.pack(fill='both', expand=True)

        nb = ttk.Notebook(outer)
        nb.pack(fill='x')
        self.tabs = []
        for name, module, fields, extra in TABS:
            tab = TabForm(nb, self, name, module, fields, extra)
            nb.add(tab, text=name)
            self.tabs.append(tab)

        bar = ttk.Frame(outer)
        bar.pack(fill='x', pady=(10, 2))
        self.status = ttk.Label(bar, text='Ready.')
        self.status.pack(side='left')
        self.stop_btn = ttk.Button(bar, text='Stop', command=self.stop, state='disabled')
        self.stop_btn.pack(side='right')

        # ---- Result panel (the thing to read) --------------------------------
        self.result_frame = ttk.LabelFrame(outer, text='Result', padding=(14, 10),
                                            style='Result.TLabelframe')
        self.result_frame.pack(fill='x', pady=(4, 8))
        self.result_frame.columnconfigure(1, weight=1)
        self.result_body = ttk.Frame(self.result_frame)
        self.result_body.pack(fill='x')
        self.result_body.columnconfigure(1, weight=1)
        self._result_message('Fill a tab and press Run. The result appears here; '
                              "the full output is in the log (click 'Show log').")

        # ---- Log (collapsible; hidden by default) ----------------------------
        self.log_visible = tk.BooleanVar(value=False)
        self.log_toggle = ttk.Checkbutton(
            outer, style='Toolbutton', variable=self.log_visible,
            command=self._toggle_log, text='▸ Show log')
        self.log_toggle.pack(fill='x', pady=(2, 0))

        self.log_frame = ttk.LabelFrame(outer, text='Log', padding=6, style='Log.TLabelframe')
        # ScrolledText is a classic Tk widget (not themed by sv-ttk), so match it
        # to the light surface by hand for a consistent look.
        self.text = scrolledtext.ScrolledText(
            self.log_frame, height=9, wrap='word', font=('TkFixedFont', 9),
            relief='flat', borderwidth=6, bg='#fbfbfb', fg='#555',
            insertbackground='#555', highlightthickness=1, highlightbackground='#d7d7d7')
        self.text.pack(fill='both', expand=True)
        self.text.configure(state='disabled')
        ttk.Button(self.log_frame, text='Clear log', command=self.clear_log).pack(
            anchor='e', pady=(4, 0))
        # log_frame stays unpacked (collapsed) until the user shows it

        self.root.after(100, self._poll)

    # ---- collapsible log ---------------------------------------------------
    def _toggle_log(self):
        if self.log_visible.get():
            self.log_frame.pack(fill='both', expand=True)
            self.log_toggle.config(text='▾ Hide log')
            self.root.geometry('780x860')
        else:
            self.log_frame.pack_forget()
            self.log_toggle.config(text='▸ Show log')
            self.root.geometry('780x680')

    def _ensure_log_shown(self):
        if not self.log_visible.get():
            self.log_visible.set(True)
            self._toggle_log()

    # ---- result panel ------------------------------------------------------
    def _clear_result(self):
        for w in self.result_body.winfo_children():
            w.destroy()

    def _result_message(self, msg, color=None):
        self._clear_result()
        lbl = ttk.Label(self.result_body, text=msg, wraplength=680, justify='left')
        if color:
            lbl.configure(foreground=color)
        lbl.grid(row=0, column=0, columnspan=2, sticky='w')

    def _result_rows(self, rows):
        self._clear_result()
        for i, (label, value) in enumerate(rows):
            ttk.Label(self.result_body, text=label).grid(
                row=i, column=0, sticky='w', padx=(0, 16), pady=2)
            ttk.Label(self.result_body, text=value, style='Value.TLabel').grid(
                row=i, column=1, sticky='w', pady=2)

    def _show_result(self, code):
        if code != 0:
            self._ensure_log_shown()
            self._result_message(
                'The run did not finish cleanly. See the log for details.',
                color='#b00020')
            return
        rows = parse_run_output(self._task, ''.join(self._out_lines))
        if rows:
            self._result_rows(rows)
        else:
            self._result_message('Done. (No summary to show — see the log below.)')

    # ---- logging -----------------------------------------------------------
    def log(self, msg):
        self.text.configure(state='normal')
        self.text.insert('end', msg)
        self.text.see('end')
        self.text.configure(state='disabled')

    def clear_log(self):
        self.text.configure(state='normal')
        self.text.delete('1.0', 'end')
        self.text.configure(state='disabled')

    # ---- running the subprocess -------------------------------------------
    def start_run(self, task, module, argv):
        if self.proc is not None:
            self.log('\n[busy] a run is already in progress - press Stop first.\n')
            return
        self._task = task
        self._out_lines = []
        cmd = [sys.executable, '-u', '-m', module] + argv
        self.log('\n$ ' + ' '.join(cmd) + '\n')
        self.status.config(text=f'Running {task}…', foreground=self.ACCENT)
        self._result_message(f'Running {task}…')
        self.stop_btn.config(state='normal')
        self._set_run_buttons('disabled')

        def worker():
            try:
                self.proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1)
                for line in self.proc.stdout:
                    self.q.put(line)
                self.proc.wait()
                self.q.put(('__DONE__', self.proc.returncode))
            except Exception as e:
                self.q.put(f'[error] {e}\n')
                self.q.put(('__DONE__', -1))

        threading.Thread(target=worker, daemon=True).start()

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            self.log('\n[stopped by user]\n')

    def _poll(self):
        try:
            while True:
                item = self.q.get_nowait()
                if isinstance(item, tuple) and item and item[0] == '__DONE__':
                    code = item[1]
                    self.proc = None
                    self.status.config(
                        text='Done.' if code == 0 else 'Finished with errors.',
                        foreground=self.SUCCESS if code == 0 else self.ERROR)
                    self.stop_btn.config(state='disabled')
                    self._set_run_buttons('normal')
                    self._show_result(code)
                else:
                    self._out_lines.append(item)
                    self.log(item)
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _set_run_buttons(self, state):
        for t in self.tabs:
            t.run_btn.config(state=state)


def main(argv=None):
    missing = []
    if tk is None:
        missing.append('tkinter')
    if sv_ttk is None:
        missing.append('sv-ttk')
    if missing:
        sys.stderr.write(
            'GUI dependencies not installed (' + ', '.join(missing) + ').\n'
            'If you want a graphical interface, install the requirements with:\n\n'
            '    pip install "ift-forcefield-optimization[gui]"\n'
            '    # or, from the source tree:  pip install -e ".[gui]"\n')
        if 'tkinter' in missing:
            sys.stderr.write(
                '\nTkinter is a system component (not a pip package), so if it is\n'
                'still missing after the command above, also install it with one of:\n'
                '    conda install tk\n'
                '    sudo apt-get install python3-tk      # Debian/Ubuntu\n')
        sys.stderr.write(
            '\nOr just use the command line: iftff train/minimize/md --help\n')
        return 1
    root = tk.Tk()
    sv_ttk.set_theme('light')          # modern flat theme; 'dark' also available
    App(root)
    root.mainloop()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
