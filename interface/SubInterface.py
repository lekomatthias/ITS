import threading
import sys
from tkinter import Button, Toplevel, Text, Scrollbar, messagebox, VERTICAL, RIGHT, Y, END

from interface.TextRedirector import TextRedirector
from interface.Interrupt import Interrupt

class SubInterface:
    def __init__(self, function, name, master):
        self.function = function
        self.name = name
        self.master = master
        self.text_widget = None
        self.popup = None
        self.original_stdout = sys.stdout
        self.stop_event = threading.Event()
        self.thread = None
        self._build_interface()

    def _build_interface(self):
        Button(
            self.master,
            text=self.name,
            width=30,
            height=2,
            command=self._start_action
        ).pack(pady=3)

    def _start_action(self):
        self._open_output_popup()
        self.thread = threading.Thread(target=self._run_function, daemon=True)
        self.thread.start()

    def _open_output_popup(self):
        self.popup = Toplevel(self.master)
        self.popup.title(f"Saída de {self.name}")
        self.popup.protocol("WM_DELETE_WINDOW", self._on_popup_close)

        self.text_widget = Text(self.popup, wrap='word', width=100, height=20, state='normal')
        self.text_widget.pack(side='left', fill='both', expand=True)

        scrollbar = Scrollbar(self.popup, command=self.text_widget.yview, orient=VERTICAL)
        scrollbar.pack(side=RIGHT, fill=Y)

        self.text_widget.config(yscrollcommand=scrollbar.set)
        self.text_widget.insert(END, f"Executando {self.name}...\n")
        self.text_widget.config(state='disabled')

    def _on_popup_close(self):
        self.stop_event.set()
        sys.stdout = self.original_stdout
        if self.popup:
            self.popup.destroy()

    def _run_function(self):
        old_stdout = sys.stdout
        sys.stdout = TextRedirector(self.text_widget)
        try:
            wrapped_func = Interrupt(self.stop_event)(self.function)
            wrapped_func()
            print(f"\n{self.name} executado com sucesso.")
        except InterruptedError:
            messagebox.showerror(
                title="Execução Interrompida",
                message=f"A execução de '{self.name}' foi interrompida."
            )
        except Exception as e:
            print(f"\nErro ao executar {self.name}: {e}")
        finally:
            sys.stdout = old_stdout
