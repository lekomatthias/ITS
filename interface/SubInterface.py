import threading
import sys
import tkinter as tk
from tkinter import ttk, Entry, Button, Label, Toplevel, Text, Scrollbar, VERTICAL, RIGHT, Y, END

class TextRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, string):
        if string:
            self.text_widget.after(0, self._write_to_widget, string)

    def _write_to_widget(self, string):
        self.text_widget.config(state='normal')
        self.text_widget.insert(END, string)
        self.text_widget.see(END)
        self.text_widget.config(state='disabled')

    def flush(self):
        pass

class SubInterface:
    def __init__(self, function, mode, name, master, list=None):
        self.function = function
        self.mode = mode  # 'button', 'prompt', 'select', 'open'
        self.name = name
        self.list = list or []
        self.master = master

        self.entry = None
        self.combobox = None
        self.text_widget = None
        self.popup = None

        self._build_interface()

    def _build_interface(self):
        # Label explicativo, se necessário
        if self.mode == 'prompt':
            Label(self.master, text=f"Entrada para {self.name}:").pack(anchor="w")
            self.entry = Entry(self.master, width=40)
            self.entry.pack(pady=3)
        elif self.mode == 'select':
            Label(self.master, text=f"Selecione o modo para {self.name}:").pack(anchor="w")
            self.combobox = ttk.Combobox(
                self.master,
                values=self.list,
                state="readonly",
                width=37
            )
            if self.list:
                self.combobox.current(0)
            self.combobox.pack(pady=3)

        Button(
            self.master,
            text=self.name,
            width=30,
            height=2,
            command=self._start_action
        ).pack(pady=3)

    def _start_action(self):
        self._open_output_popup()
        if self.mode == 'open':
            self._run_tkinter_loop()
        else:
            thread = threading.Thread(target=self._run_function, daemon=True)
            thread.start()

    def _open_output_popup(self):
        self.popup = Toplevel(self.master)
        self.popup.title(f"Saída de {self.name}")

        self.text_widget = Text(self.popup, wrap='word', width=100, height=20, state='normal')
        self.text_widget.pack(side='left', fill='both', expand=True)

        scrollbar = Scrollbar(self.popup, command=self.text_widget.yview, orient=VERTICAL)
        scrollbar.pack(side=RIGHT, fill=Y)

        self.text_widget.config(yscrollcommand=scrollbar.set)
        self.text_widget.insert(END, f"Executando {self.name}...\n")
        self.text_widget.config(state='disabled')

    def _run_tkinter_loop(self):
        old_stdout = sys.stdout
        sys.stdout = TextRedirector(self.text_widget)
        try:
            self.function(root=tk.Toplevel(self.master))
            print(f"\n{self.name} executado com sucesso.")
        except Exception as e:
            print(f"\nErro ao executar {self.name}: {e}")
        finally:
            sys.stdout = old_stdout

    def _run_function(self):
        old_stdout = sys.stdout
        sys.stdout = TextRedirector(self.text_widget)
        try:
            if self.mode == 'button':
                self.function()
            elif self.mode == 'prompt':
                user_input = self.entry.get()
                self.function(mode=user_input)
            elif self.mode == 'select':
                selection = self.combobox.get()
                self.function(mode=selection)
            else:
                print(f"Modo desconhecido: {self.mode}")
            print(f"\n{self.name} executado com sucesso.")
        except Exception as e:
            print(f"\nErro ao executar {self.name}: {e}")
        finally:
            sys.stdout = old_stdout
