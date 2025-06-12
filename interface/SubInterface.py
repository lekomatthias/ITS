import threading
from tkinter import Button, Toplevel, Text, Scrollbar, VERTICAL, RIGHT, Y, END

class SubInterface:
    def __init__(self, function, name, master):
        self.function = function
        self.name = name
        self.master = master
        self.text_widget = None
        self.popup = None
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

    def _run_function(self):
        raise NotImplementedError("Implementar em subclasses")
