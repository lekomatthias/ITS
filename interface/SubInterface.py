from tkinter import messagebox, ttk, Entry, Button, Label


class SubInterface:
    def __init__(self, function, mode, name, master, list=None):
        self.function = function
        self.mode = mode  # 'button', 'prompt' ou 'select'
        self.name = name
        self.list = list if list is not None else []
        self.master = master

        self.entry = None
        self.combobox = None

        self._build_interface()

    def _build_interface(self):
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
            command=self.execute
        ).pack(pady=3)

    def execute(self):
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
                raise ValueError(f"Modo desconhecido: {self.mode}")

            messagebox.showinfo("Sucesso", f"{self.name} executado com sucesso.")

        except Exception as e:
            messagebox.showerror("Erro", f"Ocorreu um erro: {e}")
