import tkinter as tk
from interface.SubInterface import SubInterface


class Menu:
    def __init__(self, functions, title="Menu"):
        self.functions = functions
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("350x450")

    def build_interface(self):
        tk.Label(self.root, text="Menu de funções", font=('Arial', 16)).pack(pady=20)

        for item in self.functions:
            frame = tk.Frame(self.root)
            frame.pack(pady=10)

            SubInterface(
                function=item['function'],
                mode=item['mode'],
                name=item['name'],
                master=frame,
                list=item.get('list', [])
            )

        tk.Button(
            self.root,
            text="Sair",
            width=30,
            height=2,
            command=self.root.quit
        ).pack(pady=20)

        self.root.mainloop()
