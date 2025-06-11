import tkinter as tk

from interface.SubInterface import SubInterface


class Menu:
    def __init__(self, functions, title="Menu"):
        self.functions = functions
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("300x400")

    def build_interface(self):
        tk.Label(self.root, text="Menu de funções", font=('Arial', 16)).pack(pady=10)

        container = tk.Frame(self.root)
        container.pack(fill='both', expand=True)

        canvas = tk.Canvas(container, borderwidth=0)
        canvas.pack(side='left', fill='both', expand=True)

        scrollable_frame = tk.Frame(canvas)

        window_id = canvas.create_window((0, 0), window=scrollable_frame, anchor='nw')

        def on_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfig(window_id, width=canvas.winfo_width())

        scrollable_frame.bind("<Configure>", on_configure)

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
        canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))

        for item in self.functions:
            frame = tk.Frame(scrollable_frame)
            frame.pack(pady=10)
            SubInterface(
                function=item['function'],
                mode=item['mode'],
                name=item['name'],
                master=frame,
                list=item.get('list', [])
            )

        tk.Button(self.root, text="Sair", width=30, height=2, command=self.root.quit).pack(pady=5)

        self.root.mainloop()
