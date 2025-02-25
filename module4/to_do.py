import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

class TodoList:
    def __init__(self, root):
        self.root = root
        self.root.title("Todo List")
        
        # Create main frame
        self.frame = ttk.Frame(root, padding="10")
        self.frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create task entry
        self.task = tk.StringVar()
        ttk.Label(self.frame, text="Task:").grid(row=0, column=0, sticky=tk.W)
        self.task_entry = ttk.Entry(self.frame, textvariable=self.task, width=40)
        self.task_entry.grid(row=0, column=1, padx=5, pady=5)
        
        # Add task button
        ttk.Button(self.frame, text="Add Task", command=self.add_task).grid(row=0, column=2, padx=5, pady=5)
        
        # Create listbox for tasks
        self.task_listbox = tk.Listbox(self.frame, width=50, height=10)
        self.task_listbox.grid(row=1, column=0, columnspan=2, padx=5, pady=5)
        
        # Delete task button
        ttk.Button(self.frame, text="Delete Task", command=self.delete_task).grid(row=1, column=2, padx=5, pady=5)
        
    def add_task(self):
        task = self.task.get()
        if task:
            self.task_listbox.insert(tk.END, task)
            self.task.set("")  # Clear entry field
        else:
            messagebox.showwarning("Warning", "Please enter a task")
            
    def delete_task(self):
        try:
            selected = self.task_listbox.curselection()
            self.task_listbox.delete(selected)
        except:
            messagebox.showwarning("Warning", "Please select a task to delete")

def main():
    root = tk.Tk()
    app = TodoList(root)
    root.mainloop()

if __name__ == "__main__":
    main()
