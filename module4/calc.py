import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

class Calculator:
    def add(self, x, y):
        """Add two numbers"""
        return x + y
    
    def subtract(self, x, y):
        """Subtract y from x"""
        return x - y
    
    def multiply(self, x, y):
        """Multiply two numbers"""
        return x * y
    
    def divide(self, x, y):
        """Divide x by y"""
        if y == 0:
            raise ValueError("Cannot divide by zero!")
        return x / y

class CalculatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Simple Calculator")
        self.calc = Calculator()
        
        # Create and set up the main frame
        self.frame = ttk.Frame(root, padding="10")
        self.frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create entry fields
        self.num1 = tk.StringVar()
        self.num2 = tk.StringVar()
        self.result = tk.StringVar()
        
        # Create and position widgets
        ttk.Label(self.frame, text="First Number:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(self.frame, textvariable=self.num1).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(self.frame, text="Second Number:").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(self.frame, textvariable=self.num2).grid(row=1, column=1, padx=5, pady=5)
        
        # Operation buttons
        ttk.Button(self.frame, text="Add", command=self.add).grid(row=2, column=0, padx=5, pady=5)
        ttk.Button(self.frame, text="Subtract", command=self.subtract).grid(row=2, column=1, padx=5, pady=5)
        ttk.Button(self.frame, text="Multiply", command=self.multiply).grid(row=3, column=0, padx=5, pady=5)
        ttk.Button(self.frame, text="Divide", command=self.divide).grid(row=3, column=1, padx=5, pady=5)
        
        # Result display
        ttk.Label(self.frame, text="Result:").grid(row=4, column=0, sticky=tk.W)
        ttk.Label(self.frame, textvariable=self.result).grid(row=4, column=1, sticky=tk.W)
    
    def get_numbers(self):
        """Get and validate input numbers"""
        try:
            num1 = float(self.num1.get())
            num2 = float(self.num2.get())
            return num1, num2
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers")
            return None, None
    
    def add(self):
        num1, num2 = self.get_numbers()
        if num1 is not None:
            self.result.set(self.calc.add(num1, num2))
    
    def subtract(self):
        num1, num2 = self.get_numbers()
        if num1 is not None:
            self.result.set(self.calc.subtract(num1, num2))
    
    def multiply(self):
        num1, num2 = self.get_numbers()
        if num1 is not None:
            self.result.set(self.calc.multiply(num1, num2))
    
    def divide(self):
        num1, num2 = self.get_numbers()
        if num1 is not None:
            try:
                self.result.set(self.calc.divide(num1, num2))
            except ValueError as e:
                messagebox.showerror("Error", str(e))

def main():
    root = tk.Tk()
    app = CalculatorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
