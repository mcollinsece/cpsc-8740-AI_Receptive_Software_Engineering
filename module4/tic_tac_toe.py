import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

class TicTacToe:
    def __init__(self, root):
        self.root = root
        self.root.title("Tic Tac Toe")
        
        # Game state
        self.current_player = "X"
        self.board = [""] * 9
        
        # Create main frame
        self.frame = ttk.Frame(root, padding="10")
        self.frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create game board buttons
        self.buttons = []
        for i in range(3):
            for j in range(3):
                button = ttk.Button(self.frame, text="", width=5,
                                  command=lambda row=i, col=j: self.make_move(row, col))
                button.grid(row=i, column=j, padx=2, pady=2)
                self.buttons.append(button)
        
        # Status label
        self.status = tk.StringVar()
        self.status.set(f"Player {self.current_player}'s turn")
        ttk.Label(self.frame, textvariable=self.status).grid(row=3, column=0, columnspan=3)
        
        # Reset button
        ttk.Button(self.frame, text="New Game", command=self.reset_game).grid(row=4, column=0, columnspan=3, pady=10)
    
    def make_move(self, row, col):
        index = 3 * row + col
        
        if self.board[index] == "":
            self.board[index] = self.current_player
            self.buttons[index].configure(text=self.current_player)
            
            if self.check_winner():
                messagebox.showinfo("Game Over", f"Player {self.current_player} wins!")
                self.reset_game()
            elif "" not in self.board:
                messagebox.showinfo("Game Over", "It's a tie!")
                self.reset_game()
            else:
                self.current_player = "O" if self.current_player == "X" else "X"
                self.status.set(f"Player {self.current_player}'s turn")
    
    def check_winner(self):
        # Check rows, columns and diagonals
        win_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]  # Diagonals
        ]
        
        for combo in win_combinations:
            if (self.board[combo[0]] == self.board[combo[1]] == self.board[combo[2]] != ""):
                return True
        return False
    
    def reset_game(self):
        self.current_player = "X"
        self.board = [""] * 9
        for button in self.buttons:
            button.configure(text="")
        self.status.set(f"Player {self.current_player}'s turn")

def main():
    root = tk.Tk()
    app = TicTacToe(root)
    root.mainloop()

if __name__ == "__main__":
    main()
