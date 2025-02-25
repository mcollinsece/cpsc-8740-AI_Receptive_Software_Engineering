# Module 4 - GUI Applications with Tkinter

This directory contains three simple GUI applications built using Python's Tkinter library.

## Applications

### Calculator (`calc.py`)
A basic calculator application that performs arithmetic operations:
- Supports addition, subtraction, multiplication, and division
- Input validation for numbers
- Error handling for division by zero
- Clean and intuitive interface with two number inputs and operation buttons

### Todo List (`to_do.py`)
A simple task management application:
- Add new tasks with a text input field
- Delete selected tasks
- Tasks displayed in a scrollable listbox
- Input validation to prevent empty tasks
- Warning messages for invalid actions

### Tic Tac Toe (`tic_tac_toe.py`)
A classic two-player Tic Tac Toe game:
- Players take turns placing X's and O's
- Automatic win detection for rows, columns, and diagonals
- Game status display showing current player's turn
- New Game button to reset the board
- Draw detection when the board is full

## Requirements
- Python 3.x
- Tkinter (usually comes with Python installation)

## Running the Applications
To run any of the applications, use Python from the command line:

```bash
python calc.py
python to_do.py
python tic_tac_toe.py
```

## Common Features
All applications share these common design elements:
- Built using Tkinter and ttk for modern widget styling
- Error handling and user feedback through message boxes
- Grid-based layouts for consistent UI organization
- Object-oriented design with clear class structures  