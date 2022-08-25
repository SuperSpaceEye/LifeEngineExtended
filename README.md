# LifeEngine Extended
C++ implementation of https://github.com/MaxRobinsonTheGreat/LifeEngine that extends the original version with new features.

**The program needs to be placed in path with only english letters.**

# Update 24.08.2022
The program right now is very broken. When I started optimizing and cleaning up code I found so many serious bugs that I don't understand how the program even functions and manages to evolve stuff.

# Important information:
- You should press enter after inputing text in line edits or it will not register.
- Setting -1 into max fps/sps/organisms line edits will disable the limit.
- To use keyboard actions you should click on simulation window first.
- When you use "Choose organism" mouse mode it will search for any organism in brush area and will stop when it finds one.
- You can use mouse actions in editor.
- Keyboard actions do not work in editor.

# Mouse actions:
- Hold left button to place/kill.
- Hold right button to delete/kill
- Hold scroll button to move the view.
- Scroll to zoom in/out.

# Keyboard button actions:
- "W" or "Up" - move view up.
- "A" or "Left" - move view left.
- "S" or "Down" - move view down.
- "D" or "Right" - move view right.
- "Shift" - hold to make movement using keys 2 times as fast.
- "R" - reset view.
- "M" - hide/show menu.
- "Space bar" - pause simulation,
- "." - pass one simulation tick.
- "numpad +" - zoom in.
- "numpad -" - zoom out.
- "1" - switch mouse mode to place/delete food.
- "2" - switch mouse mode to kill organisms.
- "3" - switch mouse mode to place/delete walls.
- "4" - switch mouse mode to place editor organisms.
- "5" - switch mouse mode to choose organisms from simulation.

# Known bugs:
- Saving and loading will not work correctly unless your path contains only english letters.
- Mouse movement tracking is imprecise.
