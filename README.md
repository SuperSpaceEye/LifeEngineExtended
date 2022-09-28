# LifeEngine Extended
C++ implementation of https://github.com/MaxRobinsonTheGreat/LifeEngine that extends the original version with new features.

**The program needs to be placed in path with only english letters.**

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
- "0" - switch mouse mode to no action.
- "1" - switch mouse mode to place/delete food.
- "2" - switch mouse mode to kill organisms.
- "3" - switch mouse mode to place/delete walls.
- "4" - switch mouse mode to place editor organisms.
- "5" - switch mouse mode to choose organisms from simulation.
- "F11" - enable/disable fullscreen.

# Known bugs:
- Saving and loading will not work correctly unless your path contains only english letters.
- Mouse movement tracking is imprecise.
- Renderer imprecisely calculates textures for small num of pixels per block. So if you want to have nice, correct images/videos when using recorder, set "Number of pixels per world block" > 10
- The program will crash sometimes.
