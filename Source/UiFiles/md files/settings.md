# Settings
- **“Float number precision”** - A decorative parameter that control how many zeros after the decimal point of floats will be displayed in labels
- **“Wait for engine to stop to render”** - If enabled, will send an engine a signal to stop before rendering simulation grid.
- **“Simplified rendering”** - If enabled, will not render eyes. Will be removed soon.
- **“Really stop render”**- To render an image, the program first calculates what cells will be seen by user, and then it copies them to the secondary grid containing only cell type and rotation, from which the image is constructed. If disabled, will parse the whole grid when “Stop render” button is pressed, which will allow to move and scale the view. If enabled, will not parse the grid or construct an image when “Stop render” is pressed.
