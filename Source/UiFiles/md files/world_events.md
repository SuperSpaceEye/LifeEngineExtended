# World Events
World events are events that are executed after simulation tick.
Right now there is two types of events: "Conditional", "Change Value".
World events are divided into rows and nodes. First executed rows from the bottom, then inside rows  nodes are executed left to right.

World events have two settings - "Update World Events every n ticks" and "Collect Info every n ticks".
- **"Update World Events every n ticks"** - Although world events are pretty lightweight in regard to performance, they are not free. So I made this parameter to control when nodes are updated.
- **"Collect Info every n ticks"** - The conditional node needs info to make decision.
If the value of this parameter is too large the conditional node will use an outdated data.
However, if the value is too small it will hurt the performance, as gathering data of simulation is not free.
Be aware, that because world events and statistics share the same location of info, when UI part sends signal to update data, it will also update data for the world events.

Every node has "Update every n tick". It works the same as "Update World Events every n ticks". When last execution time exceeds this parameter, the node will execute.

Each world events branch also has "Repeat branch". If event branch reaches the end and parameter was toggled, the execution will begin from the start, else it will just stop.

World events in "World Events Editor" and simulation are separate, so changes in the editor will not affect world events in simulation, unless you apply them.

### Conditional Event
When event branch reaches conditional node, it will continuously check if the statement is true. If it is, the execution of the next node will begin, otherwise it will repeat the check.

### Change Value
Change value node allows for World Events to actually influence the simulation.
With this node you can change "some" simulation and block parameters.
This node has several modes with how it can change selected value.
- **"Linear"** - Will change the value to target across time.
The parameter "Time horizon" controls for how long the value is changed.
During the execution of this mode, any changes to the value will not be applied.
If two Linear nodes started executing at the same time, and have the same time horizon, the one in higher branch will set final target value, otherwise the one finishing last will set the value.
- **"Step"** - Will change the value to target value upon reaching.

The modes below were added by omgdev. All nodes are executed upon reaching.
- **"Increase by"** - Will increase chosen variable by target amount.
- **"Decrease by"** - Will decrease chosen variable by target amount.
- **"Multiply by"** - Will multiply chosen variable by target amount.
- **"Divide by"** - Will divide chosen variable by target amount.

### Running World Events
After creating World Events click "Apply events". 
World events will not be applied if there are nodes without set value.
If applying was successful, the world events will start execution.

While the world events are running, you can't change some values.
To change the values, pause the simulation or pause execution of world events in the tab "Current World Events Viewer" of world events window.

If simulation resets, world events will also automatically reset and start from the beginning. If execution of world events is stopped or the simulation resets, the simulation settings will be set to the state they were before the execution of world events started, unless you stop them with "Stop Events No Setting Reset".

You can pause/resume execution of world events with buttons "Pause events"/"Resume events". These buttons will not reset world events.
If world events are already applied and were stopped, you can use "Start Events" button to re-enable already applied world events.
