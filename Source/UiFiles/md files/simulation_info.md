Simulation is processed in stages with each stage going one after another - eat food/produce food, apply damage, kill organisms, get observations, think decisions, make decisions, try produce children

The food values are not discrete. Instead, the food is a float value. Each producer can add to this value until it reaches max_food value. If the amount of food is less than food_threshold, then organisms cannot eat this food, and it will not be displayed on screen. If organism try to eat more food than there exists in world block, then it will eat only the available food amount.

There are two modes of organism's movement - discrete and continuous:
- With discrete movement organisms move 1 world block in one tick in any of 4 directions. 
- With continuous mode the organisms have a velocity. Each tick to the organism's velocity a force is applied (If it has mover) which is calculated by brain (otherwise a random force is applied). The organism will maintain this velocity and move through space until it meets an obstacle.
