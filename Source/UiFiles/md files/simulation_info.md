Simulation is processed in stages with each stage going one after another - eat food/produce food, apply damage, kill organisms, get observations, think decisions, make decisions, try produce children

The food values are not discrete. Instead, the food is a float value. Each producer can add to this value until it reaches max_food value. If the amount of food is less than food_threshold, then organisms cannot eat this food, and it will not be displayed on screen. If organism try to eat more food than there exists in world block, then it will eat only the available food amount.

