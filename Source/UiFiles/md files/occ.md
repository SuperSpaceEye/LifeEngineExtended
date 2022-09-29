# Organism Construction Code
Organism Construction Code or OCC is a way to represent anatomy of organisms as a DNA like structure.
The OCC is an array of different instructions, executed one after another. The anatomy is constructed on a grid, with "cursor" being the main component. During the process of anatomy constructions, different elements are used:
### Construction elements:
-  "cursor" - a point in space depending on a position of which organism blocks are placed.
- "origin" - a changeable position that a cursor can return to if appropriate command is executed.
- "rotation cursor" - the base rotation of blocks that will be placed with.
- "group" - a sequence of random instruction with the size randomly chosen between [1, max_size].

The OCC can be viewed and edited in an "edit OCC" option in editor window.
### Rules of OCC edit window:
- To write a sequence of instructions, each instruction must be a valid instruction.
- Instruction must be finished with a ";" sign.
- The spaces, indentation or new lines do not affect the execution of OCC.
- You can make comment with a "/" sign.
- After execution, OCC must create at least 1 organism block.
- Instructions can be written in either full or shortened form.

### All OCC mutations:
- Append random - appends a group to the end of OCC sequence.
- Insert random - inserts a group to a random uniformly sampled position in OCC sequence.
- Change random - first gets a group, then randomly chooses the position inside the OCC sequence, then overwrites the part of existing OCC with instructions from group until either a group is fully written, or OCC reaches end.
- Delete random - will delete OCC instruction the size of group starting from random position, until either the number of instructions deleted = group size, or OCC reaches end.
- Swap random - will choose random position, randomly decide the direction of movement (left or right), randomly decide the distance instructions will be moved, randomly decide the distance from chosen position, then it will start to swap elements of group size, until either it successfully swapped the elements, or OCC ends.

### All OCC instructions:

Shift Instructions - Shifts cursor to direction, or if there is SetBlock instruction after it, sets the block above the cursor, not changing it's position.
- ShiftUp or SU
- ShiftUpLeft or SUL
- ShiftLeft or SL
- ShiftLeftDown or SLD
- ShiftDown or SD
- ShiftDownRight or SDR
- ShiftRight or SR
- ShiftRightUp or SRU


Apply rotation instructions - Sets the rotation of rotation cursor to the new rotation.
- ApplyRotationUp or ARU
- ApplyRotationLeft or ARL
- ApplyRotationDown or ARD
- ApplyRotationRight or ARR


Set rotation instructions - Sets the rotation of a block directly underneath the cursor to the new rotation. Will work only if there is already a block underneath the cursor.
- SetRotationUp or SRTU
- SetRotationLeft or SRL
- SetRotationDown or SRD
- SetRotationRight or SRR


- ResetToOrigin or RTO -  Resets the position of a cursor to the position of origin.
- SetOrigin or SO -  Sets the position of an origin to the position of a cursor.


Set block instructions - -  Sets the block on a grid to type.
- SetBlockMouth or SBM
- SetBlockProducer or SBP
- SetBlockMover or SBMV
- SetBlockKiller or SBK
- SetBlockArmor or SBA
- SetBlockEye or SBE


Set under block instructions -  Sets the block on a grid to mouth directly underneath of the cursor. Is not affected by shift instructions
- SetUnderBlockMouth or SUBM
- SetUnderBlockProducer or SUBP
- SetUnderBlockMover or SUBMV
- SetUnderBlockKiller or SURK
- SetUnderBlockArmor or SUBA
- SetUnderBlockEye or SUBE