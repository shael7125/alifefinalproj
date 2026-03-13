## Overview

This repository modifies a simulator that randomly generates robots and allows the user to train and visulaize their fitness. The simulator is modified as follows:

1) Change fitness measure to instead evaluate how closely a robot positions its own center of mass to a target. The robot is rewarded for rapidly closing this distance. This is meant to simulate an immune cell's response to the presence of an antigen, where rapid movement is key to protect the body from infection. Robots are trained to move faster towards the antigen by measuring fitness as the negative of the distance between the robot's COM and the antigen.

2) Evolve a given robot using a parallel hill climber method. The original (parent) robot's fitness is evaluated. The parent is mutated to create a child. The child's fitness is then evaluated. These fitnesses are compared, and the robot with the higher fitness is kept for the next generation. This is repeated over 10 generations to iteratively "climb the hill."

Four situations were explored and documented via video.
1) no evolution, no learning: robot_0
2) train robot_0 to create robot_1: no evolution, learned
3) evolve robot_0 to create robot_2: evolution, no learning
4) train robot_2 to create robot_3: evolution and learning

The videos of the performance of these robots can be explored through visualizer.py. Interestingly, 10 generations of evolution seeemed to make the robot slower! This is most likely because my fitness algorithm is optimizing for the wrong function. Future development for this project would include workshopping this fitness algorithm to dynamically check where the target is, as this would perhaps allow the generation of robots that were able to respond to changing antigen/target location.
