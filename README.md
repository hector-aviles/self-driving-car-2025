## Requirements:

* Ubuntu 20.04
* ROS Noetic (http://wiki.ros.org/noetic/Installation/Ubuntu)
* Webots 2022a (https://github.com/cyberbotics/webots/releases/download/R2022a/webots_2022a_amd64.deb)

## Installation

Note: It is assumed that Ubuntu OS, ROS platform and Webots simulator are already installed. 

* $ sudo pip3 install beepy
* $ sudo apt install ros-noetic-webots-ros
* $ sudo apt install ros-noetic-ros-numpy
* $ cd ~
* $ git clone https://github.com/hector-aviles/self-driving-car-2025
* $ cd self-driving-car-2025/catkin_ws
* $ catkin_make
* $ echo "source ~/self-driving-car-2025/catkin_ws/devel/setup.bash" >> ~/.bashrc
* $ source ~/.bashrc

## To collect driving samples, run:

$ roslaunch get_samples manual_control.launch world:=/right/world_00

$ rosrun utils logger.py

Change to world:=/left/world_00 to start the self-driving vehicle on the left lane. World numbers ranges from 00 to 15.

## To execute autonomous driving using a probabilistic logic state-action policy, just run:

$ roslaunch airj2024 two_ways.launch

## Videos
* A vehicle approaching in the opposite direction
  
https://github.com/hector-aviles/self-driving-car-2025/assets/67079858/0860044b-ea15-4925-a62f-529eb55a9948

* A vehicle travelling in the transverse direction

https://github.com/hector-aviles/self-driving-car-2025/assets/67079858/03caaf6b-ab8c-444d-a79c-f0df0bf894c9

## Contact

Héctor Avilés<br>
havilesa@upv.edu.mx <br>
Marco Negrete<br>
marco.negrete@ingenieria.unam.edu

