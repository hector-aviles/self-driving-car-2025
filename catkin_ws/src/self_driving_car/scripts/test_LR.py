#!/usr/bin/env python3
"""
This node implements the action policy of the self-driving car
obtained from modeling and training MDPs using MDP-ProbLog.
This node provides several functions to check if there are
other vehicles around the car and to execute the three
different behaviors: cruise, follow and change_lane. 
"""
import rospy
from std_msgs.msg import Float64MultiArray, Empty, Bool, String, Float64
from rosgraph_msgs.msg import Clock 
import sys
import pandas as pd
import pickle
import time
import numpy as np
import os

def callback_free_N(msg):
    global free_N
    free_N = msg.data

def callback_free_NW(msg):
    global free_NW
    free_NW = msg.data

def callback_free_W(msg):
    global free_W
    free_W = msg.data
    
def callback_free_SW(msg):
    global free_SW
    free_SW = msg.data
    
def callback_free_NE(msg):
    global free_NE
    free_NE = msg.data    

def callback_free_E(msg):
    global free_E
    free_E = msg.data    

def callback_free_SE(msg):
    global free_SE
    free_SE = msg.data    
    
def callback_success(msg):
    global success
    success = msg.data
    
def callback_curr_lane(msg):
    global curr_lane
    curr_lane = msg.data
    
def callback_change_lane_finished(msg):
    global change_lane_finished
    change_lane_finished = msg.data    

def cruise():
    global pub_keep_distance, pub_cruise, pub_change_lane_on_left, pub_change_lane_on_right, pub_action    
    pub_keep_distance.publish(False)
    pub_change_lane_on_left.publish(False)
    pub_change_lane_on_right.publish(False)    
    pub_cruise.publish(True)    

def keep_distance():
    global pub_keep_distance, pub_cruise, pub_change_lane_on_left, pub_change_lane_on_right, pub_action, curr_lane
    pub_cruise.publish(False)
    pub_change_lane_on_left.publish(False)
    pub_change_lane_on_right.publish(False)    
    pub_keep_distance.publish(True)    

def change_lane_on_left():
    global pub_keep_distance, pub_cruise, pub_change_lane_on_left, pub_action, curr_lane

    pub_keep_distance.publish(False)
    pub_cruise.publish(False)    
    pub_change_lane_on_right.publish(False)
    pub_change_lane_on_left.publish(True)
    
def change_lane_on_right():
    global pub_keep_distance, pub_cruise, pub_change_lane_on_right, pub_action, curr_lane

    pub_keep_distance.publish(False)
    pub_cruise.publish(False)    
    pub_change_lane_on_left.publish(False)    
    pub_change_lane_on_right.publish(True)    
            
def main(speed_left, speed_right):
    global free_N, free_NW, free_W, free_SW, free_NE, free_E, free_SE,  curr_lane, change_lane_finished
    global pub_keep_distance, pub_cruise, pub_change_lane_on_left, pub_change_lane_on_right, pub_action
    
    vel_cars_left_lane = int(speed_left)
    vel_cars_right_lane = int(speed_right)    
    
    print("INITIALIZING POLICY...", flush=True)
    rospy.init_node("test_CART")
    rate = rospy.Rate(10) #Hz

    rospy.Subscriber("/free/north", Bool, callback_free_N)       
    rospy.Subscriber("/free/north_west", Bool, callback_free_NW)
    rospy.Subscriber("/free/west"      , Bool, callback_free_W)
    rospy.Subscriber("/free/south_west", Bool, callback_free_SW)
    rospy.Subscriber("/free/north_east"      , Bool, callback_free_NE)        
    rospy.Subscriber("/free/east"      , Bool, callback_free_E)    
    rospy.Subscriber("/free/south_east"      , Bool, callback_free_SE)
    rospy.Subscriber("/current_lane", Bool, callback_curr_lane)
   
       
    pub_policy_started  = rospy.Publisher("/policy_started", Empty, queue_size=1)
    pub_cruise = rospy.Publisher("/cruise/enable", Bool, queue_size=1)
    pub_keep_distance    = rospy.Publisher("/follow/enable", Bool, queue_size=1)
    pub_change_lane_on_left = rospy.Publisher("/start_change_lane_on_left", Bool, queue_size=1)
    pub_change_lane_on_right = rospy.Publisher("/start_change_lane_on_right", Bool, queue_size=1)    
    pub_action = rospy.Publisher("/action", String, queue_size=1)
    
    pub_speed_cars_left_lane = rospy.Publisher("/speed_cars_left_lane", Float64, queue_size=1)
    pub_speed_cars_right_lane = rospy.Publisher("/speed_cars_right_lane", Float64, queue_size=1)       

    free_NW = True
    free_W  = True
    free_SW = True
    free_NE = True
    free_E  = True                
    free_SE = True
    curr_lane = True

    path = os.getcwd()
    file = path + "/src/self_driving_car/scripts/decision_models/LR_20.lr"
    print(file, flush=True)    
    try:
       model = pickle.load(open(file, 'rb'))  
    except Exception as error:
       print("Error loading model:", error)        

    # Pause policy.py a little bit
    rate = rospy.Rate(1) #Hz
    i = 0
    while not rospy.is_shutdown() and i < 2:
        pub_policy_started.publish()
        rate.sleep()
        print("Publishing policy_started", i )
        i = i + 1
    rate = rospy.Rate(10) #Hz
    
    action = action_prev = "NA"            
    while not rospy.is_shutdown():
        pub_policy_started.publish()
        pub_speed_cars_left_lane.publish(vel_cars_left_lane)
        pub_speed_cars_right_lane.publish(vel_cars_right_lane)          
        
        try:

           # Patch   
           if curr_lane:
             free_NE = free_N
           else: 
             free_NW = free_N   
        
           X = pd.DataFrame(columns=["curr_lane", "free_E", "free_NE", "free_NW", "free_SE", "free_SW", "free_W"])

           row_val = {
             "curr_lane": curr_lane,
             "free_E": free_E,
             "free_NE": free_NE,
             "free_NW": free_NW,
             "free_SE": free_SE,
             "free_SW": free_SW,
             "free_W": free_W
           }

           X.loc[len(X)] = row_val

        except Exception as error:
           print("An error occurred constructing predictors:", error) 
               
        try:
           start_time = time.time()
           y = model.predict(X)
           action = y[0]
           end_time = time.time()
           testing_time = end_time - start_time
           print("Prediction time:", testing_time, flush = True)           
        except Exception as error:
           print("An error occurred getting prediction:", error)
           
        print(action, flush = True)
        action_prev = action
        print("curr_lane", curr_lane, "free_NE", free_NE, "free_NW", free_NW, "free_SW", free_SW, "free_W", free_W, "free_SE", free_SE, "free_E", free_E,  flush = True)
    
        if action == "Cruise":
           pub_action.publish("Cruise")
           cruise()                
        elif action == "Keep":
           pub_action.publish("Keep distance")
           keep_distance()        
        elif action == "Change_to_left":
           pub_action.publish("Change_to_left")
           change_lane_on_left()
           print ("Waiting for change lane to finish...", flush = True, end="")
           rospy.wait_for_message("/change_lane_finished", Bool, timeout=10000.0)
           print (" End", flush = True)           
        elif action == "Change_to_right":
           pub_action.publish("Change_to_right")
           change_lane_on_right()                
           print ("Waiting for change lane to finish...", flush = True, end="")
           rospy.wait_for_message("/change_lane_finished", Bool, timeout=10000.0)
           print (" End", flush = True)
                                                      
        rate.sleep()

if __name__ == "__main__":

    if len(sys.argv) != 3:
       print("Usage: rosrun self_driving_car test_CART.py speed_left speed_right (m/s)", len(sys.argv))
    else:   
       try:
          speed_left = sys.argv[1]
          speed_right = sys.argv[2]
          main(speed_left, speed_right)
       except:
          rospy.ROSInterruptException
       pass

    

