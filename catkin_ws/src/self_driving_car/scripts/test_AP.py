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

module_path = os.getcwd()
module_path = module_path + "/src/self_driving_car/scripts/decision_models/"
sys.path.append(module_path)
from APClassifier import RuleClassifier

# Import the ActionPolicy class
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class ActionPolicy(BaseEstimator, ClassifierMixin):
    def __init__(self):
        # Precomputed lookup table of size 128
        self.actions = [None] * 128
        self.actions[0] = "keep"
        self.actions[1] = "keep"
        self.actions[2] = "keep"
        self.actions[3] = "keep"
        self.actions[4] = "keep"
        self.actions[5] = "cruise"
        self.actions[6] = "change_to_right"
        self.actions[7] = "cruise"
        self.actions[8] = "cruise"
        self.actions[9] = "keep"
        self.actions[10] = "cruise"
        self.actions[11] = "keep"
        self.actions[12] = "cruise"
        self.actions[13] = "cruise"
        self.actions[14] = "change_to_right"
        self.actions[15] = "cruise"
        self.actions[16] = "keep"
        self.actions[17] = "keep"
        self.actions[18] = "keep"
        self.actions[19] = "keep"
        self.actions[20] = "keep"
        self.actions[21] = "cruise"
        self.actions[22] = "change_to_right"
        self.actions[23] = "cruise"
        self.actions[24] = "cruise"
        self.actions[25] = "keep"
        self.actions[26] = "cruise"
        self.actions[27] = "keep"
        self.actions[28] = "cruise"
        self.actions[29] = "cruise"
        self.actions[30] = "change_to_right"
        self.actions[31] = "cruise"
        self.actions[32] = "keep"
        self.actions[33] = "keep"
        self.actions[34] = "keep"
        self.actions[35] = "keep"
        self.actions[36] = "keep"
        self.actions[37] = "cruise"
        self.actions[38] = "change_to_right"
        self.actions[39] = "cruise"
        self.actions[40] = "cruise"
        self.actions[41] = "keep"
        self.actions[42] = "cruise"
        self.actions[43] = "keep"
        self.actions[44] = "cruise"
        self.actions[45] = "cruise"
        self.actions[46] = "change_to_right"
        self.actions[47] = "cruise"
        self.actions[48] = "keep"
        self.actions[49] = "keep"
        self.actions[50] = "keep"
        self.actions[51] = "keep"
        self.actions[52] = "keep"
        self.actions[53] = "cruise"
        self.actions[54] = "change_to_right"
        self.actions[55] = "cruise"
        self.actions[56] = "cruise"
        self.actions[57] = "keep"
        self.actions[58] = "cruise"
        self.actions[59] = "keep"
        self.actions[60] = "cruise"
        self.actions[61] = "cruise"
        self.actions[62] = "change_to_right"
        self.actions[63] = "cruise"
        self.actions[64] = "keep"
        self.actions[65] = "keep"
        self.actions[66] = "keep"
        self.actions[67] = "keep"
        self.actions[68] = "keep"
        self.actions[69] = "cruise"
        self.actions[70] = "change_to_right"
        self.actions[71] = "cruise"
        self.actions[72] = "cruise"
        self.actions[73] = "change_to_left"
        self.actions[74] = "cruise"
        self.actions[75] = "change_to_left"
        self.actions[76] = "cruise"
        self.actions[77] = "cruise"
        self.actions[78] = "change_to_right"
        self.actions[79] = "cruise"
        self.actions[80] = "keep"
        self.actions[81] = "keep"
        self.actions[82] = "keep"
        self.actions[83] = "keep"
        self.actions[84] = "keep"
        self.actions[85] = "cruise"
        self.actions[86] = "change_to_right"
        self.actions[87] = "cruise"
        self.actions[88] = "cruise"
        self.actions[89] = "change_to_left"
        self.actions[90] = "cruise"
        self.actions[91] = "change_to_left"
        self.actions[92] = "cruise"
        self.actions[93] = "cruise"
        self.actions[94] = "change_to_right"
        self.actions[95] = "cruise"
        self.actions[96] = "keep"
        self.actions[97] = "keep"
        self.actions[98] = "keep"
        self.actions[99] = "keep"
        self.actions[100] = "keep"
        self.actions[101] = "cruise"
        self.actions[102] = "change_to_right"
        self.actions[103] = "cruise"
        self.actions[104] = "cruise"
        self.actions[105] = "change_to_left"
        self.actions[106] = "cruise"
        self.actions[107] = "change_to_left"
        self.actions[108] = "cruise"
        self.actions[109] = "cruise"
        self.actions[110] = "change_to_right"
        self.actions[111] = "cruise"
        self.actions[112] = "keep"
        self.actions[113] = "keep"
        self.actions[114] = "keep"
        self.actions[115] = "keep"
        self.actions[116] = "keep"
        self.actions[117] = "cruise"
        self.actions[118] = "change_to_right"
        self.actions[119] = "cruise"
        self.actions[120] = "cruise"
        self.actions[121] = "change_to_left"
        self.actions[122] = "cruise"
        self.actions[123] = "change_to_left"
        self.actions[124] = "cruise"
        self.actions[125] = "cruise"
        self.actions[126] = "change_to_right"
        self.actions[127] = "cruise"

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        cols = ['curr_lane', 'free_E', 'free_NE', 'free_NW', 'free_SE', 'free_SW', 'free_W']
        data = X[cols].to_numpy().astype(bool).astype(int)
        powers = 2 ** np.arange(7)
        indices = data @ powers
        return np.array(self.actions)[indices]

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
    global free_NW, free_W, free_SW, free_NE, free_E, free_SE,  curr_lane, change_lane_finished
    global pub_keep_distance, pub_cruise, pub_change_lane_on_left, pub_change_lane_on_right, pub_action
    
    vel_cars_left_lane = int(speed_left)
    vel_cars_right_lane = int(speed_right)    
    
    
    print("INITIALIZING POLICY...", flush=True)
    rospy.init_node("test_AP")
    rate = rospy.Rate(10) #Hz
   
    rospy.Subscriber("/free/north", Bool, callback_free_N)              
    rospy.Subscriber("/free/north_west", Bool, callback_free_NW)
    rospy.Subscriber("/free/west"      , Bool, callback_free_W)
    rospy.Subscriber("/free/south_west", Bool, callback_free_SW)
    rospy.Subscriber("/free/north_east", Bool, callback_free_NE)
    rospy.Subscriber("/free/east", Bool, callback_free_E)    
    rospy.Subscriber("/free/south_east", Bool, callback_free_SE)
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
    free_NE = False
    free_E  = True                
    free_SE = True
    curr_lane = True

    # Create instance of ActionPolicy instead of loading from file
    print("Creating ActionPolicy classifier...", flush=True)
    try:
        model = ActionPolicy()
        print("ActionPolicy classifier created successfully", flush=True)
    except Exception as error:
        print("Error creating ActionPolicy classifier:", error)        

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
           
        print("Predicted action:", action, flush = True)
        action_prev = action
        print("curr_lane", curr_lane, "free_NE", free_NE, "free_NW", free_NW, "free_SW", free_SW, "free_W", free_W, "free_SE", free_SE, "free_E", free_E,  flush = True)
                
        if action == "cruise":
           pub_action.publish("cruise")
           cruise()                
        elif action == "keep":
           pub_action.publish("keep")
           keep_distance()        
        elif action == "change_to_left":
           pub_action.publish("change_to_left")
           change_lane_on_left()
           print ("Waiting for change lane to finish...", flush = True, end="")
           rospy.wait_for_message("/change_lane_finished", Bool, timeout=10000.0)
           print (" End", flush = True)           
        elif action == "change_to_right":
           pub_action.publish("change_to_right")
           change_lane_on_right()                
           print ("Waiting for change lane to finish...", flush = True, end="")
           rospy.wait_for_message("/change_lane_finished", Bool, timeout=10000.0)
           print (" End", flush = True)
        else:
           print(f"Unknown action: {action}", flush = True)
                                                      
        rate.sleep()

if __name__ == "__main__":

    if len(sys.argv) != 3:
       print("Usage: rosrun self_driving_car test_AP.py speed_left speed_right (m/s)", len(sys.argv))
    else:   
       try:
          speed_left = sys.argv[1]
          speed_right = sys.argv[2]
          main(speed_left, speed_right)
       except:
          rospy.ROSInterruptException
       pass
