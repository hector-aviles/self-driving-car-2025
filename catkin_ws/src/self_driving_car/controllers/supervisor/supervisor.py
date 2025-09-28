#!/usr/bin/env python3
import numpy as np
import rospy
import math
from std_msgs.msg import Empty, Float64
from geometry_msgs.msg import Pose2D
from geometry_msgs.msg import Twist
from controller import Supervisor

TIME_STEP = 10
robot = Supervisor()


# Cruise speed cars in left lane 
def callback_speed_cars_left_lane( msg ):
  global speed_cars_left_lane
  speed_cars_left_lane = msg.data
  
# Cruise speed cars in right lane 
def callback_speed_cars_right_lane( msg ):
  global speed_cars_right_lane
  speed_cars_right_lane = msg.data  

def main():

    global start, speed_cars_left_lane, speed_cars_right_lane
    print('Starting Controller Supervisor...')
    
    speed_cars_left_lane = 0.0
    speed_cars_right_lane = 0.0

    cars = [robot.getFromDef('vehicle_1'), robot.getFromDef('vehicle_2')]

    tf = []
    i = 0
    for car in cars:
        if car is not None:
              tf.append(car.getField("translation"))
              values = tf[i].getSFVec3f()
              #print(i, ")", "Initial:", values)
              rand_val = np.random.uniform(-2,2,1)
              #print("Random number", rand_val)
              values[0] = values[0] + rand_val
              #print("New x value", values[0])
              tf[i].setSFVec3f(values)
              car.resetPhysics()   
        i = i + 1 

    bmw  = robot.getFromDef('BMW_X5')  

    #linear_velocity_North = cars[0].getField("translation")
    start = False
    rospy.init_node("supervisor_node")
    loop = rospy.Rate(1000/TIME_STEP)
    
    rospy.Subscriber("/speed_cars_left_lane", Float64, callback_speed_cars_left_lane)
    rospy.Subscriber("/speed_cars_right_lane", Float64, callback_speed_cars_right_lane)          
        
    pub_bmw_pose  = rospy.Publisher("/self_driving_pose", Pose2D, queue_size=1)
    pub_car_1_pose  = rospy.Publisher("/car_1_pose", Pose2D, queue_size=1)
    pub_car_2_pose  = rospy.Publisher("/car_2_pose", Pose2D, queue_size=1)

    msg_bmw_pose = Pose2D()
    msg_car_pose = Pose2D()
    
    print("SupervisorJIFS.->Waiting for start signal")
    rospy.wait_for_message("/policy_started", Empty, timeout=50000.0)
    print("SupervisorJIFS.->Start signal received")    
        
    while robot.step(TIME_STEP) != -1 and not rospy.is_shutdown():
        # lateral_car_1.setVelocity([0,-5,0, 0,0,0])
        # lateral_car_2.setVelocity([0,-6,0, 0,0,0])
        i = 0 
        for car in cars:
            if car is not None:        
               values = tf[i].getSFVec3f()
               msg_car_pose.x = values[0] 
               msg_car_pose.y = values[1]
               msg_car_pose.theta = values[2]   
                           
               if msg_car_pose.y < 0:            
                  car.setVelocity([speed_cars_left_lane,0,0, 0,0,0])
               else:
                  car.setVelocity([speed_cars_right_lane,0,0, 0,0,0])
                           
               if i == 0:
                  pub_car_1_pose.publish(msg_car_pose)  
               elif i == 1:
                  pub_car_2_pose.publish(msg_car_pose)               
            i = i + 1                     
               
        bmw_pose = bmw.getPosition()
        bmw_orient = bmw.getOrientation()
        msg_bmw_pose.x = bmw_pose[0]
        msg_bmw_pose.y = bmw_pose[1]
        msg_bmw_pose.theta = math.atan2(bmw_orient[3], bmw_orient[0])       
        pub_bmw_pose.publish(msg_bmw_pose)
                          
        loop.sleep()
  
  
        
if __name__ == "__main__":
    try:
        main()
    except:
        pass

