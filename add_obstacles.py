#!/usr/bin/python3
import sys
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from moveit_commander import PlanningSceneInterface
 
def add_table(scene):
    # Define the table size and position
    table_pose = geometry_msgs.msg.PoseStamped()
    table_pose.header.frame_id = "base_link"  # Change this depending on your setup
    table_pose.pose.position.x = 0.0
    table_pose.pose.position.y = 0.0
    table_pose.pose.position.z = -0.2  # height of the table
    table_name = "table"
 
    # Define the size of the table (x, y, z)
    scene.add_box(table_name, table_pose, size=(1.2, 1.2, 0.4))
    rospy.sleep(1)
 
def add_cylinder(scene, eef_link):
    # Define the cylinder size and position
    cylinder_pose = geometry_msgs.msg.PoseStamped()
    cylinder_pose.header.frame_id = eef_link  # Attach to the end effector
    cylinder_pose.pose.position.x = 0.0
    cylinder_pose.pose.position.y = 0.0
    cylinder_pose.pose.position.z = 0.21  # adjust height above the gripper
    cylinder_name = "cylinder"
 
    # Define the size of the cylinder (radius, height)
    scene.add_cylinder(cylinder_name, cylinder_pose, height=0.41, radius=0.06)
    scene.attach_mesh(eef_link, cylinder_name)
    rospy.sleep(1)
 
def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('add_objects_to_scene', anonymous=True)
 
    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
 
    rospy.sleep(2)  # Give some time for the scene to initialize
 
    # Add a table to the scene
    add_table(scene)
 
    # Attach a cylindrical object to the end effector
    group_name = "manipulator"  # Replace with your group name
    move_group = moveit_commander.MoveGroupCommander(group_name)
    eef_link = move_group.get_end_effector_link()
    print(eef_link)
    add_cylinder(scene, eef_link)
 
    rospy.sleep(1)  # Wait for the scene to update
 
    # Keep the node alive
    rospy.spin()
 
    moveit_commander.roscpp_shutdown()
 
if __name__ == '__main__':
    main()
