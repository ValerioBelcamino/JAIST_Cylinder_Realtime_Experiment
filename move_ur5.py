#!/usr/bin/python3

import rospy
import sys
import moveit_commander
import geometry_msgs.msg
import moveit_msgs.msg
from trajectory_msgs.msg import JointTrajectory
import time


def create_geometry_msg(x, y, z, rx, ry, rz, rw):
	msg = geometry_msgs.msg.Pose()
	msg.position.x = x
	msg.position.y = y
	msg.position.z = z
	msg.orientation.x = rx
	msg.orientation.y = ry
	msg.orientation.z = rz
	msg.orientation.w = rw
	return msg

def move_ur5():
	moveit_commander.roscpp_initialize(sys.argv)
	rospy.init_node("move_ur5_node", anonymous=True)
	pub = rospy.Publisher('scaled_pos_joint_traj_controller/command', JointTrajectory)
	
	robot = moveit_commander.RobotCommander()
	scene = moveit_commander.PlanningSceneInterface()
	
	group_name = "manipulator"
	move_group = moveit_commander.MoveGroupCommander(group_name)
	
	initial_x = 0.00
	initial_y = 0.30
	initial_z = 0.5
	initial_rx = 1.0
	initial_ry = 0.0
	initial_rz = 0.0
	initial_rw = 0.0


	target_pose = create_geometry_msg(initial_x,
									  initial_y,
									  initial_z,
									  initial_rx,
									  initial_ry,
									  initial_rz,
									  initial_rw)

	
	move_group.set_pose_target(target_pose)
	
	#plan = move_group.go(wait=True)
	#print(plan)
	waypoints = []
	waypoints.append(target_pose)

	(plan, fraction) = move_group.compute_cartesian_path(waypoints, 0.01, 0.0)
	
	print(plan)
	print(fraction)
	#exit()
	input("press Enter to continue")
	move_group.execute(plan, wait=True)
	#pub.publish(plan)
	
	move_group.clear_pose_targets()


if __name__ == "__main__":
	try:
		move_ur5()
	except rospy.ROSInterruptException:
		pass
