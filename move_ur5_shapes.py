#!/usr/bin/python3

import rospy
import sys
import moveit_commander
import geometry_msgs.msg
import moveit_msgs.msg
from trajectory_msgs.msg import JointTrajectory
import yaml
import time
import os


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
	 pub = rospy.Publisher(
		  'scaled_pos_joint_traj_controller/command', JointTrajectory)

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

	 target_pose0 = create_geometry_msg(initial_x + 0.15,
												  initial_y ,
												  initial_z,
												  initial_rx,
												  initial_ry,
												  initial_rz,
												  initial_rw)

	 target_pose1 = create_geometry_msg(initial_x,
												  initial_y + 0.3,
												  initial_z,
												  initial_rx,
												  initial_ry,
												  initial_rz,
												  initial_rw)

	 target_pose2 = create_geometry_msg(initial_x - 0.15,
												  initial_y ,
												  initial_z,
												  initial_rx,
												  initial_ry,
												  initial_rz,
												  initial_rw)


	 move_group.set_pose_target(target_pose)

	 # plan = move_group.go(wait=True)
	 # print(plan)
	 waypoints = [
						  target_pose,
						  target_pose0,
						  target_pose1,
						  target_pose2,
						  target_pose
					 ]

	 # waypoints.append(target_pose)

	 (plan, fraction) = move_group.compute_cartesian_path(waypoints, 0.01, 0.0)

	 # print(plan)
	 print(fraction)
	 if fraction < 1.0:
		  rospy.logwarn(
				"Only a portion of the path was planned: {:.2f}%".format(fraction * 100))

	 # Save the plan to a YAML file
	 trajectory = plan.joint_trajectory
 
	 # Convert trajectory to a dictionary
	 traj_dict = {
		  'joint_names': trajectory.joint_names,
		  'points': []
	 }
 
	 for point in trajectory.points:
		  point_dict = {
				'positions': point.positions,
				'velocities': point.velocities,
				'accelerations': point.accelerations,
				'effort': point.effort,
				'time_from_start': point.time_from_start.to_sec()
		  }
		  traj_dict['points'].append(point_dict)
 
	 # Write to YAML file
	 # □  ◇  ○  ⧖  △
	 shape = '△'
	 file_path = os.path.join(
	 						os.path.dirname(os.path.abspath(__file__)),
	 						f'{shape}.yaml')

	 with open(file_path, 'w') as outfile:
		  yaml.dump(traj_dict, outfile, default_flow_style=False)
 
	 rospy.loginfo("Trajectory saved to {}".format(file_path))

	 input("press Enter to continue")
	 move_group.execute(plan, wait=True)
	 # pub.publish(plan)
	 
	 move_group.clear_pose_targets()


if __name__ == "__main__":
	 try:
		  move_ur5()
	 except rospy.ROSInterruptException:
		  pass



