#!/usr/bin/python3

import rospy
import sys
import moveit_commander
import yaml
import time
import os
from std_msgs.msg import Bool
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
 

def load_and_execute(msg):
	global move_group, shapes, idx, time_since_last_action, out_file_path, available_for_msgs
	print(available_for_msgs)
	if msg.data == True and available_for_msgs:
		available_for_msgs = False
		print(f'\n\n{shapes[idx]}')

		elapsed_time = time.time() - time_since_last_action

		print(f'{elapsed_time} seconds since last action.')

		with open(out_file_path, 'a+') as of:
			of.write(f'{shapes[idx]}: {actions[idx]}: {elapsed_time}\n')

		print(f'Message arrived, preparing to execute {shapes[idx]} trajectory!')
		file_path = os.path.join(
						os.path.dirname(os.path.abspath(__file__)),
						f'{shapes[idx]}.yaml')
		traj_dict = load_trajectory(file_path)
		execute_trajectory(traj_dict, move_group)
		 
		if idx < 4:
			idx += 1
		else:
			idx = 0
		time_since_last_action = time.time()
		available_for_msgs = True
	else:
		return


def load_trajectory(file_path):
	with open(file_path, 'r') as infile:
		traj_dict = yaml.full_load(infile)
		print(f'Read trajectory {file_path}')
	return traj_dict
 

def execute_trajectory(traj_dict, move_group):

	# Create a RobotTrajectory message
	trajectory = RobotTrajectory()
	trajectory.joint_trajectory.joint_names = traj_dict['joint_names']
 
	for point_dict in traj_dict['points']:
		point = JointTrajectoryPoint()
		point.positions = point_dict['positions']
		point.velocities = point_dict['velocities']
		point.accelerations = point_dict['accelerations']
		point.effort = point_dict['effort']
		point.time_from_start = rospy.Duration(point_dict['time_from_start'])
		trajectory.joint_trajectory.points.append(point)
 
	# Execute the trajectory
	move_group.execute(trajectory, wait=True)
	#print('start sleeping')
	#time.sleep(5.0)
	#print('stop sleeping')

	rospy.loginfo("Trajectory execution completed.")
	activation_pub.publish(True)
 
	
 
if __name__ == '__main__':
	try:
		# Initialize moveit_commander and rospy node
		moveit_commander.roscpp_initialize(sys.argv)
		rospy.init_node('load_and_execute_trajectory', anonymous=True)
		rospy.Subscriber('/StartNextAction', Bool, load_and_execute, queue_size=1)
		activation_pub = rospy.Publisher('/ActivateClassifier', Bool, queue_size=1)

		# Initialize the move group
		group_name = "manipulator"
		move_group = moveit_commander.MoveGroupCommander(group_name)
		print("MoveGroup initialized\n")
		shapes = ['□', '◇', '⧖', '△', '○']
		actions = ['linger', 'massaging', 'patting', 'pinching', 'press']
		actions = ['pull', 'squeeze', 'rub', 'scratching', 'shaking']
		actions = ['trembling', 'tapping', 'stroke', 'massaging', 'press']
		actions = ['patting', 'squeeze', 'stroke', 'pull', 'pinching']
		actions = ['linger', 'rub', 'scratching', 'shaking', 'trembling']

		idx = 0
		available_for_msgs = True

		participant = '1left'
		trial = 4

		out_file_path_base = os.path.join(
						os.path.dirname(os.path.abspath(__file__)),
						f'Experiment/{participant}')
		if not os.path.exists(out_file_path_base):
			os.makedirs(out_file_path_base)
		out_file_path = os.path.join(out_file_path_base, f'{trial}.txt')



		input('Press START to continue.')

		time_since_last_action = time.time()
		activation_pub.publish(True)

		#for shape in ['□', '◇', '⧖', '△', '○']:
			# Define the file path to load the trajectory
			#file_path = file_path = os.path.join(
								#os.path.dirname(os.path.abspath(__file__)),
								#f'{shape}.yaml')
			#traj_dict = load_trajectory(file_path)
			#execute_trajectory(traj_dict, move_group)



			#input("Press ENTER to continue")
			#exit()


		
		print('Rospy Spinning!')		
		rospy.spin()

		# Shutdown MoveIt!
		moveit_commander.roscpp_shutdown()
		print("Kiled moveit commander!")

	except rospy.ROSInterruptException:
		pass
 

