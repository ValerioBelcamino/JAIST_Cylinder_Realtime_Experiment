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


def reset_timer(msg):
	global time_since_last_action
	print('Restarting Time')
	time_since_last_action = time.time()


def execute_looping_trajectory(msg):
	global move_group, shapes, idx, trajectories, stop_looping
	if idx > 4:
		print('>>>>> Finito execute_looping_trajectory <<<<<')
		return
	print(f'\n\n{shapes[idx]}')


	current_traj_dict = trajectories[idx]

	while not stop_looping:
		print(f'Executing {shapes[idx]} trajectory!')
		
		if not execute_trajectory(current_traj_dict, move_group):
			break
	print("Stopping")
	stop_looping = False
	dummy_pub.publish(True)
	 


def start_next_action(msg):
	global shapes, idx, time_since_last_action, out_file_path, available_for_msgs, trajectories, stop_looping, dummy_pub
	print(available_for_msgs)
	if msg.data == True and available_for_msgs:
		stop_looping = True
		available_for_msgs = False

		elapsed_time = time.time() - time_since_last_action

		print(f'{elapsed_time} seconds since last action.')

		with open(out_file_path, 'a+') as of:
			of.write(f'{shapes[idx]}: {actions[idx]}: {elapsed_time}\n')

		
		file_path = os.path.join(
						os.path.dirname(os.path.abspath(__file__)),
						f'{shapes[idx]}.yaml')
		 
		if idx < 4:
			idx += 1
			print(f'Message arrived, changing shape to {shapes[idx]}!')
		else:
			idx += 1
			print('>>>>> Finito start_next_action <<<<<')
			return

		available_for_msgs = True

	else:
		return


def load_trajectory_by_shape(shape):
	file_path = os.path.join(
						os.path.dirname(os.path.abspath(__file__)),
						f'{shape}.yaml')
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
	success = move_group.execute(trajectory, wait=True)
	#print('start sleeping')
	#time.sleep(5.0)
	#print('stop sleeping')

	rospy.loginfo("Trajectory execution completed.")
	#activation_pub.publish(True)
	return success
	
 
if __name__ == '__main__':
	try:
		# Initialize moveit_commander and rospy node
		moveit_commander.roscpp_initialize(sys.argv)
		rospy.init_node('load_and_execute_trajectory', anonymous=True)
		rospy.Subscriber('/StartNextAction', Bool, start_next_action, queue_size=1)
		rospy.Subscriber('/StartLoopingTrajectory', Bool, execute_looping_trajectory, queue_size=1)
		rospy.Subscriber('/ActivateClassifier', Bool, reset_timer, queue_size=1)
		dummy_pub = rospy.Publisher('/StartLoopingTrajectory', Bool)
		#activation_pub = rospy.Publisher('/ActivateClassifier', Bool, queue_size=1)

		# Initialize the move group
		group_name = "manipulator"
		move_group = moveit_commander.MoveGroupCommander(group_name)
		print("MoveGroup initialized\n")
		shapes = ['□', '◇', '⧖', '△', '○']
		trajectories = []
		actions = ['patting', 'massaging', 'patting', 'pinching', 'press']
		#actions = ['pull', 'squeeze', 'rub', 'scratching', 'shaking']
		#actions = ['trembling', 'tapping', 'stroke', 'massaging', 'press']
		#actions = ['patting', 'squeeze', 'stroke', 'pull', 'pinching']
		#actions = ['linger', 'rub', 'scratching', 'shaking', 'trembling']

		for s in shapes:
			trajectories.append(load_trajectory_by_shape(s))

		idx = 0
		available_for_msgs = True
		stop_looping = False

		participant = '6left'
		trial = 0

		out_file_path_base = os.path.join(
						os.path.dirname(os.path.abspath(__file__)),
						f'Experiment/{participant}')
		if not os.path.exists(out_file_path_base):
			os.makedirs(out_file_path_base)
		out_file_path = os.path.join(out_file_path_base, f'{trial}.txt')



		input('Press START to continue.')
		dummy_pub.publish(True)
		time_since_last_action = time.time()
		
		print('Rospy Spinning!')		
		rospy.spin()

		# Shutdown MoveIt!
		moveit_commander.roscpp_shutdown()
		print("Kiled moveit commander!")

	except rospy.ROSInterruptException:
		pass
 

