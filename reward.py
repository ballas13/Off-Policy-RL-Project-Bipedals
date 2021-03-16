
def get_reward(desired_vel, velocity, height, previous_height, fall):
	reward = 0
	if velocity < 0:
		reward = -0.5
	elif velocity > desired_vel:
		reward = desired_vel / velocity
	else:
		reward = velocity / desired_vel  

	if fall:
		reward-=2
	else:
		if abs(previous_height - 0.75) < 0.05 and abs(height - 0.75) > 0.05:
			reward-=1
		elif abs(height - 0.75) < 0.05:
			if reward > 0.9:
				if reward > 0.95:
					reward+=1
				reward+=1
			reward+=0.5

	return reward

