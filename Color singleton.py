import os
import time
import csv
import random
from math import isnan
## To install numpy go to: "Tools -> Package Manager" under "Cmd" type "install numpy==1.16.4"
try:
	import numpy as np
except ValueError:
	print('First install numpy: open "Tools -> Package Manager" under "Cmd" type "install numpy==1.16.4')
	
import viz
import viztask
import vizcam
import vizact
import vizconnect
import vizshape
import vizinfo
import vizfx
import vizproximity
import vizmat
from vizfx.postprocess.color import ColorScaleEffect

import steamvr


### FROM PP ###
def set_graphics_quality():
    """Vsync ON; Multisample 8, glFinish ON."""
    # Just to make sure vertical sync is ON, against tearing.
    viz.vsync(1)
    # Helps reduce latency.
    viz.setOption('viz.glFinish', 1)
    viz.setMultiSample(8)

def create_text2d(message='Bitte Warten', position = [0,1,4]):
    # Text instructions for participant.
    text2D = viz.addText(message,pos=position)
    text2D.alignment(viz.ALIGN_CENTER_BOTTOM)
    text2D.setBackdrop(viz.BACKDROP_RIGHT_BOTTOM)
    text2D.resolution(1)
    text2D.fontSize(1)
    text2D.setScale([0.2]*3)
    text2D.disable(viz.LIGHTING)
    text2D.color(0,0,0)
    return text2D
### END FROM PP ###

def settings_init(exp):
	class Settings:
		def __init__(self, name):
			self.name = name
			
			self.data_path = './data'
			
			self.log_heading = 'pp_nr,pp_id,session,block,trial,dis_pos_rel,filler2_pos,dis_pos_cont,tar_shape,t_search_on,tar_col_name,tar_pos,filler1_pos,dis_pos,finger_on_shape,finger_on_target,dis_condition,finger_on_filler,t_trial_start,search_time,finger_on_start,long_start,t_on_start,interval_beforesearch,finger_on_distractor,max_search_time,reach_end_cutoff_t,nr_trials,nr_blocks,min_dist_start,min_dist_shape' + '\n'
			# 'pp_nr,pp_id,session,block,trial,dis_pos,long_start,filler1_pos,dis_pos_rel,t_search_on,finger_on_target,filler2_pos,dis_condition,tar_shape,finger_on_filler,tar_pos,tar_col_name,t_trial_start,search_time,finger_on_shape,finger_on_distractor,finger_on_start,max_search_time,reach_end_cutoff_t,nr_trials,nr_blocks,min_dist_start,min_dist_shape' + '\n'
			self.tolog_exp_vars = ['max_search_time','reach_end_cutoff_t','nr_trials','nr_blocks','min_dist_start','min_dist_shape']
			
			# 5 factors
			# target shape: diamond or sphere
					# 2 levels
			self.target_shapes = ['diamond','sphere']
			
			# target color: red or blue
					# 2 levels
			self.target_colors = ['red','blue']
			
			# distractor color: same or different from target
					# 2 levels
			self.distractor_colors = ['same','different']
			
			# target position: 4 (shapes and) positions
					# 4 levels
			self.target_positions = [0,1,2,3]
			
			# distractor position (relative to target; positions away from target clockwise): 1, 2, 3
				# 3 levels
			self.distractor_positions = [1,2,3]
			
			# total different kind of trials: 2 * 2 * 2 * 4 * 3 = 96
			#nr_trial_types = len(self.target_shapes) * len(self.target_colors) * len(self.distractor_colors) * len(self.target_positions) * len(self.distractor_positions)
			self.nr_trial_types = len(self.target_shapes) * len(self.target_colors) * len(self.distractor_colors) * len(self.target_positions) * len(self.distractor_positions)
			
			# time in seconds
			# how long is the search display presented:
			self.max_search_time = 1
			# how much time do people have to reach to the target:
			# is adjusted based on participant's performance
			self.reach_end_cutoff_t = 1
			self.reach_end_cutoff_list = [1.0] * 100
			
			if name == "experiment":
				self.nr_trials = 672
				self.nr_blocks = 7
			elif name == "practice":
				self.nr_trials = 40
				self.nr_blocks = 2
			else:
				raise ValueError('Settings name not defined')
				
			#shape height
#			self.shape_height = 0.7
			self.shape_height = 0.77
				
			# shapes depth
			self.shape_distance = 0.57
			
			#define location of start and shapes
			#start position
			self.start_pos = [0.0,0.8595,0.223]
			
			#shapes position
			self.shape_positions = []
			# angles in degrees on circle at which shapes are presented
			# angle 0 is vertically above circle center; counting counterclockwise
			# 											i.e. positive angle is left of circle center
#			angles = [(13.5+27), 13.5, -13.5, (-13.5-27)]
			angles = [(10+20), 10, -10, (-10-20)]
#			count = 0
			for angle in angles:
				shape_pos = list(pointOnCircle([0, self.shape_height],0.46,angle))
#				if count == 3:
#					shape_pos.append(self.shape_distance+0.3)
#				else:
				shape_pos.append(self.shape_distance)
				self.shape_positions.append(shape_pos)
#				count += 1
			
			# max distance between finger position and start or target center to detect "touch"
			self.min_dist_start = 0.01
#			self.min_dist_shape = 0.1
			self.min_dist_shape = 0.07
			
			#define colors
			self.red = [1,0.5,0.5]
			self.blue = [0.5,0.5,1]
			
			## OPTOTRAK INIT ##
			###################
			# Optotrak server and marker number
			self.cf_optoserver = '127.0.0.1' #'134.176.232.248' #'134.176.76.83'
			self.cf_optomarker = 4

			# Optotrak coordinate system correction
			# 1=x, 2=y, 3=z, use negative to invert
			self.cf_optofilter = (-3, -2, -1)

			# Offset between home button and Optotrak origin (meters)
			# (i.e. left table corner, x = horizontal)
			self.cf_optohome_x = 0.43
			self.cf_optohome_y = 0.025
			self.cf_optohome_z = 0.080

			# Offset between Optotrak origin and left table corner (meters)
			self.cf_optozero_x = 0.01
			self.cf_optozero_y = 0.005
			self.cf_optozero_z = 0.01
			self.cf_DEBUG = True
			self.cf_tablewidth=0.8
			self.cf_tableheight=.855
			self.cf_tabledepth=1.6
			self.cf_hand_size=.01

			self.opto_offset = [0.0, 0.0, 0.0]
			
	exp_settings = Settings(exp)
	return exp_settings
		
		
def logger_init():
	class Logger:
		def __init__(self):
			self.tar_shape = []
			self.tar_col = []
			self.dis_col = []
			self.tar_pos = []
			self.dis_pos = []
	log_logger = Logger()
	return log_logger

def trial_vars_init(exp):
	class Trial_vars:
		def __init__(self, exp):
			# initialize all vars for trial
			self.t_trial_start = np.nan
			self.t_search_on = np.nan
			self.search_time = np.nan
			self.shapes_irrel = list(range(len(exp.target_positions)))
			self.tar_col = np.nan
			self.tar_col_name = np.nan
			self.dis_col = np.nan
			self.dis_condition = np.nan
			self.tar_pos = np.nan
			self.dis_pos = np.nan
			# position of distractor relative to target (nr of positions away clockwise)
			self.dis_pos_rel = np.nan
			self.long_start = 0
			self.finger_on_start = 0
			self.finger_on_target = 0
			self.finger_on_filler = 0
			self.finger_on_distractor = 0
			self.finger_on_shape = -1
	trial_variables = Trial_vars(exp)
#	trial_variables.
	return trial_variables

def randomization(exp, log):
	all_tar_shapes = []
	all_tar_col = []
	all_dis_col = []
	all_tar_pos = []
	all_dis_pos = []
	
	for tar_shape in range(len(exp.target_shapes)):
		for tar_col in range(len(exp.target_colors)):
			for dis_col in range(len(exp.distractor_colors)):
				for tar_pos in range(len(exp.target_positions)):
					for dis_pos in exp.distractor_positions:
						all_tar_shapes.append(tar_shape)
						all_tar_col.append(tar_col)
						all_dis_col.append(dis_col)
						all_tar_pos.append(tar_pos)
						all_dis_pos.append(dis_pos)
						
	block = list(zip(all_tar_shapes, all_tar_col, all_dis_col, all_tar_pos, all_dis_pos))
	random.shuffle(block)
	all_tar_shapes, all_tar_col, all_dis_col, all_tar_pos, all_dis_pos = zip(*block)
	
	log.tar_shape.extend(all_tar_shapes[:exp.nr_trials/exp.nr_blocks])
	log.tar_col.extend(all_tar_col[:exp.nr_trials/exp.nr_blocks])
	log.dis_col.extend(all_dis_col[:exp.nr_trials/exp.nr_blocks])
	log.tar_pos.extend(all_tar_pos[:exp.nr_trials/exp.nr_blocks])
	log.dis_pos.extend(all_dis_pos[:exp.nr_trials/exp.nr_blocks])
	
	## Check counterbalancing ##
#	counter = 0
#	condition = []
#	for tar_shape in range(len(exp.target_shapes)):
#		for tar_col in range(len(exp.target_colors)):
#			for dis_col in range(len(exp.distractor_colors)):
#				for tar_pos in range(len(exp.target_positions)):
#					for dis_pos in range(len(exp.distractor_positions)):
#						condition.append(0)
#						for trial in range(len(log.tar_shape)):
#							if (log.tar_shape[trial] == tar_shape) & (log.tar_col[trial] == tar_col) & (log.dis_col[trial] == dis_col) & (log.tar_pos[trial] == tar_pos) & (log.dis_pos[trial] == dis_pos):
#								condition[counter] += 1
#						counter +=1
#	print condition
#	print(len(condition))
#	print(sum(condition))
#	print(len(log.tar_shape))

def participantInfo():

	#Add an InfoPanel with a title bar
	participantInfo = vizinfo.InfoPanel('',title='Participant Information',align=viz.ALIGN_CENTER, icon=False)

	#Add name and ID fields
	textbox_nr = participantInfo.addLabelItem('NR',viz.addTextbox())
	participantInfo.addSeparator()
	textbox_id = participantInfo.addLabelItem('ID',viz.addTextbox())
	participantInfo.addSeparator()

	#Add submit button aligned to the right and wait until it's pressed
	submitButton = participantInfo.addItem(viz.addButtonLabel('Submit'),align=viz.ALIGN_RIGHT_CENTER)
	yield viztask.waitButtonUp(submitButton)
	
	#Collect participant data
	exp.pp_nr = "%02d" % (int(textbox_nr.get()),)
	exp.pp_id = textbox_id.get()
	pr.pp_nr = exp.pp_nr
	pr.pp_id = exp.pp_id


	participantInfo.remove()
	
def makeDiamond(position, color):
#	diamond = vizshape.addCube(size=0.1)
	diamond = vizshape.addCube(size=0.07)
	diamond.setPosition(exp.shape_positions[position])
	diamond.color(color)
	diamond.setEuler(15,45,0)
	return diamond
	
def makeSphere(position, color):
#	sphere = vizshape.addSphere(radius=0.07)
	sphere = vizshape.addSphere(radius=0.05)
	sphere.setPosition(exp.shape_positions[position])
	sphere.color(color)
	return sphere
	
def makeFinger(position):
	sphere = vizshape.addSphere(radius=0.01)
	sphere.setPosition(position)
	sphere.color([0.5,1,0.5])
	return sphere

def checkTouch(object_pos,finger_pos,distance_threshold):
	if (np.linalg.norm(np.array(object_pos) - np.array(finger_pos))) <= distance_threshold:
		return 1
	else:
		return 0
		
def pointOnCircle(center, radius, angle):
	'''
		Finding the x,y coordinates on circle, based on given angle
		Set 0 angle to vertically above circle center
		Angles go counterclockwise
		Imput angle is degrees
	'''
	from math import cos, sin, radians
	#angle in degree to radians
	angle = radians(angle+90)

	x = center[0] + (radius * cos(angle))
	y = center[1] + (radius * sin(angle))

	return x,y

def calibrateHand():
	if exp.optotrak:
		optoLink = viz.link(hand, m_hand)	
	
	text_line1 = create_text2d('Please put finger on physical start position\n press -space- to start calibration')
	yield viztask.waitKeyDown(' ')
	text_line1.message("Calibration in progress")
	#collect samples
	samples = []
	old_sample = m_hand.getPosition()
	samples.append(old_sample) 
	while len(samples) < 100:
		new_sample = m_hand.getPosition()
		if not (new_sample == old_sample):
			samples.append(new_sample)
			old_sample = new_sample
		yield viz.waitTime(0.001)
	check_position = np.mean(samples, axis = 0)
	
	position_offset = np.array(exp.start_pos) - np.array(check_position)
	print position_offset
	hand0 = optofilter.position(hand, offset=(position_offset[0],
											  position_offset[1]+0.006,
											  position_offset[2]))
	if exp.optotrak:
		optoLink = viz.link(hand0, m_hand)
	text_line1.message("Calibration done")
	yield viz.waitTime(0.75)									  
	text_line1.visible(viz.OFF)	
	
def trial(bl,tr,exp,log):
	print(str(tr+1) + ' of ' + str(exp.nr_trials))
	tr_vars = trial_vars_init(exp)
	## prealocate space for hand_samples recording during task; not for hand_samples_check i.e. during check if at start position
	hand_samples_check = []
	hand_times_check = []
	hand_samples = [np.NaN]*200
	hand_times = [np.NaN]*200
	hand_samples_count = 0
	
	tr_vars.t_trial_start = viz.tick()
	
	feedback = create_text2d(' ')
	feedback.visible(viz.OFF)
	
	tr_vars.interval_beforesearch = (random.randint(8,10)/10.0)
	
	## retrieve shape settings 
	##########################
	
	#colors
	tr_vars.tar_col_name = exp.target_colors[log.tar_col[tr]]
	tr_vars.dis_condition = exp.distractor_colors[log.dis_col[tr]]
	if tr_vars.tar_col_name == 'red':
		tr_vars.tar_col = exp.red
		if exp.distractor_colors[log.dis_col[tr]] == 'same':
			tr_vars.dis_col = exp.red
		else:
			tr_vars.dis_col = exp.blue
	elif tr_vars.tar_col_name == 'blue':
		tr_vars.tar_col = exp.blue
		if exp.distractor_colors[log.dis_col[tr]] == 'same':
			tr_vars.dis_col = exp.blue
		else:
			tr_vars.dis_col = exp.red
	#shapes
	tr_vars.tar_shape = exp.target_shapes[log.tar_shape[tr]]
	tr_vars.tar_pos =  log.tar_pos[tr]
	tr_vars.dis_pos = (tr_vars.tar_pos + log.dis_pos[tr])%4
	# dis_pos_cont as continuous counting from tar_pos 
	tr_vars.dis_pos_cont = log.dis_pos[tr]
	# dis_pos_rel as relative to tar_pos: negative is left from, positive is right from
	tr_vars.dis_pos_rel = tr_vars.dis_pos - tr_vars.tar_pos
	
	if tr_vars.tar_shape == 'diamond':
		shape1 = makeDiamond(tr_vars.tar_pos,tr_vars.tar_col)
		tr_vars.shapes_irrel.remove(tr_vars.tar_pos)
		shape2 = makeSphere(tr_vars.dis_pos,tr_vars.dis_col)
		tr_vars.shapes_irrel.remove(tr_vars.dis_pos)
		if len(tr_vars.shapes_irrel) != 2:
			raise Exception('length of shapes_irrel should be 2. length shapes_irrel: {}'.format(len(tr_vars.shapes_irrel)))
		shape3 = makeSphere(tr_vars.shapes_irrel[0],tr_vars.tar_col)
		shape4 = makeSphere(tr_vars.shapes_irrel[1],tr_vars.tar_col)
	elif tr_vars.tar_shape == 'sphere':
		shape1 = makeSphere(tr_vars.tar_pos,tr_vars.tar_col)
		tr_vars.shapes_irrel.remove(tr_vars.tar_pos)
		shape2 = makeDiamond(tr_vars.dis_pos,tr_vars.dis_col)
		tr_vars.shapes_irrel.remove(tr_vars.dis_pos)
		if len(tr_vars.shapes_irrel) != 2:
			raise Exception('length of shapes_irrel should be 2. length shapes_irrel: {}'.format(len(tr_vars.shapes_irrel)))
		shape3 = makeDiamond(tr_vars.shapes_irrel[0],tr_vars.tar_col)
		shape4 = makeDiamond(tr_vars.shapes_irrel[1],tr_vars.tar_col)
	
	shape1.visible(viz.OFF)
	shape2.visible(viz.OFF)
	shape3.visible(viz.OFF)
	shape4.visible(viz.OFF)
	
	#implement sound
	shape1_sound = shape1.playsound('./sounds/0737.wav')
	shape1_sound.pause()
	shape2_sound = shape2.playsound('./sounds/0739.wav')
	shape2_sound.pause()
	shape3_sound = shape3.playsound('./sounds/0739.wav')
	shape3_sound.pause()
	shape4_sound = shape4.playsound('./sounds/0739.wav')
	shape4_sound.pause()
	
	hand_samples_check.append('check_start')
	hand_times_check.append(viz.tick())
	
	old_sample = m_hand.getPosition()
	hand_samples_check.append(old_sample)
	hand_times_check.append(viz.tick())

	tr_vars.t_on_start = 0
	pls_break = 0
	while True:
		new_sample = m_hand.getPosition()
		if not (new_sample == old_sample):
			hand_samples_check.append(new_sample)
			hand_times_check.append(viz.tick())
			old_sample = new_sample

			tr_vars.finger_on_start = checkTouch(exp.start_pos,new_sample,exp.min_dist_start)
			
			# option to recalibrate in trial
			if viz.key.isDown('a'):
				yield calibrateHand()
				feedback.message("Press -0- to continue")
				feedback.visible(viz.ON)
				yield viztask.waitKeyDown(viz.KEY_KP_0)
				feedback.visible(viz.OFF)
			
			# wait for time after pp touch start before presenting shapes
			# make sure they stay on start until shapes are presented
			# if they move time after start counter resets
			if tr_vars.finger_on_start and tr_vars.t_on_start == 0:
				tr_vars.t_on_start = viz.tick()
			elif not tr_vars.finger_on_start and not tr_vars.t_on_start == 0:
				tr_vars.t_on_start = 0
			elif tr_vars.finger_on_start and (viz.tick() - tr_vars.t_on_start) > tr_vars.interval_beforesearch:
				pls_break = 1
			
		if pls_break:
			break
		if ((viz.tick() - tr_vars.t_trial_start) > 2.5) and (tr_vars.long_start == 0):
			feedback.message("Move finger to start position")
			feedback.visible(viz.ON)
			tr_vars.long_start = 1
		elif ((viz.tick() - tr_vars.t_trial_start) > 4) and (tr_vars.long_start):
			feedback.visible(viz.OFF)
		yield viz.waitTime(0.001)
			
	
	feedback.visible(viz.OFF)
	
	shape1.visible(viz.ON)
	shape2.visible(viz.ON)
	shape3.visible(viz.ON)
	shape4.visible(viz.ON)
	
	# t in seconds
	tr_vars.t_search_on = viz.tick() 
	
	hand_samples[hand_samples_count] = 'start_task'
	hand_times[hand_samples_count] = tr_vars.t_search_on
	hand_samples_count += 1
	
	pls_break = 0
	while True:
		new_sample = m_hand.getPosition()
		if not (new_sample == old_sample):
			hand_samples[hand_samples_count] = new_sample
			hand_times[hand_samples_count] = viz.tick()
			hand_samples_count += 1
			old_sample = new_sample

			if checkTouch(exp.shape_positions[tr_vars.tar_pos],new_sample,exp.min_dist_shape):
				tr_vars.finger_on_target = 1
				tr_vars.finger_on_shape = tr_vars.tar_pos
				shape1_sound.play()
				shape1.visible(viz.OFF)
				pls_break = 1
			elif checkTouch(exp.shape_positions[tr_vars.dis_pos],new_sample,exp.min_dist_shape):
				tr_vars.finger_on_distractor = 1
				tr_vars.finger_on_shape = tr_vars.dis_pos
				shape2_sound.play()
				shape2.visible(viz.OFF)
				pls_break = 1
			else:
				for idx, position_irrel in enumerate(tr_vars.shapes_irrel):
					if checkTouch(exp.shape_positions[position_irrel],new_sample,exp.min_dist_shape):
						if idx == 0:
							shape3_sound.play()
							shape3.visible(viz.OFF)
						elif idx == 1:
							shape4_sound.play()
							shape4.visible(viz.OFF)
						tr_vars.finger_on_filler = 1
						tr_vars.finger_on_shape = position_irrel
						pls_break = 1
		if pls_break:
			break

		# break after display has been presented for 1 second
		if (viz.tick() - tr_vars.t_search_on) >= exp.max_search_time:
			break
			
		yield viz.waitTime(0.001)

	t_search_off = viz.tick()
	tr_vars.search_time = t_search_off - tr_vars.t_search_on
	
	if not tr_vars.finger_on_shape == -1:
		hand_samples[hand_samples_count] = 'on_shape'
		hand_times[hand_samples_count] = viz.tick()
		hand_samples_count += 1
		while (viz.tick() - t_search_off) < 0.25:
			new_sample = m_hand.getPosition()
			if not (new_sample == old_sample):
				hand_samples[hand_samples_count] = new_sample
				hand_times[hand_samples_count] = viz.tick()
				hand_samples_count += 1
				old_sample = new_sample
			yield viz.waitTime(0.001)
	
	
	shape1.remove()
	shape2.remove()
	shape3.remove()
	shape4.remove()
	shape1_sound.remove()
	shape2_sound.remove()
	shape3_sound.remove()
	shape4_sound.remove()
	
	hand_samples[hand_samples_count] = 'end_task'
	hand_times[hand_samples_count] = viz.tick()
	
	# add time threshold
	if tr_vars.finger_on_target and (tr_vars.search_time <= exp.reach_end_cutoff_t):
		feedback_correct_sound.play()
		feedback.message("Correct")
	elif tr_vars.finger_on_distractor or tr_vars.finger_on_filler:
		feedback_wrongshape_sound.play()
		feedback.message("Wrong shape")
	else:
		feedback_tooslow_sound.play()
		feedback.message("Too slow")
	feedback.visible(viz.ON)
	
	feedback_on = viz.tick()
	
	## LOG SHIZZLE ##
	#################
	#don't log:
	del tr_vars.dis_col
	del tr_vars.tar_col
	# split filler pos
	tr_vars.filler1_pos,tr_vars.filler2_pos = tr_vars.shapes_irrel
	del tr_vars.shapes_irrel
	
	## TRIAL & BEHAV DATA LOG ##
#	trial_attr = 'pp_nr,pp_id,session,block,trial'
	trial_string = str(exp.pp_nr) + ',' + str(exp.pp_id)+ ',' + exp.name + ',' + str(bl) + ',' + str(tr)
	# add all attr from tr_vars
	for attr, value in tr_vars.__dict__.iteritems():
		if not attr.startswith('__'):
#			trial_attr += ',' + str(attr)
			trial_string += ',' + str(value)
	# add relevant attr from exp
	for attr in exp.tolog_exp_vars:
#		trial_attr += ',' + str(attr)
		trial_string += ',' + str(getattr(exp,attr))
	trial_string += '\n'
	with open(exp.data_log_file, 'a') as f:
		f.write(trial_string)
		
#	print(trial_attr)
	
	## HAND DATA LOG ##
	if not len(hand_samples) == len(hand_times):
		raise ValueError('nr of hand_samples not same as nr of hand times')
	with open(exp.data_hand_file, 'a') as f:
		f.write('{0} start_trial {1}\n'.format(tr_vars.t_trial_start,tr))
		f.write('COORD start {0}\n'.format(' '.join(map(str, exp.start_pos)))) 
		f.write('COORD target {0}\n'.format(' '.join(map(str, exp.shape_positions[tr_vars.tar_pos])))) 
		f.write('COORD distractor {0}\n'.format(' '.join(map(str, exp.shape_positions[tr_vars.dis_pos]))))
		f.write('COORD filler1 {0}\n'.format(' '.join(map(str, exp.shape_positions[tr_vars.filler1_pos]))))
		f.write('COORD filler2 {0}\n'.format(' '.join(map(str, exp.shape_positions[tr_vars.filler2_pos]))))
		f.write('VAR pp_nr {0}\n'.format(exp.pp_nr))
		f.write('VAR pp_id {0}\n'.format(exp.pp_id))
		f.write('VAR block_nr {0}\n'.format(bl))
		f.write('VAR tar_pos {0}\n'.format(tr_vars.tar_pos))
		f.write('VAR dis_pos {0}\n'.format(tr_vars.dis_pos))
		f.write('VAR dis_pos_rel {0}\n'.format(tr_vars.dis_pos_rel))
		f.write('VAR dis_pos_cont {0}\n'.format(tr_vars.dis_pos_cont))
		f.write('VAR tar_col {0}\n'.format(tr_vars.tar_col_name))
		f.write('VAR dis_condition {0}\n'.format(tr_vars.dis_condition))
		f.write('VAR tar_shape {0}\n'.format(tr_vars.tar_shape))
		f.write('VAR exp_type {0}\n'.format(exp.name))
		for xd, sample in enumerate(hand_samples_check):
			if isinstance(sample, str):
				f.write('MSG {0} {1}\n'.format(hand_times_check[xd], sample))
			else:
				f.write('{0} {1}\n'.format(hand_times_check[xd], ' '.join(map(str, sample))))
		for xd, sample in enumerate(hand_samples):
			if isinstance(sample, float) and isnan(sample):
#				print('break on hand_sample nr {}'.format(xd))
				break
			elif isinstance(sample, str):
				f.write('MSG {0} {1}\n'.format(hand_times[xd], sample))
			else:
				f.write('{0} {1}\n'.format(hand_times[xd], ' '.join(map(str, sample))))
	
	# update variable time cutoff
	exp.reach_end_cutoff_list = exp.reach_end_cutoff_list[1:] + [tr_vars.search_time]
	exp.reach_end_cutoff_t = np.percentile(exp.reach_end_cutoff_list,80)
	
	yield viztask.waitTime(1 - (viz.tick()- feedback_on))
	feedback.remove()
	
	feedback_correct_sound.stop()
	feedback_wrongshape_sound.stop()
	feedback_tooslow_sound.stop()
		

def experiment():
	text_line1 = create_text2d('Please Wait')
#	text_line2 = create_text2d('',[0,1.8,4])
	text_line1.visible(viz.OFF)	
	
	#GET PP INFO
	yield participantInfo()
	
	yield viz.waitTime(0.5)
	
	# recalibrate hand to physical start position
	yield calibrateHand()
	
	#INITIALIZE LOG FILES
	# file name incl pp nr and id + number based on time so to avoid accidental overwriting
	rd_vers = str(int(time.time()))
	# initialize and head data log file
	exp.data_log_file = '{0}/data_log_{1}{2}_{3}.csv'.format(exp.data_path, exp.pp_nr, exp.pp_id, rd_vers)
	pr.data_log_file = exp.data_log_file
	with open(exp.data_log_file, 'w') as f:
		f.write(exp.log_heading)
	
	# initialize data hand log file
	exp.data_hand_file = '{0}/data_hand_{1}{2}_{3}.csv'.format(exp.data_path, exp.pp_nr, exp.pp_id, rd_vers)
	pr.data_hand_file = exp.data_hand_file
	with open(exp.data_hand_file, 'w') as f:
		f.write('hand data file {0} {1} {2}\n'.format(exp.pp_nr, exp.pp_id, rd_vers))
	
	# write exp settings. Seperator = ':'
	exp_settings_write = ''
	for attr, value in exp.__dict__.iteritems():
		if not attr.startswith('__'):
			exp_settings_write += str(attr) + ':' + str(value) + '\n'
	with open('{0}/exp_settings_{1}{2}_{3}.csv'.format(exp.data_path, exp.pp_nr, exp.pp_id, rd_vers), 'w') as f:
		f.write(exp_settings_write)
	del exp_settings_write
	
	
	# PRACTICE BLOCKs
	for bl in range(pr.nr_blocks):
		text_line1.message("To start training block " + str(bl+1) + "\npress -0-")
		text_line1.visible(viz.ON)
		yield viztask.waitKeyDown(viz.KEY_KP_0)
		text_line1.message("3")
		yield viztask.waitTime(1)
		text_line1.message("2")
		yield viztask.waitTime(1)
		text_line1.message("1")
		yield viztask.waitTime(1)
		text_line1.visible(viz.OFF)
		
		for tr in range(pr.nr_trials/pr.nr_blocks):
			tr += bl*pr.nr_trials/pr.nr_blocks
			yield trial(bl,tr,pr,log_pr)
			
		text_line1.message("Training block " + str(bl+1) + ' of ' + str(pr.nr_blocks) + ' finished\nCall the experimenter...')
		text_line1.visible(viz.ON)
		yield viztask.waitKeyDown('a')
		
	# EXPERIMENTAL BLOCKs
	exp.reach_end_cutoff_list = pr.reach_end_cutoff_list
	exp.reach_end_cutoff_t = pr.reach_end_cutoff_t
	for bl in range(exp.nr_blocks):
		
		text_line1.message("To start experiment block " + str(bl+1) + "\npress -0-")
		text_line1.visible(viz.ON)
		yield viztask.waitTime(0.25)
		yield viztask.waitKeyDown(viz.KEY_KP_0)
		text_line1.message("3")
		yield viztask.waitTime(1)
		text_line1.message("2")
		yield viztask.waitTime(1)
		text_line1.message("1")
		yield viztask.waitTime(1)
		text_line1.visible(viz.OFF)
		
		for tr in range(exp.nr_trials/exp.nr_blocks):
			tr += bl*exp.nr_trials/exp.nr_blocks
			yield trial(bl,tr,exp,log)
		
		if bl == ((exp.nr_blocks/2)-1):
			text_line1.message("Experiment block " + str(bl+1) + ' of ' + str(exp.nr_blocks) + ' finished\nPlease call the experimenter...')
			text_line1.visible(viz.ON)
			yield viztask.waitKeyDown('a')
		elif bl == (exp.nr_blocks -1):
			text_line1.message('The end\nThank you!\nPlease call the experimenter...')
			text_line1.visible(viz.ON)
			yield viztask.waitKeyDown('a')
		else:
			text_line1.message("Experiment block " + str(bl+1) + ' of ' + str(exp.nr_blocks) + ' finished\nYou can take a break now\npress -0- to continue')
			text_line1.visible(viz.ON)
			yield viztask.waitKeyDown(viz.KEY_KP_0)
			
	
	viz.quit()
			
############
### MAIN ###
############

if not os.path.exists('data'):
    os.makedirs('data')

	## SETUP ##
	###########
exp = settings_init('experiment')
pr = settings_init('practice')
log = logger_init()
log_pr = logger_init()

for n in range(exp.nr_blocks):
	randomization(exp,log)
for n in range(pr.nr_blocks):
	randomization(pr,log_pr)
	
hmd = steamvr.HMD()
viz.link(hmd.getSensor(), viz.MainView, offset=[0,0,.392])

set_graphics_quality()
viz.go()

	### SETUP ENVIRONMENT ###
	#########################
	
# set viewpoint (and shapes)
#viz.MainView.setPosition([0,0,1.4])

# Create directional lights
light1 = vizfx.addDirectionalLight(euler=(40,20,0), color=[0.9,0.9,0.9])
light2 = vizfx.addDirectionalLight(euler=(-65,15,0), color=[0.9,0.9,0.9])

# more lights
mylight = viz.addLight()
mylight.enable()
mylight.position(0, 1, 0)
mylight.spread(180)
mylight.intensity(0.5)

# disable headlight
headLight = viz.MainView.getHeadLight()
headLight.disable()

# environment #

x = [0,-5,0,5]
z = [5,0,-5,0]
rot = [0,90,0,90]
for n in range(4):
			wall = vizshape.addBox(size=(10.0,3.0,1.0),
				right=True,left=True,
				top=True,bottom=True,
				front=True,back=True,
				splitFaces=False)
			wall.setEuler(rot[n],0,0)
			wall.setPosition(x[n],1.5,z[n])
			wall.color(0.85,0.85,0.85)
			
roof = vizshape.addBox(size=(10.0,1.0,10.0),
		right=True,left=True,
		top=True,bottom=True,
		front=True,back=True,
		splitFaces=False)
roof.setPosition(0,3.5,0)

floor = vizshape.addBox(size=(10.0,1.0,10.0),
		right=True,left=True,
		top=True,bottom=True,
		front=True,back=True,
		splitFaces=False)
floor.setPosition(0,-0.5,0)
floor.color(0.85,0.85,0.85)

table = vizshape.addBox(size=(1.2,0.85,1.0),
		right=True,left=True,
		top=True,bottom=True,
		front=True,back=True,
		splitFaces=False)
table.setPosition(0,0.425,0.6)

startbutton = vizshape.addBox(size=(0.02,0.006,0.02),
		right=True,left=True,
		top=True,bottom=True,
		front=True,back=True,
		splitFaces=False)
startbutton.setPosition(list(np.array(exp.start_pos) - np.array([0.0,0.0,0.006])))
startbutton.color([0,0,0])

# Connect to Optotrak (don't forget to run FirstPrinciples!)
exp.optotrak = None
exp.optotrak = viz.add('optotrak.dle', 0, exp.cf_optoserver)

if exp.optotrak:
	print("Optotrak connected.")

	if exp.cf_DEBUG:
		print("Available markers:")
		for i, m in enumerate(exp.optotrak.getMarkerList()):
			print("{}: {}".format(i, m.getName()))

	# Create filters to correct Optotrak coordinate system and offset
	optofilter = viz.add('filter.dle')
	hand_marker = exp.optotrak.getMarker(exp.cf_optomarker)
	hand0 = optofilter.swap(hand_marker, pos=exp.cf_optofilter)
	base_offset = [ 0.01875045, 0.0007511499999999999,  0.0691429 ]
	hand = optofilter.position(hand0, offset=(-exp.cf_tablewidth/2 + exp.cf_optozero_x + base_offset[0],
											  exp.cf_tableheight + exp.cf_optozero_y + base_offset[1],
											  exp.cf_tabledepth/2 + exp.cf_optozero_z + base_offset[2]))

else:
	print("Could not connect to Optotrak! Motion tracking and recording disabled.")

print("Initialization done.")

# Hand marker (green sphere)
m_hand = vizshape.addSphere(
    radius=exp.cf_hand_size, color=[0.5,1,0.5], scene=viz.Scene1)
m_hand.disable(viz.INTERSECTION)
if exp.optotrak:
    optoLink = viz.link(hand, m_hand)

#tijd = viz.tick()
#check_position = m_hand.getPosition()
#position_offset = np.array(exp.start_pos) - np.array(check_position)
#print position_offset
#
#hand2 = optofilter.position(hand0, offset=(-exp.cf_tablewidth/2 + exp.cf_optozero_x + position_offset[0],
#											  exp.cf_tableheight + exp.cf_optozero_y + position_offset[1] + 0.006,
#											  exp.cf_tabledepth/2 + exp.cf_optozero_z + position_offset[2]))
#if exp.optotrak:
#    optoLink = viz.link(hand2, m_hand)

# Initialize feedback sounds
sound_node = viz.addGroup(pos = [0,1.2,0])
feedback_correct_sound = sound_node.playsound('./sounds/woo-hoo.wav')
feedback_correct_sound.pause()
feedback_wrongshape_sound = sound_node.playsound('./sounds/gasp.wav')
feedback_wrongshape_sound.pause()
feedback_tooslow_sound = sound_node.playsound('./sounds/aww.wav')
feedback_tooslow_sound.pause()

	### EXPERIMENT ###
	##################

viztask.schedule(experiment)