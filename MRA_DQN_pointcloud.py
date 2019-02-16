import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2
import random
import datetime
import os
import time

from unityagents import UnityEnvironment

#matplotlib inline

env_name = "/home/weiy/manage_code/DRL_based_SelfDrivingCarControl/RL_algorithms/unityagents/jeju_camp.x86_64" # Name of the Unity environment to launch
train_mode = True # Whether to run the environment in training or inference mode

env = UnityEnvironment(file_name=env_name)

# Examine environment parameters
print(str(env))

# Set the default brain to work with
default_brain = env.brain_names[0]
brain = env.brains[default_brain]


# Reset the environment
env_info = env.reset(train_mode=train_mode)[default_brain]

# Examine the state space for the default brain
print("Sensor data (LIDAR): \n{}".format(env_info.vector_observations[0]))

# Examine the observation space for the default brain
Num_obs = len(env_info.visual_observations)



algorithm = 'DQN'
Num_action = brain.vector_action_space_size

# parameter for DQN
Num_replay_memory = 100000
Num_start_training = 50000
Num_training = 1000000
Num_update = 10000
Num_batch = 32
Num_test = 100000
Num_skipFrame = 4
Num_stackFrame = 4
Num_colorChannel = 1

Epsilon = 1.0
Final_epsilon = 0.1
Gamma = 0.99
Learning_rate = 0.00025

# Parameter for LSTM
Num_dataSize = 366
Num_cellState = 512

# Parameters for network
img_size = 80
sensor_size = 360

first_dense  = [Num_cellState, 512]
second_dense = [first_dense[1], Num_action]

# add by Yuanwei 2019-1-14 testing HRA
first_dense_speed  = [Num_cellState, 512]
second_dense_speed = [first_dense_speed[1], Num_action]

first_dense_overtake  = [Num_cellState, 512]
second_dense_overtake = [first_dense_overtake[1], Num_action]

first_dense_lanechange  = [Num_cellState, 512]
second_dense_lanechange = [first_dense_lanechange[1], Num_action]
# end by Yuanwei 2019-1-14


# Path of the network model
load_path = ''

# Parameters for session
Num_plot_episode = 5
Num_step_save = 50000

GPU_fraction = 0.1

# Initialize weights and bias
def weight_variable(shape):
    return tf.Variable(xavier_initializer(shape))

def bias_variable(shape):
	return tf.Variable(xavier_initializer(shape))

# Xavier Weights initializer
def xavier_initializer(shape):
	dim_sum = np.sum(shape)
	if len(shape) == 1:
		dim_sum += 1
	bound = np.sqrt(2.0 / dim_sum)
	return tf.random_uniform(shape, minval=-bound, maxval=bound)

# Assign network variables to target network
def assign_network_to_target():
	# Get trainable variables
	trainable_variables = tf.trainable_variables()
	# network lstm variables
	trainable_variables_network = [var for var in trainable_variables if var.name.startswith('network')]

	# target lstm variables
	trainable_variables_target = [var for var in trainable_variables if var.name.startswith('target')]

    # assign network variables to target network
	for i in range(len(trainable_variables_network)):
		sess.run(tf.assign(trainable_variables_target[i], trainable_variables_network[i]))

# Code for tensorboard
def setup_summary():
    episode_speed      = tf.Variable(0.)
    episode_overtake   = tf.Variable(0.)
    episode_lanechange = tf.Variable(0.)

    tf.summary.scalar('Average_Speed/' + str(Num_plot_episode) + 'episodes', episode_speed)
    tf.summary.scalar('Average_overtake/' + str(Num_plot_episode) + 'episodes', episode_overtake)
    tf.summary.scalar('Average_lanechange/' + str(Num_plot_episode) + 'episodes', episode_lanechange)

    summary_vars = [episode_speed, episode_overtake, episode_lanechange]
    summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
    update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
    summary_op = tf.summary.merge_all()
    return summary_placeholders, update_ops, summary_op




tf.reset_default_graph()

# Input
x_image = tf.placeholder(tf.float32, shape = [None, img_size, img_size, Num_colorChannel * Num_stackFrame * Num_obs])
x_normalize = (x_image - (255.0/2)) / (255.0/2)

x_sensor = tf.placeholder(tf.float32, shape = [None, Num_stackFrame, Num_dataSize])
x_unstack = tf.unstack(x_sensor, axis = 1)

with tf.variable_scope('network'):  
    # LSTM cell
    cell = tf.contrib.rnn.BasicLSTMCell(num_units = Num_cellState)            
    rnn_out, rnn_state = tf.nn.static_rnn(inputs = x_unstack, cell = cell, dtype = tf.float32)
    
    # Densely connect layer variables
    w_fc1 = weight_variable(first_dense)
    b_fc1 = bias_variable([first_dense[1]])

    w_fc2 = weight_variable(second_dense)
    b_fc2 = bias_variable([second_dense[1]])

    # add by Yuanwei 2019-1-14 testing HRA
    w_fc1_speed = weight_variable(first_dense_speed)
    b_fc1_speed = bias_variable([first_dense_speed[1]])

    w_fc2_speed = weight_variable(second_dense_speed)
    b_fc2_speed = bias_variable([second_dense_speed[1]])
    
    w_fc1_overtake = weight_variable(first_dense_overtake)
    b_fc1_overtake = bias_variable([first_dense_overtake[1]])

    w_fc2_overtake = weight_variable(second_dense_overtake)
    b_fc2_overtake = bias_variable([second_dense_overtake[1]])
    
    w_fc1_lanechange = weight_variable(first_dense_lanechange)
    b_fc1_lanechange = bias_variable([first_dense_lanechange[1]])

    w_fc2_lanechange = weight_variable(second_dense_lanechange)
    b_fc2_lanechange = bias_variable([second_dense_lanechange[1]])
    # end by Yuanwei 2019-1-14
    
# Network
rnn_out = rnn_out[-1]

h_fc1 = tf.nn.relu(tf.matmul(rnn_out, w_fc1)+b_fc1)
output = tf.matmul(h_fc1,  w_fc2)+b_fc2

# add by Yuanwei 2019-1-14 testing HRA
h_fc1_speed = tf.nn.relu(tf.matmul(rnn_out, w_fc1_speed)+b_fc1_speed)
output_speed = tf.matmul(h_fc1_speed,  w_fc2_speed)+b_fc2_speed

h_fc1_overtake = tf.nn.relu(tf.matmul(rnn_out, w_fc1_overtake)+b_fc1_overtake)
output_overtake = tf.matmul(h_fc1_overtake,  w_fc2_overtake)+b_fc2_overtake

h_fc1_lanechange = tf.nn.relu(tf.matmul(rnn_out, w_fc1_lanechange)+b_fc1_lanechange)
output_lanechange = tf.matmul(h_fc1_lanechange,  w_fc2_lanechange)+b_fc2_lanechange
# end by Yuanwei 2019-1-14

with tf.variable_scope('target'):
    # LSTM cell
    cell_target = tf.contrib.rnn.BasicLSTMCell(num_units = Num_cellState)            
    rnn_out_target, rnn_state_target = tf.nn.static_rnn(inputs = x_unstack, cell = cell_target, dtype = tf.float32)
    
    # Densely connect layer variables target
    w_fc1_target = weight_variable(first_dense)
    b_fc1_target = bias_variable([first_dense[1]])

    w_fc2_target = weight_variable(second_dense)
    b_fc2_target = bias_variable([second_dense[1]])

    # add by Yuanwei 2019-1-14 testing HRA
    w_fc1_target_speed = weight_variable(first_dense_speed)
    b_fc1_target_speed = bias_variable([first_dense_speed[1]])

    w_fc2_target_speed = weight_variable(second_dense_speed)
    b_fc2_target_speed = bias_variable([second_dense_speed[1]])
    
    w_fc1_target_overtake = weight_variable(first_dense_overtake)
    b_fc1_target_overtake = bias_variable([first_dense_overtake[1]])

    w_fc2_target_overtake = weight_variable(second_dense_overtake)
    b_fc2_target_overtake = bias_variable([second_dense_overtake[1]])
    
    w_fc1_target_lanechange = weight_variable(first_dense_lanechange)
    b_fc1_target_lanechange = bias_variable([first_dense_lanechange[1]])

    w_fc2_target_lanechange = weight_variable(second_dense_lanechange)
    b_fc2_target_lanechange = bias_variable([second_dense_lanechange[1]])
    # end by Yuanwei 2019-1-14

    
# Target Network
rnn_out_target = rnn_out_target[-1]

h_fc1_target  = tf.nn.relu(tf.matmul(rnn_out_target, w_fc1_target)+b_fc1_target)
output_target = tf.matmul(h_fc1_target, w_fc2_target)+b_fc2_target

# add by Yuanwei 2019-1-14 testing HRA
h_fc1_target_speed = tf.nn.relu(tf.matmul(rnn_out_target, w_fc1_target_speed)+b_fc1_target_speed)
output_target_speed = tf.matmul(h_fc1_target_speed,  w_fc2_target_speed)+b_fc2_target_speed

h_fc1_target_overtake = tf.nn.relu(tf.matmul(rnn_out_target, w_fc1_target_overtake)+b_fc1_target_overtake)
output_target_overtake = tf.matmul(h_fc1_target_overtake,  w_fc2_target_overtake)+b_fc2_target_overtake

h_fc1_target_lanechange = tf.nn.relu(tf.matmul(rnn_out_target, w_fc1_target_lanechange)+b_fc1_target_lanechange)
output_target_lanechange = tf.matmul(h_fc1_target_lanechange,  w_fc2_target_lanechange)+b_fc2_target_lanechange
# end by Yuanwei 2019-1-14


# Loss function and Train
action_target = tf.placeholder(tf.float32, shape = [None, Num_action])
y_target = tf.placeholder(tf.float32, shape = [None])

# add by Yuanwei 2019-1-14 testing HRA
action_target_speed = tf.placeholder(tf.float32, shape = [None, Num_action])
y_target_speed = tf.placeholder(tf.float32, shape = [None])

action_target_overtake = tf.placeholder(tf.float32, shape = [None, Num_action])
y_target_overtake = tf.placeholder(tf.float32, shape = [None])

action_target_lanechange = tf.placeholder(tf.float32, shape = [None, Num_action])
y_target_lanechange = tf.placeholder(tf.float32, shape = [None])
# end by Yuanwei 2019-1-14

y_prediction = tf.reduce_sum(tf.multiply(output, action_target), reduction_indices = 1)

# add by Yuanwei 2018-1-15 testing HRA
y_prediction_speed = tf.reduce_sum(tf.multiply(output_speed, action_target_speed), reduction_indices = 1)
y_prediction_overtake = tf.reduce_sum(tf.multiply(output_overtake, action_target_overtake), reduction_indices = 1)
y_prediction_lanechange = tf.reduce_sum(tf.multiply(output_lanechange, action_target_lanechange), reduction_indices = 1)
# end by Yuanwei 2019-1-15

# modified by Yanwei 2019-1-15 testing HRA

#Loss = tf.reduce_mean(tf.square(y_prediction - y_target))

Loss = tf.reduce_mean(tf.square(y_prediction_speed - y_target_speed))          \
        + tf.reduce_mean(tf.square(y_prediction_overtake - y_target_overtake)) \
        + tf.reduce_mean(tf.square(y_prediction_lanechange - y_target_lanechange)) 

# end by Yuanwei 2019-1-15

train_step = tf.train.AdamOptimizer(learning_rate = Learning_rate, epsilon = 1e-02).minimize(Loss)

## Initialize variables
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = GPU_fraction

sess = tf.InteractiveSession(config=config)

init = tf.global_variables_initializer()
sess.run(init)


# Load the file if the saved file exists
saver = tf.train.Saver()

# check_save = 1
check_save = input('Inference? / Training?(1=Inference/2=Training): ')

if check_save == '1':
    # Directly start inference
    Num_start_training = 0
    Num_training = 0
    
    # Restore variables from disk.
    saver.restore(sess, load_path)
    print("Model restored.")

# date - hour - minute of training time
date_time = str(datetime.date.today()) + '_' + str(datetime.datetime.now().hour) + '_' + str(datetime.datetime.now().minute)

# Make folder for save data
os.makedirs('../saved_networks/' + date_time + '_' + algorithm + '_sensor')

# Summary for tensorboard
summary_placeholders, update_ops, summary_op = setup_summary()
summary_writer = tf.summary.FileWriter('../saved_networks/' + date_time + '_' + algorithm + '_sensor', sess.graph)




# Initialize input 
def state_initialization(env_info):    
    state = env_info.vector_observations[0][:-7]

    state_set = []

    for i in range(Num_skipFrame * Num_stackFrame):
        state_set.append(state)
    
    # Stack the frame according to the number of skipping and stacking frames using observation set
    state_stack = np.zeros((Num_stackFrame, Num_dataSize))
    for stack_frame in range(Num_stackFrame):
        state_stack[(Num_stackFrame - 1) - stack_frame, :] = state_set[-1 - (Num_skipFrame * stack_frame)]

    state_stack = np.uint8(state_stack)

    return state_stack, state_set

# Resize input information 
def resize_state(env_info, state_set):
    state = env_info.vector_observations[0][:-7]

    # Add state to the state_set
    state_set.append(state)
    
    # Stack the frame according to the number of skipping and stacking frames using observation set
    state_stack = np.zeros((Num_stackFrame, Num_dataSize))
    for stack_frame in range(Num_stackFrame):
        state_stack[(Num_stackFrame - 1) - stack_frame, :] = state_set[-1 - (Num_skipFrame * stack_frame)]
        
    del state_set[0]

    state_stack = np.uint8(state_stack)
    
    return state_stack, state_set

# Get progress according to the number of steps
def get_progress(step, Epsilon):
    if step <= Num_start_training:
        # Observation
        progress = 'Observing'
        train_mode = True
        Epsilon = 1
    elif step <= Num_start_training + Num_training:
        # Training
        progress = 'Training'
        train_mode = True
        
        # Decrease the epsilon value
        if Epsilon > Final_epsilon:
            Epsilon -= 1.0/Num_training
    elif step < Num_start_training + Num_training + Num_test:
        # Testing
        progress = 'Testing'
        train_mode = False
        Epsilon = 0
    else:
        # Finished
        progress = 'Finished'
        train_mode = False
        Epsilon = 0
        
    return progress, train_mode, Epsilon 

# Select action according to the progress of training
def select_action(progress, sess, state_stack, Epsilon):
    if progress == "Observing":
        # Random action 
        Q_value = 0

        # add by Yuanwei 2019-1-15
        Q_value_speed = 0
        Q_value_overtake = 0
        Q_value_lanechange = 0
        # end by Yuanwei 2019-1-15

        action = np.zeros([Num_action])
        action[random.randint(0, Num_action - 1)] = 1.0
    elif progress == "Training":
        # if random value(0-1) is smaller than Epsilon, action is random. 
        # Otherwise, action is the one which has the max Q value
        if random.random() < Epsilon:
            Q_value = 0
            action = np.zeros([Num_action])
            action[random.randint(0, Num_action - 1)] = 1
        else:
            # Commented by Yuanwei 2019-1-15
            #Q_value = output.eval(feed_dict={x_sensor: [state_stack]})
            # end by Yuanwei 2019-1-15

            # add by Yuanwei 2019-1-15
            Q_value_speed = output_speed.eval(feed_dict={x_sensor: [state_stack]})
            Q_value_overtake = output_overtake.eval(feed_dict={x_sensor: [state_stack]})
            Q_value_lanechange = output_lanechange.eval(feed_dict={x_sensor: [state_stack]})

            Q_value = 0.33*Q_value_speed + 0.33*Q_value_overtake + 0.33*Q_value_lanechange
            # end by Yuanwei 2019-1-15

            action = np.zeros([Num_action])
            action[np.argmax(Q_value)] = 1
    else:
        # Max Q action 
        # Commented by Yuanwei 2019-1-15
        Q_value = output.eval(feed_dict={x_sensor: [state_stack]})
        # end by Yuanwei 2019-1-15

        # add by Yuanwei 2019-1-15
        Q_value_speed = output_speed.eval(feed_dict={x_sensor: [state_stack]})
        Q_value_overtake = output_overtake.eval(feed_dict={x_sensor: [state_stack]})
        Q_value_lanechange = output_lanechange.eval(feed_dict={x_sensor: [state_stack]})

        Q_value = 0.33*Q_value_speed + 0.33*Q_value_overtake + 0.33*Q_value_lanechange
        # end by Yuanwei 2019-1-15

        action = np.zeros([Num_action])
        action[np.argmax(Q_value)] = 1
        
    return action, Q_value

def train(Replay_memory, sess, step):
    # Select minibatch
    minibatch =  random.sample(Replay_memory, Num_batch)

    # Save the each batch data
    state_batch      = [batch[0] for batch in minibatch]
    action_batch     = [batch[1] for batch in minibatch]
    reward_batch     = [batch[2] for batch in minibatch]
    state_next_batch = [batch[3] for batch in minibatch]
    terminal_batch 	 = [batch[4] for batch in minibatch]

    # Update target network according to the Num_update value
    if step % Num_update == 0:
        assign_network_to_target()
    
    # Get y_target
    y_batch = []
    Q_target = output_target.eval(feed_dict = {x_sensor: state_next_batch})

    # add by Yuanwei 2019-1-15 testing HRA
    y_batch_speed = []
    Q_target_speed = output_target_speed.eval(feed_dict = {x_sensor: state_next_batch})
    
    y_batch_overtake = []
    Q_target_overtake = output_target_overtake.eval(feed_dict = {x_sensor: state_next_batch})
    
    y_batch_lanechange = []
    Q_target_lanechange = output_target_lanechange.eval(feed_dict = {x_sensor: state_next_batch})
    # end by Yuanwei 2019-1-15

    # Get target values
    for i in range(len(minibatch)):
        if terminal_batch[i] == True:
            y_batch.append(reward_batch[i])

            # add by Yuanwei 2019-1-15 testing HRA
            y_batch_speed.append(reward_batch[i])
            
            y_batch_overtake.append(reward_batch[i])
            
            y_batch_lanechange.append(reward_batch[i])
            # end by Yuanwei 2019-1-15

        else:
            y_batch.append(reward_batch[i] + Gamma * np.max(Q_target[i]))

            # add by Yuanwei 2018-1-15 testing HRA
            y_batch_speed.append(reward_batch[i] + Gamma * np.max(Q_target_speed[i]))
            y_batch_overtake.append(reward_batch[i] + Gamma * np.max(Q_target_overtake[i]))
            y_batch_lanechange.append(reward_batch[i] + Gamma * np.max(Q_target_lanechange[i]))
            # end by Yuanwei 2019-1-15

    # modified by Yuanwei 2019-1-15 testing HRA
    _, loss = sess.run([train_step, Loss], feed_dict = {#action_target: action_batch,
                                                        action_target_speed: action_batch, 
                                                        action_target_overtake: action_batch,
                                                        action_target_lanechange: action_batch, 
                                                        #y_target: y_batch,
                                                        y_target_speed: y_batch_speed,
                                                        y_target_overtake: y_batch_overtake,
                                                        y_target_lanechange: y_batch_lanechange, 
                                                        x_sensor: state_batch})
    # end by Yuawei 2019-1-15
# Experience Replay 
def Experience_Replay(progress, Replay_memory, state_stack, action, reward, next_state_stack, terminal):
    if progress != 'Testing':
        # If length of replay memeory is more than the setting value then remove the first one
        if len(Replay_memory) > Num_replay_memory:
            del Replay_memory[0]

        # Save experience to the Replay memory
        Replay_memory.append([state_stack, action, reward, next_state_stack, terminal])
    else:
        # Empty the replay memory if testing
        Replay_memory = []
    
    return Replay_memory



# Initial parameters
Replay_memory = []

step = 1
score = 0
score_board = 0

episode = 0
step_per_episode = 0

speed_list = []
overtake_list = []
lanechange_list = []

train_mode = True
env_info = env.reset(train_mode=train_mode)[default_brain]

state_stack, state_set = state_initialization(env_info)



check_plot = 0

# Training & Testing
while True:
   
    # Get Progress, train mode
    progress, train_mode, Epsilon  = get_progress(step, Epsilon)
    
    # Select Actions 
    action, Q_value = select_action(progress, sess, state_stack, Epsilon)
    action_in = [np.argmax(action)]
    
    # Get information for plotting
    vehicle_speed  = 100 * env_info.vector_observations[0][-8]
    num_overtake   = env_info.vector_observations[0][-7]
    num_lanechange = env_info.vector_observations[0][-6]
    
    # Get information for update
    env_info = env.step(action_in)[default_brain]

    next_state_stack, state_set = resize_state(env_info, state_set) 
    reward = env_info.rewards[0]
    terminal = env_info.local_done[0]
    
    if progress == 'Training':
        # Train!! 
        train(Replay_memory, sess, step)

        # Save the variables to disk.
        if step == Num_start_training + Num_training:
            save_path = saver.save(sess, '../saved_networks/' + date_time + '_' + algorithm + '_sensor' + "/model.ckpt")
            print("Model saved in file: %s" % save_path)
    
    # If progress is finished -> close! 
    if progress == 'Finished':
        print('Finished!!')
        env.close()
        break
        
    Replay_memory = Experience_Replay(progress, 
                                      Replay_memory, 
                                      state_stack,
                                      action, 
                                      reward, 
                                      next_state_stack,
                                      terminal)
    
    # Update information
    step += 1
    score += reward
    step_per_episode += 1
    
    state_stack = next_state_stack
    
    # Update tensorboard
    if progress != 'Observing':
        speed_list.append(vehicle_speed)
        
        if episode % Num_plot_episode == 0 and check_plot == 1 and episode != 0:
            avg_speed      = sum(speed_list) / len(speed_list)
            avg_overtake   = sum(overtake_list) / len(overtake_list)
            avg_lanechange = sum(lanechange_list) / len(lanechange_list)
            
            tensorboard_info = [avg_speed, avg_overtake, avg_lanechange]
            for i in range(len(tensorboard_info)):
                sess.run(update_ops[i], feed_dict = {summary_placeholders[i]: float(tensorboard_info[i])})
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, step)
            score_board = 0
            
            speed_list = []
            overtake_list = []
            lanechange_list = []

            check_plot = 0
            
    # If terminal is True
    if terminal == True:
        # Print informations
        print('step: ' + str(step) + ' / '  + 'episode: ' + str(episode) + ' / ' + 'progress: ' + progress  + ' / ' + 'epsilon: ' + str(Epsilon)  +' / ' + 'score: ' + str(score))

        check_plot = 1

        if progress != 'Observing':
            episode += 1
            
            score_board += score
            overtake_list.append(num_overtake)
            lanechange_list.append(num_lanechange)
        
            
        score = 0
        step_per_episode = 0

        # Initialize game state
        env_info = env.reset(train_mode=train_mode)[default_brain]
        state_stack, state_set = state_initialization(env_info)
        
env.close()





