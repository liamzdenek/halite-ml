import datetime
import os
import sys

import numpy as np
import json

from keras.models import Sequential,load_model
from keras.layers import Input,Dense,Flatten,Merge
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping,ModelCheckpoint,TensorBoard

REPLAY_FOLDER = sys.argv[1]
training_input_early = []
training_input_turn = []
training_input_late = []
training_target = []


VISIBLE_DISTANCE = 6
num_data_points=4
input_dim=num_data_points*(2*VISIBLE_DISTANCE+1)*(2*VISIBLE_DISTANCE+1)
print("INPUT DIM: ", input_dim);
np.random.seed(0) # for reproducibility




early_model = Sequential([
	Dense(512, input_dim=input_dim),
	LeakyReLU(),
	Dense(256),
	LeakyReLU(),
	Dense(128),
	LeakyReLU(),
	Dense(128)
])

input_turn = Sequential([
	Dense(1, input_dim=1),
]);

late_model = Sequential([
	Dense(512, input_dim=input_dim),
	LeakyReLU(),
	Dense(256),
	LeakyReLU(),
	Dense(128),
	LeakyReLU(),
	Dense(16)
])

late_model_merged = Sequential([
	Merge([input_turn, late_model], mode='concat', concat_axis=1),
	Dense(32),
])

model = Sequential([
	Merge([early_model, late_model_merged], mode='concat', concat_axis=1),
	Dense(128),
	LeakyReLU(),
	Dense(128),
	LeakyReLU(),
	Dense(5, activation='softmax')
])
model.compile('nadam','categorical_crossentropy', metrics=['accuracy'])

def stack_to_input(stack, position):
	return np.take(
		np.take(
			stack,
			np.arange(-VISIBLE_DISTANCE,VISIBLE_DISTANCE + 1)+position[0],
			axis=1,
			mode='wrap'
		),
		np.arange(-VISIBLE_DISTANCE,VISIBLE_DISTANCE + 1)+position[1],
		axis=2,
		mode='wrap'
	).flatten().astype(np.float32)

def stack_to_lateinput(stack, position):
	distance = 3;
	data = np.take(
		stack,
		np.arange(-VISIBLE_DISTANCE*distance,(VISIBLE_DISTANCE+1)*distance)+position[0],
		axis=1,
		mode='wrap'
	);

	data = np.take(
		data,
		np.arange(-VISIBLE_DISTANCE*distance,(VISIBLE_DISTANCE+1)*distance)+position[1],
		axis=2,
		mode='wrap'
	)


	# this and down is the new part. take the average of a distance^2 area

	data = np.mean(np.reshape(data, (4,(VISIBLE_DISTANCE*2+1)*distance, -1, 3)), axis=3);
	data = np.swapaxes(data, 1,2);
	data = np.mean(np.reshape(data, (4,(VISIBLE_DISTANCE*2+1), -1, 3)), axis=3);
	data = np.swapaxes(data, 1,2);
	data = data.flatten().astype(np.float32)
	return data;

def stack_to_turn(frame):
	return min(frame/150, 1)

for replay_name in os.listdir(REPLAY_FOLDER):
	if replay_name[-4:]!='.hlt':continue
	print('Loading {}'.format(replay_name))
	replay = json.load(open('{}/{}'.format(REPLAY_FOLDER,replay_name)))

	frames=np.array(replay['frames'])
	player=frames[:,:,:,0]
	players,counts = np.unique(player[-1],return_counts=True)
	target_id = players[counts.argmax()]
	if target_id == 0: continue

	prod = np.repeat(np.array(replay['productions'])[np.newaxis],replay['num_frames'],axis=0)
	strength = frames[:,:,:,1]

	moves = (np.arange(5) == np.array(replay['moves'])[:,:,:,None]).astype(int)[:128]
	stacks = np.array([player==target_id,(player!=target_id) & (player!=0),prod/20,strength/255])
	stacks = stacks.transpose(1,0,2,3)[:len(moves)].astype(np.float32)

	###
	### EARLY and TARGET
	###

	position_indices = stacks[:,0].nonzero()
	sampling_rate = 1/stacks[:,0].mean(axis=(1,2))[position_indices[0]]
	sampling_rate *= moves[position_indices].dot(np.array([1,10,10,10,10])) # weight moves 10 times higher than still
	sampling_rate /= sampling_rate.sum()
	sample_indices = np.transpose(position_indices)[np.random.choice(np.arange(len(sampling_rate)),
								    min(len(sampling_rate),2048),p=sampling_rate,replace=False)]

	replay_input = np.array([stack_to_input(stacks[i],[j,k]) for i,j,k in sample_indices])
	replay_target = moves[tuple(sample_indices.T)]


	training_input_early.append(replay_input.astype(np.float32))

	#target
	training_target.append(replay_target.astype(np.float32))
	
	###
	### LATE
	###	
	replay_input=np.array([stack_to_lateinput(stacks[i],[j,k]) for i,j,k in sample_indices])
	replay_target = moves[tuple(sample_indices.T)]

	training_input_late.append(replay_input.astype(np.float32))

	###
	### TURN
	###

	training_input_turn += [max(i/150, 1) for i in range(len(replay_input))];

	

now = datetime.datetime.now()
tensorboard = TensorBoard(log_dir='./logs/'+now.strftime('%Y.%m.%d %H.%M'))
training_input_early = np.concatenate(training_input_early,axis=0)
training_input_late = np.concatenate(training_input_late,axis=0)
training_target = np.concatenate(training_target,axis=0)
indices = np.arange(len(training_input_early))
np.random.shuffle(indices) #shuffle training samples
training_input_early = training_input_early[indices]
training_input_late = training_input_late[indices]
training_target = training_target[indices]

training_input_turn = np.array(training_input_turn);

print("END");
print("TRAINING EARLY:  ", training_input_early.shape);
print("TRAINING TURN:   ", training_input_turn.shape);
print("TRAINING LATE:   ", training_input_late.shape);
print("TRAINING TARGET: ", training_target.shape);

model.fit([training_input_early, training_input_turn, training_input_late],training_target,validation_split=0.2,
          callbacks=[EarlyStopping(patience=10),
                     ModelCheckpoint('model.h5',verbose=1,save_best_only=True),
                     tensorboard],
          batch_size=1024, nb_epoch=1000)

model = load_model('model.h5')

still_mask = training_target[:,0].astype(bool)
print('STILL accuracy:',model.evaluate([training_input_early[still_mask],training_input_turn[still_mask],training_input_late[still_mask]],training_target[still_mask],verbose=0)[1])
print('MOVE accuracy:',model.evaluate([training_input_early[~still_mask], training_input_turn[~still_mask], training_input_late[~still_mask]],training_target[~still_mask],verbose=0)[1])
