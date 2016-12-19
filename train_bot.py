import datetime
import os
import sys

import numpy as np
import json

from keras.models import Sequential,load_model,Model
from keras.layers import Dense,Highway,Reshape, Input,Dropout, merge
from keras.layers.convolutional import Convolution2D;
from keras.layers.core import Flatten, Reshape, Activation, Lambda;
from keras.engine.topology import Merge;
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping,ModelCheckpoint,TensorBoard,ReduceLROnPlateau
from keras.optimizers import Adadelta;
from common import stack_to_input, VISIBLE_DISTANCE, num_data_points, cone_dim, input_dim
import tensorflow as tf
import keras.backend as K

REPLAY_FOLDER = sys.argv[1]
training_input = []
training_target = []

np.random.seed(0) # for reproducibility

final_layer_neurons = 10;

def new_cone_model():
	cone_model = Sequential([
		Dense(512, input_dim=input_dim),
		Dropout(0.2),
		LeakyReLU(),
		Dense(200),
		Dense(128),
		Dense(128),
		LeakyReLU(),
		Dense(final_layer_neurons),
		LeakyReLU(),
	])
	return cone_model

north_input = Input(shape=(input_dim,));
south_input = Input(shape=(input_dim,));
west_input = Input(shape=(input_dim,));
east_input = Input(shape=(input_dim,));
stay_input = Input(shape=(1,));

north_cone = new_cone_model()(north_input);
south_cone = new_cone_model()(south_input);
west_cone = new_cone_model()(west_input);
east_cone = new_cone_model()(east_input);


merged_vector = merge([stay_input, north_cone, south_cone, west_cone, east_cone], mode="concat", concat_axis=1)

predictions = Sequential([
	Dense(10, input_shape=(final_layer_neurons*4 + 1,)),
	Dense(5),
	Activation('softmax'),
])

predictions = predictions(merged_vector);

#predictions = Lambda(my_lambda, output_shape=(5,))(merged_vector);


model = Model(input=[stay_input, north_input, south_input, west_input, east_input], output=predictions);

#model = Sequential([
	#Merge([north_cone, south_cone, west_cone, east_cone], mode="concat", concat_axis=1),
	#Highway(input_dim=input_dim),
	#Reshape((2*VISIBLE_DISTANCE+1, 2*VISIBLE_DISTANCE+1, num_data_points), input_shape=(input_dim,)),
	#Convolution2D(64, 3, 3, border_mode='same'),
	#Flatten(),
#	Dense(5, activation='softmax')])
opt = Adadelta();
model.compile(optimizer=opt,loss='categorical_crossentropy', metrics=['accuracy'])

size = len(os.listdir(REPLAY_FOLDER))    
for index, replay_name in enumerate(os.listdir(REPLAY_FOLDER)):
    if replay_name[-4:]!='.hlt':continue
    print('Loading {} ({}/{})'.format(replay_name, index, size))
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

    position_indices = stacks[:,0].nonzero()
    sampling_rate = 1/stacks[:,0].mean(axis=(1,2))[position_indices[0]]
    sampling_rate *= moves[position_indices].dot(np.array([1,10,10,10,10])) # weight moves 10 times higher than still
    sampling_rate /= sampling_rate.sum()
    sample_indices = np.transpose(position_indices)[np.random.choice(np.arange(len(sampling_rate)),
                                                                    min(len(sampling_rate),2048),p=sampling_rate,replace=False)]

    replay_input = np.array([stack_to_input(stacks[i],[j,k]) for i,j,k in sample_indices])
    replay_target = moves[tuple(sample_indices.T)]

    #print("REPLAY INPUT: ", replay_input);   
 
    training_input.append(replay_input.astype(np.float32))
    training_target.append(replay_target.astype(np.float32))
    print("PRE-FLATTEN SHAPE: ", replay_input.shape);

#training_target = np.swapaxes(np.array(training_target), 0, 1);
#print("TRAINING TARGET SHAPE: ", training_target.shape);


now = datetime.datetime.now()
tensorboard = TensorBoard(log_dir='./logs/'+now.strftime('%Y.%m.%d %H.%M'))
training_input = np.concatenate(training_input,axis=0)
training_target = np.concatenate(training_target,axis=0)
indices = np.arange(len(training_input))
np.random.shuffle(indices) #shuffle training samples
training_input = training_input[indices]
training_target = training_target[indices]

training_input = np.swapaxes(training_input, 0, 1);


print("TRAINING SHAPE: ", training_input.shape);

model.fit(
	[
		np.repeat([1.0], training_input[0].shape[0]),
		training_input[0],
		training_input[1],
		training_input[2],
		training_input[3],
	],
	training_target,
	validation_split=0.2,
	callbacks=[
		EarlyStopping(patience=25),
		ReduceLROnPlateau(patience=10),
		ModelCheckpoint('model.h5',verbose=1,save_best_only=True),
		tensorboard
	],
	batch_size=1024,
	nb_epoch=1000
)

model = load_model('model.h5')

still_mask = training_target[:,0].astype(bool)
print('STILL accuracy:',model.evaluate(
	[
		np.repeat([1.0], training_input[0][still_mask].shape[0]),
		training_input[0][still_mask],
		training_input[1][still_mask],
		training_input[2][still_mask],
		training_input[3][still_mask],
	],
	training_target[still_mask],verbose=0)[1]
)
print('MOVE accuracy:',model.evaluate(
	[
		np.repeat([0.5], training_input[0][~still_mask].shape[0]),
		training_input[0][~still_mask],
		training_input[1][~still_mask],
		training_input[2][~still_mask],
		training_input[3][~still_mask],
	],
	training_target[~still_mask],verbose=0)[1]
)
