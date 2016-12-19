from networking import *
import os
import sys
import numpy as np
from common import stack_to_input, VISIBLE_DISTANCE, cone_dim, input_dim


def predict(model, args, input_dim):
	#if args[0].shape[1] != input_dim:
	#	print("Args[0] requires shape", (None,input_dim), ", got", args[0].shape);
	#if args[1].shape[0] != args[0].shape[0]:
	#	print("Args[1] requires shape's first index to be ", args[0].shape[0]," got", args[1].shape[0]);l
	args = np.swapaxes(args, 0, 1);
	args = [
		np.repeat([1.0], args[0].shape[0]),
		args[0],
		args[1],
		args[2],
		args[3],
	]
	return model.predict([arg for arg in args]);


def firststart():
	#print("HERE");
	import tensorflow 
	from keras.models import load_model
	model = load_model('model.h5')
	#print("TEST SHAPE: ", np.random.randn(input_dim).shape);
	#print("TEST SHAPE 2: ", np.array([0]).shape);
	test = predict(
		model,
		np.random.randn(1, 4, input_dim),
		input_dim
	) # make sure model is compiled during init
	#print("GOT SHAPE: ", test.shape);
	return model

if 'TF_CPP_MIN_LOG_LEVEL' in os.environ and (os.environ['TF_CPP_MIN_LOG_LEVEL'] == "" or os.environ['TF_CPP_MIN_LOG_LEVEL'] == "3"):
	old_stderr = sys.stderr;
	old_stdout = sys.stdout;
	with open(os.devnull, 'w') as sys.stderr:
		with open(os.devnull, 'w') as sys.stdout:
			model = firststart();
	sys.stderr = old_stderr;
	sys.stdout = old_stdout;
else:
	model = firststart();

myID, gameMap = getInit()

def frame_to_stack(frame):
    game_map = np.array([[(x.owner, x.production, x.strength) for x in row] for row in frame.contents])
    return np.array([(game_map[:, :, 0] == myID),  # 0 : owner is me
                      ((game_map[:, :, 0] != 0) & (game_map[:, :, 0] != myID)),  # 1 : owner is enemy
                      game_map[:, :, 1]/20,  # 2 : production
                      game_map[:, :, 2]/255,  # 3 : strength
                      ]).astype(np.float32)


sendInit('latest-version')
turn = 0;
while True:
	stack = frame_to_stack(getFrame())
	positions = np.transpose(np.nonzero(stack[0]))
	#input_stack = np.concatenate((
	#	np.array([stack_to_input(stack, p) for p in positions]),
		#np.repeat(np.array([min(turn/150,1)]), len(positions)),
		#np.array([stack_to_lateinput(stack, p) for p in positions]),
	#), axis=1)
	input_stack = np.array([stack_to_input(stack, p) for p in positions]);
	#print("SHAPE: ", input_stack.shape)
	output = predict(model, input_stack, input_dim)
	#if len(positions) > 1:
	#	print("POSITIONS: ", positions);
	#	print("OUTPUT: ", output);
	#	print("OUTPUT ARGMAX: ", output[0].argmax());
	#	print("1: ", positions[0][1], positions[0][0]);
	sendFrame([Move(Location(positions[i][1],positions[i][0]), output[i].argmax()) for i in range(len(positions))])
	turn += 1;
