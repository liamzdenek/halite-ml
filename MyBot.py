from networking import *
import os
import sys
import numpy as np

VISIBLE_DISTANCE = 6
input_dim = 4*(2*VISIBLE_DISTANCE+1)*(2*VISIBLE_DISTANCE+1)


def predict(model, args, input_dim):
	if args[0].shape[1] != input_dim:
		print("Args[0] requires shape", (None,input_dim), ", got", args[0].shape);
	if args[1].shape[0] != args[0].shape[0]:
		print("Args[1] requires shape's first index to be ", args[0].shape[0]," got", args[1].shape[0]);
	return model.predict(args);


def firststart():
	print("HERE");
	import tensorflow 
	from keras.models import load_model
	model = load_model('model.h5')
	print("TEST SHAPE: ", np.random.randn(1,input_dim).shape);
	print("TEST SHAPE 2: ", np.array([0]).shape);
	test = predict(model, [
		np.random.randn(1, input_dim),
		np.array([0]),
		np.random.randn(1, input_dim),
	], input_dim) # make sure model is compiled during init
	print("GOT SHAPE: ", test.shape);
	return model

if os.environ['TF_CPP_MIN_LOG_LEVEL'] == "" or os.environ['TF_CPP_MIN_LOG_LEVEL'] == "3":
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

def stack_to_input(stack, position):
    return np.take(np.take(stack,
                np.arange(-VISIBLE_DISTANCE,VISIBLE_DISTANCE + 1)+position[0],axis=1,mode='wrap'),
                np.arange(-VISIBLE_DISTANCE,VISIBLE_DISTANCE + 1)+position[1],axis=2,mode='wrap').flatten()

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

def frame_to_stack(frame):
    game_map = np.array([[(x.owner, x.production, x.strength) for x in row] for row in frame.contents])
    return np.array([(game_map[:, :, 0] == myID),  # 0 : owner is me
                      ((game_map[:, :, 0] != 0) & (game_map[:, :, 0] != myID)),  # 1 : owner is enemy
                      game_map[:, :, 1]/20,  # 2 : production
                      game_map[:, :, 2]/255,  # 3 : strength
                      ]).astype(np.float32)


sendInit('general-freedom')
turn = 0;
while True:
	stack = frame_to_stack(getFrame())
	positions = np.transpose(np.nonzero(stack[0]))
	input_stack = [
		np.array([stack_to_input(stack, p) for p in positions]),
		np.repeat(np.array([min(turn/150,1)]), len(positions)),
		np.array([stack_to_lateinput(stack, p) for p in positions]),
	];
	output = predict(model, input_stack, input_dim)
	sendFrame([Move(Location(positions[i][1],positions[i][0]), output[i].argmax()) for i in range(len(positions))])
	turn += 1;
