import numpy as np;

VISIBLE_DISTANCE = 10
num_data_points = 4
#input_dim=num_data_points*(2*VISIBLE_DISTANCE+1)*(2*VISIBLE_DISTANCE+1)

cone_dim = 0;
cone_init = 0;
for i in range(0,VISIBLE_DISTANCE):
	cone_dim += 1+cone_init*2;
	cone_init+=1;

input_dim = cone_dim *num_data_points;

max_cone_radius = 3;
cone_points = np.array([(i, point) for i in range(0, VISIBLE_DISTANCE) for point in np.arange(min(-i, max_cone_radius) , max(i+1, max_cone_radius))]);
#print("CONE SHAPE: ", cone_points);
#print("CONE SHAPE: ", cone_points.shape);
#print("CONE DIM: ", cone_dim);

cone_points = np.swapaxes(cone_points, 0, 1);

cone_we = cone_points;
cone_ns = [ cone_points[1], cone_points[0] ];

#print("CONE POINTS: ", cone_points);

def get_cone(stack, position, relative):
	#print("CONE DISTANCE: ", VISIBLE_DISTANCE, "CONE AREA: ", cone_dim); 
	stack = np.swapaxes(stack, 0, 1)
	stack = np.swapaxes(stack, 1, 2)
	#print("STACK SHAPE: ", stack.shape);
	points = [];
	if relative[0] != 0:
		points = [position[0] + cone_we[0] * relative[0], position[1] + cone_we[1]];
		#print("CONE WE: ", cone_we);
	else:
		points = [position[0] + cone_ns[0], position[1] + cone_ns[1] * relative[1]];
		#print("CONE NS: ", cone_ns);
	#print ("RELATIVE: ", relative);
	#print ("POSITION: ", position);
	#print ("POINTS: ", points);
	a = points[0] % stack.shape[0];
	b = points[1] % stack.shape[1];
	retval = stack[a, b].flatten();
	#retval = np.take(
	#	np.take(stack, points[0], mode='wrap'),
	#	points[1],
	#	mode="wrap"
	#)
	#print("RETVAL: ", retval.shape);
	#retval = np.swapaxes(retval, 0, 1);
	return retval;

def stack_to_input(stack, position):
	ret = [];
	#print("STACK: ", stack.shape);	
	retval = [
		get_cone(stack, position, (0, -1)),
		get_cone(stack, position, (0, 1)),
		get_cone(stack, position, (1, 0)),
		get_cone(stack, position, (-1, 0)),
	]
	return retval
	#return np.take(np.take(stack,
	#            np.arange(-VISIBLE_DISTANCE,VISIBLE_DISTANCE + 1)+position[0],axis=1,mode='wrap'),
	#            np.arange(-VISIBLE_DISTANCE,VISIBLE_DISTANCE + 1)+position[1],axis=2,mode='wrap').flatten()


