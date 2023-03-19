# system_str = 'cartpole'
system_str = 'ballbeam'
# system_str = 'quadrotor'
# system_str = 'bicycle'

# fps = 15
fps = 150
interval = round(1000/fps)
period_eq = round(2*fps)

step_size = 5/fps

# step_method = 'euler'
# step_method = 'rk2'
step_method = 'rk4'

disturb = True
# disturb = False

# save_gif = True
save_gif = False

if save_gif:
    frames = 5*period_eq
    repeat = False
else:
    frames = None
    repeat = None
