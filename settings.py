system_str = 'cartpole'
# system_str = 'ballbeam'
# system_str = 'quadrotor'
# system_str = 'bicycle'

fps = 15
interval = round(1000/fps)
period_eq = round(2*fps)
frames = 5*period_eq

stepsize = 5/fps
step_method = 'rk4'
disturb = False

# save_gif = True
save_gif = False