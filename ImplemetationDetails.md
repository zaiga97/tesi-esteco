# Environment
## Actions
  The action that the agent can do, this is simply the movement the agent performs.
  
    [x] (Dx, Dy)
    []  (l, alpha) with l in [0, 1] and alpha in [-pi, pi]
  
  - Actions need to be scaled every time we change the time scale of the environment or the type of agent that is being simulated.
  - Actions need to be somehow limited to prevent the agent from moving too far in a single step.

  A case could be made for using accelerations. This will complicate the state-action interactions (eg. require current velocity) and I will avoid it.
  
## States
  The state is the information our agent will use for deciding the next action. For this reason the state rapresentation is crucial. We will also like the state dimension to be costant so we will limit the number of other agents our main character is able to percieve only the one that is closer to the intersection for each lane.
  
  - basic state = {x, y, x_f, y_f, x_a1, y_a1, v_a1, a_a1, x_a2, y_a2, v_a2, a_a2}
  
  To reduce dimesionality we will also make velocity and acceleration discrete. 
  - simplyfied state = {x, y, x_f, y_f, x_a1, y_a1, dv_a1, da_a1, x_a2, y_a2, dv_a2, da_a2}
    - dv = { v<5, 5<v<20, 15<v<40, 35<v<inf }
    - av = { a<0, a~0, a>0 }

  - graphical state = ?

# Metrics

  For evaluating the agent performace I will use several metrics:
  
  - PET: post encouter time
  - MD: minimum distance
  - Crossing time
  - State occupation (heat map)

  Keep in mind a simple metrics doesn't exist; otherwise the problem could be reduced to maximizing this metric. One could try and teach a classifier to separate real and fake trajectories (similar to GAN discriminator) but then the problem will be evaluating this NN performances.
  
# Data

  The data are elaborated via the following way:  
  - Trajectories that appear to be broken (finish in a place and an other starts 2m away in the following second) are united.
  - Short trajectories (both in lenght (<5m) than in number of points (<10)) are removed because are usually errors of the detection system
  - Interpolation is used to sample the trajectory every 0.25s at regular intervals. The time index is then substituted with a more usuable integer index

  The data is the reorganized in 3 dataframes:

  1) traj_df: {id, t, x, y, vx, vy, ax, ay, p(shapely.geometry.Point), v(shapely.geometry.Point), a(shapely.geometry.Point)}
  2) agent_df: {id, cat, traj(shapely.geometry.LineString), xs, xf, ys, yf, ts, tf}
  3) episode_df: {id, other_id}

  
