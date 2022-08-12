# Environment
## Actions
  The action that the agent can do, this is simply the movement the agent performs.
  
  [x] (Dx, Dy)
  
  []  (l, alpha)
  
  - Actions need to be scaled every time we change the time scale of the environment or the type of agent that is being simulated.
  - Actions need to be somehow limited to prevent the agent from moving too far in a single step.
  
## States
  The state is the information our agent will use for deciding the next action. For this reason the state rapresentation is crucial.
  We will also like the state dimension to be costant so we will limit the number of other agents our main character is able to percieve
  only the one that is closer to the intersection for each lane.
  
  - basic state = {x, y, x_f, y_f, x_a1, y_a1, v_a1, a_a1, x_a2, y_a2, v_a2, a_a2}
  
  To reduce dimesionality we will also make velocity and acceleration discrete. 
  - simplyfied state = {x, y, x_f, y_f, x_a1, y_a1, dv_a1, da_a1, x_a2, y_a2, dv_a2, da_a2}
    - dv = { v<5, 5<v<20, 15<v<40, 35<v<inf }
    - av = { a<0, a~0, a>0 }

  - graphical state = ?
