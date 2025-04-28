# CCE5106-RL-Assignment
```
CCE5106-RL-Assignment/
│  pyproject.toml          # installs the package ‘cce5106’
│  README.md
└─ cce5106/
   ├─ envs/
   │   └─ um_flight_env.py     # Gymnasium environment you may tweak
   ├─ sims/
   │   ├─ collision_sim.py     # straight-line traffic simulator
   │   └─ flight.py            # individual Flight objects
   ├─ agents/
   │   └─ train_umflightenv_sb3.py  # training script to finish
   └─ utils/
       └─ utils.py             # helper (random string generator)
```  

* `envs/um_flight_env.py` - **Gymnasium** wrapper that exposes a stateful `CollisionCourseSimulator` as `UMFlightEnv`.  
* `sims/*` – pure physics / geometry, no RL imports. Keep it simple & stable.  
* `agents/train_umflightenv_sb3.py` – where you plug in Stable-Baselines3 (SB3) or another RL library.  
* `utils/*` – convenience helpers.

For the assignment you only need to edit
* `cce5106/agents/train_umflightenv_sb3.py` - finish the training loop, callbacks, logging, etc. (See TODOs)
* (optionally) `cce5106/envs/um_flight_env.py` - you can try changing the reward scaling, reward function design, observation and action spaces design, and see the effects on the training performance.
* (optionally) `cce5106/sims/collision_sim.py` - you can change the simulator speed & altitude deltas applied per discrete action, i.e. `CollisionCourseSimulator.DALT` & `CollisionCourseSimulator.DVEL`


---


## Inside **`UMFlightEnv`**

`UMFlightEnv` is a standard `gymnasium.Env` subclass which implements the following:

| Concept                                                   | How it’s implemented                                                                                                            |
|-----------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|
| **Simulation backend**                                    | instantiates `CollisionCourseSimulator` (from `cce5106.sims.collision_sim`) on `reset()` and advances it one tick per `step()`  | 
| **Action space**                                          | `Gymnasium.spaces.MultiDiscrete([5] * (n + m))`, one discrete control per aircraft: `NOOP / VEL_DEC / VEL_INC / FL_DEC / FL_INC` |
| **Observation space**                                     | Flat `np.float32` vector of length `2 x 5 x (n+m) + 1` (current & previous 5-tuple per flight + normalised time)                |
| **Reward**                                                | Negative weighted sum of three terms computed in `reward_function()`: `R = - (α·collisions + β·clearances + γ·invalid_actions)` |
| **Episode termination**                                   | `truncated` when `max_steps` elapsed or too many clearances; `terminated` when too many invalid commands                        |
| **Rendering**                                             | 2-D Matplotlib animation (`render`)|
#### External dependencies

* **Gymnasium** – interface (`Env`, `spaces`, `ObsType`, …)  
* **NumPy** – vector maths  
* **Matplotlib** – debug rendering  
* **PyYAML** – save / load scenarios  
* **Stable-Baselines3** – *only* in the training script, not in the env itself  

---

### Next steps for students

1. **Fork or clone** the repo and set it up as shown above.  
2. Open `cce5106/agents/train_umflightenv_sb3.py` and **finish the TODOs** (callbacks, hyper-params, logging).

Happy training!

