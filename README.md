# VR Experiment Pilot
**Task description summary:** Each trial four shapes are presented consisting of three spheres and one cube, or three cubes and one sphere. The goal is to reach as quickly as possible towards the unique shape.

**In more detail:** The participant is seated in front of a physical table. A marker is placed on the participant's index finger of which the location is tracked with a Certus Optotrak. The finger position is represented in VR by a green sphere. In the VR environment, a table is presented which is alligned to the physical table. On the physical and VR table a starting position is present. In order to start each iteration of the task (i.e. trial) the participant needs to put their index finger on the start position. Then four shapes, either one sphere among three cubes or one cube among three sphere, are presented which float above the table, in front of the participant. Then the participants has one second to reach towards the unique shape i.e. touch the unique shape with their index finger. On some trials all shapes have the same color (baseline condition), on other trials one of the non-target shapes is in a different color (distractor condition). Auditory and visual feedback about performance is presented at the end of each trial.

**Analysis of 3D samples (finger position data) summary:**  Missing samples are interpolated via splines. Then, the data is butterworth filtered. The start of a reaching movement is detected when the finger velocity exceeds a threshold. The end of a reaching movement is detected when the finger position touches any shape. Reaching movements are extracted and resampled to 101 samples equally spaced along the movement amplitude (i.e. normalization to distance). Movements with an intermediate stop (i.e. when the velocity profile has a local minimum) are excluded. The movement trajectory, curvature, latency, etc. are calculated and compared between the baseline and distractor conditions.
#### VR Experiment Script for Vizard 6
- `Color Singleton.py`

#### Data Analysis
Parse raw 3D samples
- `hand_parser_VR.py`

#### Pilot Data (3 participants) in `data` folder
- `data_hand` raw 3D samples representing participants' finger location
- `data_log` table containing (in)dependent variables recorded during experiment; each row = one trial (separator ',')
- `exp_settings` experimental settings (e.g. location of shapes, number of trials, etc.; separator ':')
