import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg

from h1_2_humanoid.assets import ISAAC_ASSET_DIR


H1_2_HIP_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=[".*hip_yaw_joint", ".*hip_roll_joint", ".*hip_pitch_joint"], #need to check if names are the same in urdf/usd file
    effort_limit=200.0,
    velocity_limit=23,
    stiffness={".*": 200.0},
    damping={".*": 2.0},
    armature={".*": 0.01},
)

H1_2_KNEE_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=[".*knee_joint"],
    effort_limit=300.0,
    velocity_limit=14,
    stiffness={".*": 300.0},
    damping={".*": 4.0},
    armature={".*": 0.01},
)

H1_2_ANKLE_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=[".*ankle_roll_joint", ".*ankle_pitch_joint"],
    effort_limit={".*ankle_roll_joint":40.0, ".*ankle_pitch_joint":60.0},
    velocity_limit=9,
    stiffness={".*": 40.0},
    damping={".*": 2.0},
    armature={".*": 0.01},
)

H1_2_UPPER_BODY_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=["torso_joint", ".*_shoulder_.*", ".*_elbow.*", ".*_wrist_.*"],
    effort_limit=100.0,
    velocity_limit=20.0,
    stiffness={".*": 100.0},
    damping={".*": 5.0},
    armature={".*": 0.01},
)

H1_2_HUMANOID_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/Robots/h1_2_handless/h1_2_handless.usd", #need to add the location of the .usd file when generate in isaac lab
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05), #to sit in z=0
        joint_pos={
            'left_hip_yaw_joint': 0.0,
            'right_hip_yaw_joint': 0.0,
            'left_hip_roll_joint': 0.0,
            'right_hip_roll_joint': 0.0,
            'left_hip_pitch_joint': -0.16,
            'right_hip_pitch_joint': -0.16,
            'left_knee_joint': 0.36,
            'right_knee_joint': 0.36,
            'left_ankle_pitch_joint': -0.2,
            'right_ankle_pitch_joint': -0.2,
            'left_ankle_roll_joint': 0.0,
            'right_ankle_roll_joint': 0.0,           #the names need to follow the urdf names
            "torso_joint": 0.0,
            ".*_shoulder_.*": 0.0,
            ".*_elbow.*": 0.0,
            ".*_wrist_.*": 0.0,
        },
    ),
    actuators={"hip": H1_2_HIP_ACTUATOR_CFG, 
               "knee": H1_2_KNEE_ACTUATOR_CFG,
               "ankle": H1_2_ANKLE_ACTUATOR_CFG,
               "upper_body":H1_2_UPPER_BODY_ACTUATOR_CFG},
    soft_joint_pos_limit_factor=0.95,
)