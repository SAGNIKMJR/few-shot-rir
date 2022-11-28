from typing import List, Optional, Union
import os
import shutil

from habitat import get_config as get_task_config
from habitat.config import Config as CN
from habitat.config.default import SIMULATOR_SENSOR
import habitat

DEFAULT_CONFIG_DIR = "configs/"
CONFIG_FILE_SEPARATOR = ","
# -----------------------------------------------------------------------------
# EXPERIMENT CONFIG
# -----------------------------------------------------------------------------
_C = CN()
_C.SEED = 0
_C.BASE_TASK_CONFIG_PATH = "configs/tasks/pointnav.yaml"
_C.TASK_CONFIG = CN()  # task_config will be stored as a config node
_C.CMD_TRAILING_OPTS = []  # store command line options as list of strings
_C.TRAINER_NAME = "ppo"
_C.ENV_NAME = "ExploreEnv"
_C.SIMULATOR_GPU_ID = 0
_C.TORCH_GPU_ID = 0
_C.PARALLEL_GPU_IDS = []
_C.MODEL_DIR = ''
_C.TENSORBOARD_DIR = "tb"
_C.VIDEO_OPTION = []
_C.VIDEO_DIR = '' 
_C.AUDIO_DIR = '' 
_C.VISUALIZATION_OPTION = [] 
_C.EVAL_CKPT_PATH_DIR = "data/checkpoints"  
_C.NUM_PROCESSES = 1
_C.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
_C.CHECKPOINT_FOLDER = "data/checkpoints"
_C.NUM_UPDATES = 10000
_C.LOG_INTERVAL = 10
_C.LOG_FILE = "train.log"
_C.CHECKPOINT_INTERVAL = 50
_C.USE_VECENV = True
_C.USE_SYNC_VECENV = False
_C.EXTRA_RGB = False
_C.EXTRA_DEPTH = False
_C.DEBUG = False
_C.EPS_SCENES = []
_C.EPS_SCENES_N_IDS = []
_C.JOB_ID = 1
_C.TRAIN_OR_EVAL_FOR_SPECIFIC_SCENE = False
_C.SPECIFIC_SCENE_NAME = None

# -----------------------------------------------------------------------------
# EVAL CONFIG
# -----------------------------------------------------------------------------
_C.EVAL = CN()
# The split to evaluate on
_C.EVAL.SPLIT = "val"
_C.EVAL.USE_CKPT_CONFIG = True
_C.EVAL.DATA_PARALLEL_TRAINING = False

# -----------------------------------------------------------------------------
# REINFORCEMENT LEARNING (RL) ENVIRONMENT CONFIG
# -----------------------------------------------------------------------------
_C.RL = CN()
_C.RL.SUCCESS_REWARD = 10.0
_C.RL.SLACK_REWARD = -0.01
_C.RL.WITH_DISTANCE_REWARD = True
_C.RL.DISTANCE_REWARD_SCALE = 1.0
# -----------------------------------------------------------------------------
# PROXIMAL POLICY OPTIMIZATION (PPO)
# -----------------------------------------------------------------------------
_C.RL.PPO = CN()
_C.RL.PPO.num_updates_per_cycle = 1
_C.RL.PPO.clip_param = 0.2
_C.RL.PPO.ppo_epoch = 4
_C.RL.PPO.num_mini_batch = 16
_C.RL.PPO.value_loss_coef = 0.5
_C.RL.PPO.entropy_coef = 0.01
_C.RL.PPO.lr_exploration_pol = 1e-3
_C.RL.PPO.eps = 1e-5
_C.RL.PPO.max_grad_norm = 0.5
_C.RL.PPO.num_steps = 5
_C.RL.PPO.hidden_size = 512
_C.RL.PPO.use_gae = True
_C.RL.PPO.use_linear_lr_decay = False
_C.RL.PPO.use_linear_clip_decay = False
_C.RL.PPO.gamma = 0.99
_C.RL.PPO.tau = 0.95
_C.RL.PPO.policy_type = None 
_C.RL.PPO.reward_type = None 
_C.RL.PPO.reward_window_size = 50
_C.RL.PPO.deterministic_eval = False
_C.RL.PPO.use_ddppo = False
_C.RL.PPO.ddppo_distrib_backend = "NCCL"
_C.RL.PPO.short_rollout_threshold = 0.25
_C.RL.PPO.sync_frac = 0.6
_C.RL.PPO.master_port = 8738
_C.RL.PPO.master_addr = "127.0.0.1"

# -----------------------------------------------------------------------------
# Uniform context sampler
# -----------------------------------------------------------------------------
_C.UniformContextSampler = CN()

_C.UniformContextSampler.lr = 5.0e-4
_C.UniformContextSampler.eps = 1.0e-5
_C.UniformContextSampler.max_grad_norm = None
_C.UniformContextSampler.betas = [0.9, 0.999]
_C.UniformContextSampler.num_epochs = 1000
_C.UniformContextSampler.batch_size = 64 
_C.UniformContextSampler.num_workers = 64 
_C.UniformContextSampler.set_num_workers_to_one_in_eval = False
_C.UniformContextSampler.num_datapoints_per_scene_train = 1000
_C.UniformContextSampler.num_datapoints_per_scene_eval = 50
_C.UniformContextSampler.predict_in_logspace = True
_C.UniformContextSampler.log_instead_of_log1p_in_logspace = False
_C.UniformContextSampler.log_gt = False
_C.UniformContextSampler.log_gt_eps = 1.0e-8
_C.UniformContextSampler.log1p_gt = False

_C.UniformContextSampler.use_spect_energy_decay_loss = False
_C.UniformContextSampler.spectEnergyDecayLoss = CN()
_C.UniformContextSampler.spectEnergyDecayLoss.type = "l1_loss" 
_C.UniformContextSampler.spectEnergyDecayLoss.weight =  1.0 
_C.UniformContextSampler.spectEnergyDecayLoss.slice_till_direct_signal = False
_C.UniformContextSampler.spectEnergyDecayLoss.direct_signal_len_in_ms = 50
_C.UniformContextSampler.spectEnergyDecayLoss.dont_collapse_across_freq_dim = False

_C.UniformContextSampler.TrainLosses = CN()
_C.UniformContextSampler.TrainLosses.types = ["stft_l1_loss"]
_C.UniformContextSampler.TrainLosses.weights = [1.0]

_C.UniformContextSampler.EvalMetrics = CN()
_C.UniformContextSampler.EvalMetrics.types = ["stft_l1_distance",
											  "diff_rt_startFrom60dB",
											  "diff_drr_3ms"]
_C.UniformContextSampler.EvalMetrics.type_for_ckpt_dump = "stft_l1_distance"

_C.UniformContextSampler.MemoryNet = CN()
_C.UniformContextSampler.MemoryNet.type = "transformer" 
_C.UniformContextSampler.MemoryNet.Transformer = CN()
_C.UniformContextSampler.MemoryNet.Transformer.no_self_attn_in_decoder = False
_C.UniformContextSampler.MemoryNet.Transformer.input_size = 1024
_C.UniformContextSampler.MemoryNet.Transformer.hidden_size = 1024
_C.UniformContextSampler.MemoryNet.Transformer.num_encoder_layers = 2
_C.UniformContextSampler.MemoryNet.Transformer.num_decoder_layers = 2
_C.UniformContextSampler.MemoryNet.Transformer.nhead = 2
_C.UniformContextSampler.MemoryNet.Transformer.dropout = 0.1
_C.UniformContextSampler.MemoryNet.Transformer.activation = "relu"

_C.UniformContextSampler.encode_each_modality_as_independent_context_entry = False
_C.UniformContextSampler.append_modality_type_tag_encoding_to_each_modality_encoding = False
_C.UniformContextSampler.modality_type_tag_encoding_size = 8

_C.UniformContextSampler.PositionalEnc = CN()
_C.UniformContextSampler.PositionalEnc.type = "sinusoidal"
_C.UniformContextSampler.PositionalEnc.num_freqs_for_sinusoidal = 8
_C.UniformContextSampler.PositionalEnc.shared_pose_encoder_for_context_n_query = False

_C.UniformContextSampler.FusionEnc = CN()
_C.UniformContextSampler.FusionEnc.type = "concatenate"

_C.UniformContextSampler.FusionDec = CN()
_C.UniformContextSampler.FusionDec.type = "concatenate"

_C.UniformContextSampler.dump_audio_waveforms = False

_C.UniformContextSampler.use_gl = False
_C.UniformContextSampler.use_gl_for_gt = False
_C.UniformContextSampler.use_rand_phase = False
_C.UniformContextSampler.use_rand_phase_for_gt = False

# -----------------------------------------------------------------------------
# TASK CONFIG
# -----------------------------------------------------------------------------
_TC = habitat.get_config()
_TC.defrost()

########## ACTIONS ###########
# -----------------------------------------------------------------------------
# PAUSE ACTION
# -----------------------------------------------------------------------------
_TC.TASK.ACTIONS.PAUSE = CN()
_TC.TASK.ACTIONS.PAUSE.TYPE = "PauseAction"

########## SENSORS ###########
# -----------------------------------------------------------------------------
# BINAURAL SPECTROGRAM MAGNITUDE SENSOR
# -----------------------------------------------------------------------------
_TC.TASK.BIN_SPECT_MAG_SENSOR = CN()
_TC.TASK.BIN_SPECT_MAG_SENSOR.TYPE = "BinSpectMagSensor"
_TC.TASK.BIN_SPECT_MAG_SENSOR.FEATURE_SHAPE = [256, 257, 2] # mp3d (n_fft=511, hop_length=62, win_length=400): [256, 259, 2]; 
# -----------------------------------------------------------------------------
# POSE SENSOR
# -----------------------------------------------------------------------------
_TC.TASK.POSE_SENSOR = CN()
_TC.TASK.POSE_SENSOR.TYPE = "PoseSensor"
_TC.TASK.POSE_SENSOR.FEATURE_SHAPE = [5]
# -----------------------------------------------------------------------------
# environment config
# -----------------------------------------------------------------------------
_TC.ENVIRONMENT.MAX_EPISODE_STEPS = 100
_TC.ENVIRONMENT.MAX_CONTEXT_LENGTH = 100
_TC.ENVIRONMENT.MAX_QUERY_LENGTH = 25
_TC.ENVIRONMENT.MAX_QUERY_LENGTH_EVAL_FINETUNE_NAF = 25
_TC.ENVIRONMENT.LOAD_CONTEXT_FROM_DISK = False
_TC.ENVIRONMENT.LOAD_QUERY_FOR_ARBITRARY_RIRS_FROM_DISK = False
_TC.ENVIRONMENT.ARBITRARY_RIR_TRAIN_QUERY_POSE_IDXS_PATH = None
_TC.ENVIRONMENT.ARBITRARY_RIR_TRAIN_QUERY_POSE_SUBGRAPH_IDXS_PATH = None
_TC.ENVIRONMENT.ARBITRARY_RIR_TRAIN_SCENE_NAMES_PATH = None
_TC.ENVIRONMENT.SEEN_ENV_EVAL_CONTEXT_POSE_IDXS_PATH = None
_TC.ENVIRONMENT.ARBITRARY_RIR_SEEN_ENV_EVAL_QUERY_POSE_IDXS_PATH = None
_TC.ENVIRONMENT.ARBITRARY_RIR_SEEN_ENV_EVAL_QUERY_POSE_SUBGRAPH_IDXS_PATH = None
_TC.ENVIRONMENT.ARBITRARY_RIR_SEEN_ENV_EVAL_SCENE_NAMES_PATH = None
_TC.ENVIRONMENT.UNSEEN_ENV_EVAL_CONTEXT_POSE_IDXS_PATH = None
_TC.ENVIRONMENT.ARBITRARY_RIR_UNSEEN_ENV_EVAL_QUERY_POSE_IDXS_PATH = None
_TC.ENVIRONMENT.ARBITRARY_RIR_UNSEEN_ENV_EVAL_QUERY_POSE_SUBGRAPH_IDXS_PATH = None
_TC.ENVIRONMENT.ARBITRARY_RIR_UNSEEN_ENV_EVAL_SCENE_NAMES_PATH = None
# -----------------------------------------------------------------------------
# simulator config
# -----------------------------------------------------------------------------
_TC.SIMULATOR.SEED = -1
_TC.SIMULATOR.SCENE_DATASET = "mp3d"
_TC.SIMULATOR.MAX_EPISODE_STEPS = 100 #
_TC.SIMULATOR.GRID_SIZE = 1.0
_TC.SIMULATOR.USE_RENDERED_OBSERVATIONS = True
_TC.SIMULATOR.RENDERED_OBSERVATIONS = "data/scene_observations/"
# -----------------------------------------------------------------------------
# audio config
# -----------------------------------------------------------------------------
_TC.SIMULATOR.AUDIO = CN()
_TC.SIMULATOR.AUDIO.RIR_DIR = "data/binaural_rirs/mp3d"
_TC.SIMULATOR.AUDIO.META_DIR = "data/metadata/mp3d"
_TC.SIMULATOR.AUDIO.GRAPH_FILE = 'graph.pkl'
_TC.SIMULATOR.AUDIO.POINTS_FILE = 'points.txt'
_TC.SIMULATOR.AUDIO.NUM_WORKER = 4
_TC.SIMULATOR.AUDIO.BATCH_SIZE = 128
_TC.SIMULATOR.AUDIO.RIR_SAMPLING_RATE = 16000
_TC.SIMULATOR.AUDIO.HOP_LENGTH = 62 
_TC.SIMULATOR.AUDIO.N_FFT = 511 
_TC.SIMULATOR.AUDIO.WIN_LENGTH = 248 
_TC.SIMULATOR.AUDIO.SWEEP_AUDIO_DIR = "data/audio_data/sweep_sounds/visual_echoes/"
_TC.SIMULATOR.AUDIO.SWEEP_AUDIO_FILENAME = "data_sweep_audio_3ms_sweep.wav"
_TC.SIMULATOR.AUDIO.VALID_ECHO_POSES_PATH = ""
_TC.SIMULATOR.AUDIO.VALID_ARBITRARY_RIR_TRAIN_POSES_PATH = ""
_TC.SIMULATOR.AUDIO.VALID_ARBITRARY_RIR_SEEN_ENV_EVAL_POSES_PATH = ""
_TC.SIMULATOR.AUDIO.VALID_ARBITRARY_RIR_UNSEEN_ENV_EVAL_POSES_PATH = ""
# -----------------------------------------------------------------------------
# Dataset extension
# -----------------------------------------------------------------------------
_TC.DATASET.VERSION = 'v1'


def merge_from_path(config, config_paths):
	"""
	merge config with configs from config paths
	:param config: original unmerged config
	:param config_paths: config paths to merge configs from
	:return: merged config
	"""
	if config_paths:
		if isinstance(config_paths, str):
			if CONFIG_FILE_SEPARATOR in config_paths:
				config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
			else:
				config_paths = [config_paths]

		for config_path in config_paths:
			config.merge_from_file(config_path)

	return config


def get_config(
		config_paths: Optional[Union[List[str], str]] = None,
		opts: Optional[list] = None,
		model_dir: Optional[str] = None,
		run_type: Optional[str] = None
) -> CN:
	r"""Create a unified config with default values overwritten by values from
	`config_paths` and overwritten by options from `opts`.
	Args:
		config_paths: List of config paths or string that contains comma
		separated list of config paths.
		opts: Config options (keys, values) in a list (e.g., passed from
		command line into the config. For example, `opts = ['FOO.BAR',
		0.5]`. Argument can be used for parameter sweeping or quick tests.
		model_dir: suffix for output dirs
		run_type: either train or eval
	"""
	config = merge_from_path(_C.clone(), config_paths)
	config.TASK_CONFIG = get_task_config(config_paths=config.BASE_TASK_CONFIG_PATH)

	if opts:
		config.CMD_TRAILING_OPTS = opts
		config.merge_from_list(opts)

	assert model_dir is not None, "set --model-dir"
	config.MODEL_DIR = model_dir
	config.TENSORBOARD_DIR = os.path.join(config.MODEL_DIR, config.TENSORBOARD_DIR)
	config.CHECKPOINT_FOLDER = os.path.join(config.MODEL_DIR, 'data')
	config.VIDEO_DIR = os.path.join(config.MODEL_DIR, 'video_dir')
	config.AUDIO_DIR = os.path.join(config.MODEL_DIR, 'audio_dir')
	config.LOG_FILE = os.path.join(config.MODEL_DIR, config.LOG_FILE)
	if config.EVAL_CKPT_PATH_DIR == "data/checkpoints":
		config.EVAL_CKPT_PATH_DIR = os.path.join(config.MODEL_DIR, 'data')

	dirs = [config.VIDEO_DIR, config.AUDIO_DIR, config.TENSORBOARD_DIR, config.CHECKPOINT_FOLDER]
	if run_type == 'train':
		# check dirs
		if any([os.path.exists(d) for d in dirs]):
			for d in dirs:
				if os.path.exists(d):
					print('{} exists'.format(d))
			key = input('Output directory already exists! Overwrite the folder? (y/n)')
			if key == 'y':
				for d in dirs:
					if os.path.exists(d):
						shutil.rmtree(d)

	config.TASK_CONFIG.defrost()

	#------------------ modifying SIMULATOR cfg --------------------
	## setting SIMULATOR'S USE_SYNC_VECENV flag
	config.TASK_CONFIG.SIMULATOR.USE_SYNC_VECENV = config.USE_SYNC_VECENV

	## setting max. number of steps of simulator
	config.TASK_CONFIG.SIMULATOR.MAX_EPISODE_STEPS = config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS

	#-------------------------- modifying cfgs for visualization -------------------
	if len(config.VIDEO_OPTION) > 0:
		config.VISUALIZATION_OPTION = ["top_down_map"]
		config.TASK_CONFIG.SIMULATOR.USE_RENDERED_OBSERVATIONS = False

	config.TASK_CONFIG.freeze()

	config.freeze()

	#---------------------------- assertions for metrics --------------------------------
	if (config.TRAINER_NAME == "uniform_context_sampler") and (run_type == "train"):
		assert config.UniformContextSampler.EvalMetrics.type_for_ckpt_dump\
			   in config.UniformContextSampler.EvalMetrics.types
	return config


def get_task_config(
		config_paths: Optional[Union[List[str], str]] = None,
		opts: Optional[list] = None
) -> habitat.Config:
	r"""
	get config after merging configs stored in yaml files and command line arguments
	:param config_paths: paths to configs
	:param opts: optional command line arguments
	:return: merged config
	"""
	config = _TC.clone()
	if config_paths:
		if isinstance(config_paths, str):
			if CONFIG_FILE_SEPARATOR in config_paths:
				config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
			else:
				config_paths = [config_paths]

		for config_path in config_paths:
			config.merge_from_file(config_path)

	if opts:
		config.merge_from_list(opts)

	config.freeze()
	return config
