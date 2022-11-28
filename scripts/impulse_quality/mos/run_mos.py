import os
import librosa
from scipy.io import wavfile
import soundfile as sf
import numpy as np
import pickle
from tqdm import tqdm
from scipy.signal import fftconvolve
import speechmetrics # install from https://github.com/aliutkus/speechmetrics


def run_mos(metrics,
            test_scene_n_query_pose_to_mono_filename_dct,
            mono_audio_source_dir,
            eval_root_dir,
            eval_type,
            dump_dir,
            eval_gt=False):
    sceneNRIRName_to_mos = {}
    if eval_gt:
        gt_sceneNRIRName_to_mos = {}

    all_scenes_this_eval_type = os.listdir(eval_root_dir)
    for sampled_scene in tqdm(all_scenes_this_eval_type):
        scene_rir_dir = os.path.join(eval_root_dir, sampled_scene)
        all_scene_rirs = os.listdir(scene_rir_dir)
        all_scene_rirs_tmp = []
        for scene_rir in all_scene_rirs:
            if scene_rir.split("_")[0] == "pred":
                all_scene_rirs_tmp.append(scene_rir)

        all_scene_rirs = all_scene_rirs_tmp

        rir_count = 0
        for sampled_scene_rir in tqdm(all_scene_rirs):
            sampled_scene_rir_path = os.path.join(scene_rir_dir, sampled_scene_rir)
            assert os.path.isfile(sampled_scene_rir_path)

            if sampled_scene_rir.split("_")[0] == "pred":
                corresponding_gt_rir = "gt" + sampled_scene_rir[4:]
                corresponding_gt_rir_path = os.path.join(scene_rir_dir, corresponding_gt_rir)
                assert os.path.isfile(corresponding_gt_rir_path)
            else:
                raise ValueError

            s = int((sampled_scene_rir.split(".")[0].split("_")[2])[1:])
            r = int((sampled_scene_rir.split(".")[0].split("_")[3])[1:])
            az = int((sampled_scene_rir.split(".")[0].split("_")[4])[2:])

            assert (sampled_scene, s, r, az) in test_scene_n_query_pose_to_mono_filename_dct
            mono_filename = test_scene_n_query_pose_to_mono_filename_dct[(sampled_scene, s, r, az)]
            mono_file_path = os.path.join(mono_audio_source_dir, mono_filename)
            assert os.path.isfile(mono_file_path)

            mono_sr, mono_audio = wavfile.read(mono_file_path)

            rir_sr, rir = wavfile.read(sampled_scene_rir_path)

            gt_rir_sr, gt_rir = wavfile.read(corresponding_gt_rir_path)

            full_conv_rir = []
            for ch_i in range(rir.shape[-1]):
                conv_rir = fftconvolve(mono_audio, rir[:, ch_i], mode="full").astype("float32")
                full_conv_rir.append(conv_rir)
            full_conv_rir = np.array(full_conv_rir).T

            gt_full_conv_rir = []
            for ch_i in range(gt_rir.shape[-1]):
                conv_rir = fftconvolve(mono_audio, gt_rir[:, ch_i], mode="full").astype("float32")
                gt_full_conv_rir.append(conv_rir)
            gt_full_conv_rir = np.array(gt_full_conv_rir).T

            scores = metrics(full_conv_rir, rate=16000)
            score = scores['mosnet'][0]

            sceneNRIRName_to_mos[sampled_scene + "_" + sampled_scene_rir.split(".")[0]] = score

            if eval_gt:
                gt_scores = metrics(gt_full_conv_rir, rate=16000)
                gt_score = gt_scores['mosnet'][0]
                gt_sceneNRIRName_to_mos[sampled_scene + "_" + sampled_scene_rir.split(".")[0]] = gt_score
            rir_count += 1

            if rir_count == 3:
                break
        break

    dump_path = os.path.join(dump_dir, f"{eval_type}_mos.pkl")
    with open(dump_path, "wb") as fo:
        pickle.dump(sceneNRIRName_to_mos, fo, protocol=pickle.HIGHEST_PROTOCOL)

    if eval_gt:
        dump_path = os.path.join(dump_dir, f"gt_{eval_type}_mos.pkl")
        with open(dump_path, "wb") as fo:
            pickle.dump(gt_sceneNRIRName_to_mos, fo, protocol=pickle.HIGHEST_PROTOCOL)


WINDOW_LENGTH = 5
metrics = speechmetrics.load('mosnet', WINDOW_LENGTH)

EVAL_GT = True

EVAL_ROOT_DIR_PREFIX = "../../../runs_eval/fs_rir/audio_waveforms"

DUMP_DIR = EVAL_ROOT_DIR_PREFIX[:-len(EVAL_ROOT_DIR_PREFIX.split("/")[-1])]

TEST_SCENE_N_QUERY_POSE_TO_MONO_FILENAME_DIR = "../../../data/compute_mos/testSceneNQueryPose2monoFilename/mp3d/"+\
                                                "allRoom_14DatapointsPerScene/test/60_qry"

MONO_AUDIO_SOURCE_DIR = "../../../data/audio_data/mosnet/libriSpeech/"
assert os.path.isdir(MONO_AUDIO_SOURCE_DIR)

for EVAL_TYPE in ["seen_eval", "unseen_eval"]:
    if EVAL_TYPE == "seen_eval":
        test_scene_n_query_pose_to_mono_filename_path = os.path.join(TEST_SCENE_N_QUERY_POSE_TO_MONO_FILENAME_DIR,
                                                                     "seen_eval_798datapoints_testSceneNQueryPose2monoFilename.pkl")
    elif EVAL_TYPE == "unseen_eval":
        test_scene_n_query_pose_to_mono_filename_path = os.path.join(TEST_SCENE_N_QUERY_POSE_TO_MONO_FILENAME_DIR,
                                                                     "unseen_eval_364datapoints_testSceneNQueryPose2monoFilename.pkl")

    assert os.path.isfile(test_scene_n_query_pose_to_mono_filename_path)
    with open(test_scene_n_query_pose_to_mono_filename_path, "rb") as fi:
        test_scene_n_query_pose_to_mono_filename_dct = pickle.load(fi)

    EVAL_ROOT_DIR = os.path.join(EVAL_ROOT_DIR_PREFIX, EVAL_TYPE)
    assert os.path.isdir(EVAL_ROOT_DIR)

    run_mos(metrics,
            test_scene_n_query_pose_to_mono_filename_dct,
            MONO_AUDIO_SOURCE_DIR,
            EVAL_ROOT_DIR,
            EVAL_TYPE,
            DUMP_DIR,
            eval_gt=EVAL_GT,)
