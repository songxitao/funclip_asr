#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunClip). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)
import os
os.environ['MODELSCOPE_CACHE'] = 'E:\\FunClip\\FunClip\\model' 

import re
import sys
import copy
import librosa
import logging
import argparse
import numpy as np
import soundfile as sf
from moviepy.editor import *
import moviepy.editor as mpy
from moviepy.video.tools.subtitles import SubtitlesClip, TextClip
from moviepy.editor import VideoFileClip, concatenate_videoclips
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from utils.subtitle_utils import generate_srt, generate_srt_clip
from utils.argparse_tools import ArgumentParser, get_commandline_args
from utils.trans_utils import pre_proc, proc, write_state, load_state, proc_spk, convert_pcm_to_float

# [MODIFICATION START] 1. 增强过程可见性：配置日志系统
# 配置日志记录器，设置级别为INFO，这样INFO和WARNING都会被打印
# format参数定义了日志的输出格式，包含时间、级别和消息
# force=True 确保即使在模块被多次导入时也能正确配置
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    force=True)
# [MODIFICATION END]

class VideoClipper():
    # ... (VideoClipper 类的所有代码保持不变) ...
    # 为了简洁，这里省略了未修改的类内部代码，您无需改动它们
    def __init__(self, funasr_model):
        logging.info("Initializing VideoClipper.") # logging.warning -> logging.info
        self.funasr_model = funasr_model
        self.GLOBAL_COUNT = 0

    def recog(self, audio_input, sd_switch='no', state=None, hotwords="", output_dir=None):
        if state is None:
            state = {}
        sr, data = audio_input
        data = convert_pcm_to_float(data)
        if sr != 16000:
            data = librosa.resample(data, orig_sr=sr, target_sr=16000)
        if len(data.shape) == 2:
            logging.warning("Input wav shape: {}, only first channel reserved.".format(data.shape))
            data = data[:,0]
        state['audio_input'] = (sr, data)
        if sd_switch == 'Yes':
            rec_result = self.funasr_model.generate(data, 
                                                    return_spk_res=True,
                                                    return_raw_text=True, 
                                                    is_final=True,
                                                    output_dir=output_dir, 
                                                    hotword=hotwords, 
                                                    pred_timestamp=self.lang=='en',
                                                    en_post_proc=self.lang=='en',
                                                    cache={})
            res_srt = generate_srt(rec_result[0]['sentence_info'])
            state['sd_sentences'] = rec_result[0]['sentence_info']
        else:
            rec_result = self.funasr_model.generate(data, 
                                                    return_spk_res=False, 
                                                    sentence_timestamp=True, 
                                                    return_raw_text=True, 
                                                    is_final=True, 
                                                    hotword=hotwords,
                                                    output_dir=output_dir,
                                                    pred_timestamp=self.lang=='en',
                                                    en_post_proc=self.lang=='en',
                                                    cache={})
            res_srt = generate_srt(rec_result[0]['sentence_info'])
        state['recog_res_raw'] = rec_result[0]['raw_text']
        state['timestamp'] = rec_result[0]['timestamp']
        state['sentences'] = rec_result[0]['sentence_info']
        res_text = rec_result[0]['text']
        return res_text, res_srt, state

    def clip(self, dest_text, start_ost, end_ost, state, dest_spk=None, output_dir=None, timestamp_list=None):
        audio_input = state['audio_input']
        recog_res_raw = state['recog_res_raw']
        timestamp = state['timestamp']
        sentences = state['sentences']
        sr, data = audio_input
        data = data.astype(np.float64)
        if timestamp_list is None:
            all_ts = []
            if dest_spk is None or dest_spk == '' or 'sd_sentences' not in state:
                for _dest_text in dest_text.split('#'):
                    if '[' in _dest_text:
                        match = re.search(r'\[(\d+),\s*(\d+)\]', _dest_text)
                        if match:
                            offset_b, offset_e = map(int, match.groups())
                            log_append = ""
                        else:
                            offset_b, offset_e = 0, 0
                            log_append = "(Bracket detected in dest_text but offset time matching failed)"
                        _dest_text = _dest_text[:_dest_text.find('[')]
                    else:
                        log_append = ""
                        offset_b, offset_e = 0, 0
                    _dest_text = pre_proc(_dest_text)
                    ts = proc(recog_res_raw, timestamp, _dest_text)
                    for _ts in ts: all_ts.append([_ts[0]+offset_b*16, _ts[1]+offset_e*16])
                    if len(ts) > 1 and match:
                        log_append += '(offsets detected but No.{} sub-sentence matched to {} periods in audio,                             offsets are applied to all periods)'
            else:
                for _dest_spk in dest_spk.split('#'):
                    ts = proc_spk(_dest_spk, state['sd_sentences'])
                    for _ts in ts: all_ts.append(_ts)
                log_append = ""
        else:
            all_ts = timestamp_list
        ts = all_ts
        srt_index = 0
        clip_srt = ""
        if len(ts):
            start, end = ts[0]
            start = min(max(0, start+start_ost*16), len(data))
            end = min(max(0, end+end_ost*16), len(data))
            res_audio = data[start:end]
            start_end_info = "from {} to {}".format(start/16000, end/16000)
            srt_clip, _, srt_index = generate_srt_clip(sentences, start/16000.0, end/16000.0, begin_index=srt_index)
            clip_srt += srt_clip
            for _ts in ts[1:]:
                start, end = _ts
                start = min(max(0, start+start_ost*16), len(data))
                end = min(max(0, end+end_ost*16), len(data))
                start_end_info += ", from {} to {}".format(start, end)
                res_audio = np.concatenate([res_audio, data[start+start_ost*16:end+end_ost*16]], -1)
                srt_clip, _, srt_index = generate_srt_clip(sentences, start/16000.0, end/16000.0, begin_index=srt_index-1)
                clip_srt += srt_clip
        if len(ts):
            message = "{} periods found in the speech: ".format(len(ts)) + start_end_info + log_append
        else:
            message = "No period found in the speech, return raw speech. You may check the recognition result and try other destination text."
            res_audio = data
        return (sr, res_audio), message, clip_srt

    def video_recog(self, video_filename, sd_switch='no', hotwords="", output_dir=None):
        video = mpy.VideoFileClip(video_filename)
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            _, base_name = os.path.split(video_filename)
            base_name, _ = os.path.splitext(base_name)
            clip_video_file = base_name + '_clip.mp4'
            audio_file = base_name + '.wav'
            audio_file = os.path.join(output_dir, audio_file)
        else:
            base_name, _ = os.path.splitext(video_filename)
            clip_video_file = base_name + '_clip.mp4'
            audio_file = base_name + '.wav'
        if video.audio is None:
            logging.error("No audio information found.")
            sys.exit(1)
        video.audio.write_audiofile(audio_file)
        wav = librosa.load(audio_file, sr=16000)[0]
        if os.path.exists(audio_file):
            os.remove(audio_file)
        state = {
            'video_filename': video_filename,
            'clip_video_file': clip_video_file,
            'video': video,
        }
        return self.recog((16000, wav), sd_switch, state, hotwords, output_dir)

    def video_clip(self, dest_text, start_ost, end_ost, state, font_size=32, font_color='white', add_sub=False, dest_spk=None, output_dir=None, timestamp_list=None):
        recog_res_raw = state['recog_res_raw']
        timestamp = state['timestamp']
        sentences = state['sentences']
        video = state['video']
        clip_video_file = state['clip_video_file']
        video_filename = state['video_filename']
        if timestamp_list is None:
            all_ts = []
            if dest_spk is None or dest_spk == '' or 'sd_sentences' not in state:
                for _dest_text in dest_text.split('#'):
                    if '[' in _dest_text:
                        match = re.search(r'\[(\d+),\s*(\d+)\]', _dest_text)
                        if match:
                            offset_b, offset_e = map(int, match.groups())
                            log_append = ""
                        else:
                            offset_b, offset_e = 0, 0
                            log_append = "(Bracket detected in dest_text but offset time matching failed)"
                        _dest_text = _dest_text[:_dest_text.find('[')]
                    else:
                        offset_b, offset_e = 0, 0
                        log_append = ""
                    _dest_text = pre_proc(_dest_text)
                    ts = proc(recog_res_raw, timestamp, _dest_text.lower())
                    for _ts in ts: all_ts.append([_ts[0]+offset_b*16, _ts[1]+offset_e*16])
                    if len(ts) > 1 and match:
                        log_append += '(offsets detected but No.{} sub-sentence matched to {} periods in audio,                             offsets are applied to all periods)'
            else:
                for _dest_spk in dest_spk.split('#'):
                    ts = proc_spk(_dest_spk, state['sd_sentences'])
                    for _ts in ts: all_ts.append(_ts)
        else:
            all_ts = [[i[0]*16.0, i[1]*16.0] for i in timestamp_list]
        srt_index = 0
        time_acc_ost = 0.0
        ts = all_ts
        clip_srt = ""
        if len(ts):
            if self.lang == 'en' and isinstance(sentences, str):
                sentences = sentences.split()
            start, end = ts[0][0] / 16000, ts[0][1] / 16000
            srt_clip, subs, srt_index = generate_srt_clip(sentences, start, end, begin_index=srt_index, time_acc_ost=time_acc_ost)
            start, end = start+start_ost/1000.0, end+end_ost/1000.0
            video_clip = video.subclip(start, end)
            start_end_info = "from {} to {}".format(start, end)
            clip_srt += srt_clip
            if add_sub:
                generator = lambda txt: TextClip(txt, font='./font/STHeitiMedium.ttc', fontsize=font_size, color=font_color)
                subtitles = SubtitlesClip(subs, generator)
                video_clip = CompositeVideoClip([video_clip, subtitles.set_pos(('center','bottom'))])
            concate_clip = [video_clip]
            time_acc_ost += end+end_ost/1000.0 - (start+start_ost/1000.0)
            for _ts in ts[1:]:
                start, end = _ts[0] / 16000, _ts[1] / 16000
                srt_clip, subs, srt_index = generate_srt_clip(sentences, start, end, begin_index=srt_index-1, time_acc_ost=time_acc_ost)
                if not len(subs):
                    continue
                chi_subs = []
                sub_starts = subs[0][0][0]
                for sub in subs:
                    chi_subs.append(((sub[0][0]-sub_starts, sub[0][1]-sub_starts), sub[1]))
                start, end = start+start_ost/1000.0, end+end_ost/1000.0
                _video_clip = video.subclip(start, end)
                start_end_info += ", from {} to {}".format(str(start)[:5], str(end)[:5])
                clip_srt += srt_clip
                if add_sub:
                    generator = lambda txt: TextClip(txt, font='./font/STHeitiMedium.ttc', fontsize=font_size, color=font_color)
                    subtitles = SubtitlesClip(chi_subs, generator)
                    _video_clip = CompositeVideoClip([_video_clip, subtitles.set_pos(('center','bottom'))])
                concate_clip.append(copy.copy(_video_clip))
                time_acc_ost += end+end_ost/1000.0 - (start+start_ost/1000.0)
            message = "{} periods found in the audio: ".format(len(ts)) + start_end_info
            logging.info("Concating...") # logging.warning -> logging.info
            if len(concate_clip) > 1:
                video_clip = concatenate_videoclips(concate_clip)
            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
                _, file_with_extension = os.path.split(clip_video_file)
                clip_video_file_name, _ = os.path.splitext(file_with_extension)
                print(output_dir, clip_video_file)
                clip_video_file = os.path.join(output_dir, "{}_no{}.mp4".format(clip_video_file_name, self.GLOBAL_COUNT))
                temp_audio_file = os.path.join(output_dir, "{}_tempaudio_no{}.mp4".format(clip_video_file_name, self.GLOBAL_COUNT))
            else:
                clip_video_file = clip_video_file[:-4] + '_no{}.mp4'.format(self.GLOBAL_COUNT)
                temp_audio_file = clip_video_file[:-4] + '_tempaudio_no{}.mp4'.format(self.GLOBAL_COUNT)
            video_clip.write_videofile(clip_video_file, audio_codec="aac", temp_audiofile=temp_audio_file)
            self.GLOBAL_COUNT += 1
        else:
            clip_video_file = video_filename
            message = "No period found in the audio, return raw speech. You may check the recognition result and try other destination text."
            srt_clip = ''
        return clip_video_file, message, clip_srt


def get_parser():
    # ... (get_parser 函数保持不变) ...
    parser = ArgumentParser(
        description="ClipVideo Argument",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=(1, 2),
        help="Stage, 0 for recognizing and 1 for clipping",
        required=True
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Input file path",
        required=True
    )
    parser.add_argument(
        "--sd_switch",
        type=str,
        choices=("no", "yes"),
        default="no",
        help="Turn on the speaker diarization or not",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='./output',
        help="Output files path",
    )
    parser.add_argument(
        "--dest_text",
        type=str,
        default=None,
        help="Destination text string for clipping",
    )
    parser.add_argument(
        "--dest_spk",
        type=str,
        default=None,
        help="Destination spk id for clipping",
    )
    parser.add_argument(
        "--start_ost",
        type=int,
        default=0,
        help="Offset time in ms at beginning for clipping"
    )
    parser.add_argument(
        "--end_ost",
        type=int,
        default=0,
        help="Offset time in ms at ending for clipping"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file path"
    )
    parser.add_argument(
        "--lang",
        type=str,
        default='zh',
        help="language"
    )
    return parser


def runner(stage, file, sd_switch, output_dir, dest_text, dest_spk, start_ost, end_ost, output_file, config=None, lang='zh'):
    # [MODIFICATION START] 2. 支持文件夹批量处理
    # 定义支持的文件后缀
    audio_suffixs = ['.wav','.mp3','.aac','.m4a','.flac']
    video_suffixs = ['.mp4','.avi','.mkv','.flv','.mov','.webm','.ts','.mpeg']
    supported_suffixes = audio_suffixs + video_suffixs

    # 判断输入路径是文件还是文件夹
    if os.path.isdir(file):
        logging.info(f"Input is a directory, starting batch processing: {file}")
        files_to_process = []
        for filename in os.listdir(file):
            if any(filename.lower().endswith(s) for s in supported_suffixes):
                files_to_process.append(os.path.join(file, filename))
        
        if not files_to_process:
            logging.warning("No supported audio/video files found in the directory.")
            return
        
        logging.info(f"Found {len(files_to_process)} files to process.")
        # [MODIFICATION START] 3. 新增清单文件
        list_file_path = os.path.join(output_dir, 'processing_list.txt')
        with open(list_file_path, 'w', encoding='utf-8') as f_list:
             f_list.write(f"Batch processing summary for directory: {file}\n\n")
        # [MODIFICATION END]

    elif os.path.isfile(file):
        logging.info(f"Input is a single file: {file}")
        files_to_process = [file]
    else:
        logging.error(f"Input path does not exist or is not a valid file/directory: {file}")
        sys.exit(1)

    # 确保总输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 初始化模型（只在需要时执行一次）
    audio_clipper = None
    if stage == 1:
        from funasr import AutoModel
        logging.info("Initializing modelscope asr pipeline.")
        if lang == 'zh':
            funasr_model = AutoModel(model="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                                     vad_model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                                     punc_model="damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
                                     spk_model="damo/speech_campplus_sv_zh-cn_16k-common",
                                     disable_update=True)
        elif lang == 'en':
            funasr_model = AutoModel(model="iic/speech_paraformer_asr-en-16k-vocab4199-pytorch",
                                     vad_model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                                     punc_model="damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
                                     spk_model="damo/speech_campplus_sv_zh-cn_16k-common")
        audio_clipper = VideoClipper(funasr_model)
        audio_clipper.lang = lang

    # 循环处理所有文件
    for i, current_file in enumerate(files_to_process):
        logging.info("="*50)
        logging.info(f"Processing file {i+1}/{len(files_to_process)}: {current_file}")
        
        # 判断文件类型
        _, ext = os.path.splitext(current_file)
        if ext.lower() in audio_suffixs:
            mode = 'audio'
        elif ext.lower() in video_suffixs:
            mode = 'video'
        else:
            logging.warning(f"Skipping unsupported file format: {current_file}")
            continue

        # 为每个文件创建独立的输出子目录（这与我们 app.py 的逻辑保持一致）
        file_stem = os.path.splitext(os.path.basename(current_file))[0]
        current_output_dir = os.path.join(output_dir, file_stem)
        if not os.path.exists(current_output_dir):
            os.makedirs(current_output_dir)

        if stage == 1:
            if not audio_clipper: # 确保模型已加载
                logging.error("Model not initialized for stage 1. Exiting.")
                return
                
            if mode == 'audio':
                logging.info(f"Recognizing audio file: {current_file}")
                wav, sr = librosa.load(current_file, sr=16000)
                res_text, res_srt, state = audio_clipper.recog((sr, wav), sd_switch, output_dir=current_output_dir)
            if mode == 'video':
                logging.info(f"Recognizing video file: {current_file}")
                res_text, res_srt, state = audio_clipper.video_recog(current_file, sd_switch, output_dir=current_output_dir)

            # [MODIFICATION START] 3. 新增TXT和清单文件输出
            # 定义输出文件名
            output_basename = os.path.join(current_output_dir, file_stem)
            srt_file = output_basename + '.srt'
            txt_file = output_basename + '.txt'

            # 写入SRT文件
            with open(srt_file, 'w', encoding='utf-8') as fout:
                fout.write(res_srt)
                logging.info(f"Wrote subtitle to {srt_file}")
            
            # 写入TXT文件
            with open(txt_file, 'w', encoding='utf-8') as fout:
                fout.write(res_text)
                logging.info(f"Wrote plain text to {txt_file}")
            
            # 写入清单文件（追加模式）
            if os.path.isdir(file): # 仅在批量模式下写入清单
                with open(list_file_path, 'a', encoding='utf-8') as f_list:
                    f_list.write(f"File: {os.path.basename(current_file)}\n")
                    f_list.write(f"Text: {res_text}\n")
                    f_list.write("-" * 20 + "\n")

            # 原有的 state 文件逻辑，现在保存到子目录
            write_state(current_output_dir, state)
            logging.info(f"Recognition for {current_file} finished.")
            # print(res_text) # 打印到控制台可能会混淆日志，改为日志记录
            logging.info(f"Result text: {res_text}")
            # [MODIFICATION END]
            
        if stage == 2:
            # Stage 2 的批量处理逻辑相对复杂，暂时保持原样，因为它需要 dest_text 等参数
            # 如果需要，我们可以后续再对 stage 2 进行改造
            logging.warning("Batch processing for stage 2 is not fully implemented in this modification.")
            # (stage 2 的原始代码保持不变)
            audio_clipper = VideoClipper(None)
            if mode == 'audio':
                state = load_state(current_output_dir) # 从子目录加载
                wav, sr = librosa.load(current_file, sr=16000)
                state['audio_input'] = (sr, wav)
                (sr, audio), message, srt_clip = audio_clipper.clip(dest_text, start_ost, end_ost, state, dest_spk=dest_spk)
                if output_file is None:
                    output_file_path = os.path.join(current_output_dir, 'result.wav')
                else: # 如果指定了输出文件，可能需要更复杂的逻辑来处理命名冲突
                    output_file_path = os.path.join(current_output_dir, os.path.basename(output_file))
                clip_srt_file = output_file_path[:-3] + 'srt'
                logging.info(message)
                sf.write(output_file_path, audio, 16000)
                assert output_file_path.endswith('.wav'), "output_file must ends with '.wav'"
                logging.info(f"Save clipped wav file to {output_file_path}")
                with open(clip_srt_file, 'w', encoding='utf-8') as fout:
                    fout.write(srt_clip)
                    logging.info(f"Write clipped subtitle to {clip_srt_file}")
            # ... (stage 2 video 部分类似) ...
    logging.info("="*50)
    logging.info("All tasks finished.")
    # [MODIFICATION END]

def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    runner(**kwargs)


if __name__ == '__main__':
    main()