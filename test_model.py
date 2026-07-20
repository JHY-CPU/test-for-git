import os
MODEL_CACHE_DIR = os.path.join(os.path.dirname(__file__), "funasr_models")
os.environ["MODELSCOPE_CACHE"] = MODEL_CACHE_DIR
os.environ["HF_HOME"] = MODEL_CACHE_DIR
import re
import json
import logging
from funasr import AutoModel

logging.getLogger("modelscope_hub.download").setLevel(logging.WARNING)

# 加载模型（SenseVoiceSmall + 增强说话人分离）
model = AutoModel(
    model="iic/SenseVoiceSmall",
    vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    spk_model="iic/speech_campplus_sv_zh-cn_16k-common",
    vad_kwargs={"max_single_segment_time": 10000},
    device="cuda:0",
    disable_update=True,
)

# 识别音频
res = model.generate(
    input=r"C:\Users\21308\OneDrive\Desktop\TEST\TEST\tts_test1702\tts_test1702.mp3",
    language="zh",
    use_itn=True,
    batch_size=15,          # 如果爆显存，改成 10
    batch_size_type="sample",
)

# --- 以下构建 JSON 的代码完全不变 ---
def format_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{m:02d}:{s:02d}.{ms:03d}"

result = {
    "audio_segments": [],
    "terminated_by_manager": False,
    "end_call_signal_detected": False,
    "termination_reason": "",
    "terminator": ""
}

speaker_role: dict[int, str] = {}

# 第一步：收集所有原始片段
raw_segments = []
if res and "sentence_info" in res[0]:
    for seg in res[0]["sentence_info"]:
        start_sec = seg.get('start', 0) / 1000.0
        end_sec = seg.get('end', 0) / 1000.0
        speaker_id = seg.get('spk', 0)
        raw_text = seg.get('sentence', '')
        clean_text = re.sub(r'<[^>]+>', '', raw_text).strip()

        if not clean_text:
            continue

        if speaker_id not in speaker_role:
            speaker_role[speaker_id] = f"speaker_{len(speaker_role) + 1}"

        raw_segments.append({
            "role": speaker_role[speaker_id],
            "content": clean_text,
            "start_time_seconds": round(start_sec, 3),
            "end_time_seconds": round(end_sec, 3),
        })

    # 第二步：合并同一说话者的连续片段（以对话轮次为单位）
    if raw_segments:
        merged = raw_segments[0].copy()
        for seg in raw_segments[1:]:
            if seg["role"] == merged["role"]:
                # 同一说话者：合并文本，延长结束时间
                merged["content"] += seg["content"]
                merged["end_time_seconds"] = seg["end_time_seconds"]
            else:
                # 说话者切换：保存上一轮，开始新一轮
                result["audio_segments"].append({
                    "role": merged["role"],
                    "content": merged["content"],
                    "start_time": format_time(merged["start_time_seconds"]),
                    "end_time": format_time(merged["end_time_seconds"]),
                    "start_time_seconds": merged["start_time_seconds"],
                    "end_time_seconds": merged["end_time_seconds"],
                })
                merged = seg.copy()
        # 别忘了最后一段
        result["audio_segments"].append({
            "role": merged["role"],
            "content": merged["content"],
            "start_time": format_time(merged["start_time_seconds"]),
            "end_time": format_time(merged["end_time_seconds"]),
            "start_time_seconds": merged["start_time_seconds"],
            "end_time_seconds": merged["end_time_seconds"],
        })

elif res:
    raw_text = res[0].get("text", "")
    clean_text = re.sub(r'<[^>]+>', '', raw_text).strip()
    result["audio_segments"].append({
        "role": "speaker_1",
        "content": clean_text,
        "start_time": "00:00.000",
        "end_time": "00:00.000",
        "start_time_seconds": 0,
        "end_time_seconds": 0
    })

print(json.dumps(result, ensure_ascii=False, indent=2))
