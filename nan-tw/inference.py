
from transformers import pipeline, AutoTokenizer, AutoModelForSpeechSeq2Seq, AutoProcessor

# 路徑設定
tsv_path = "./nan-tw/test.tsv"
clips_dir = "./nan-tw"
output_path = "output_translate.tsv"

# 載入模型和 processor
model_name = "./whisper-nan-tw/final-checkpoint"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)



# 使用 pipeline 載入模型
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=tokenizer,
    feature_extractor=processor.feature_extractor,
    device=0  # 使用 GPU 0 (如果是外顯 GPU，改成 device=1)
  
)

# 讀取資料
import pandas as pd
from tqdm import tqdm
import os
from jiwer import cer

df = pd.read_csv(tsv_path, sep="\t")
results = []

# CER 累計變數
total_cer = 0.0
count = 0

for _, row in tqdm(df.iterrows(), total=len(df)):
    rel_audio_path = row["path"].replace("\\", "/")
    full_audio_path = os.path.join(clips_dir, rel_audio_path)
    ref_text = row.get("sentence", "").strip()

    if not os.path.exists(full_audio_path):
        print(f"檔案不存在: {full_audio_path}")
        continue

    # 使用 pipeline 進行音訊辨識
    try:
        pred_text = pipe(full_audio_path)["text"].strip()
    except Exception as e:
        print(f"音訊處理失敗: {full_audio_path}, 錯誤: {e}")
        pred_text = ""

    # 計算 CER
    try:
        sample_cer = cer(ref_text, pred_text)
        total_cer += sample_cer
        count += 1
    except:
        sample_cer = "N/A"

    results.append((rel_audio_path, ref_text, pred_text, sample_cer))

# 輸出結果表
out_df = pd.DataFrame(results, columns=["path", "reference", "prediction", "cer"])
out_df.to_csv(output_path, sep="\t", index=False, encoding="utf-8")
print(f"✅ 推論完成，結果已輸出至：{output_path}")

# 平均 CER
if count > 0:
    avg_cer = total_cer / count
    print(f"📊 平均 CER：{avg_cer:.4f}")
else:
    print("⚠️ 沒有成功計算 CER 的樣本")