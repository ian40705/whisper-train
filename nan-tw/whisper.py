import os
import numpy as np
import pandas as pd
import librosa
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from datasets import Dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import gradio as gr
import evaluate

# 1. 設定資料夾路徑（請改成你本機的路徑）
data_dir = "C:/Users/USER/Desktop/nan-tw"   # nan-tw 資料夾，內含 clips 和 train.tsv

# 2. 讀取 train/test.tsv
df = pd.read_csv(os.path.join(data_dir, "test.tsv"), sep="\t")

# 3. 修正 Windows 路徑（確保用正斜線）
df["path"] = df["path"].str.replace("\\", "/", regex=False)

# 4. 產生完整音訊路徑
df["audio"] = df["path"].apply(lambda x: os.path.join(data_dir, "clips", x.split('/')[-1]))
df["text"] = df["sentence"]
df = df[["audio", "text"]]

# 5. 分割訓練與驗證集
split = int(len(df) * 0.9)
df_train = df[:split].reset_index(drop=True)
df_val = df[split:].reset_index(drop=True)

# 6. 建立 Dataset 物件
train_ds = Dataset.from_pandas(df_train)
val_ds = Dataset.from_pandas(df_val)


import torch

def data_collator(features):
    # 展開 input_features (Whisper 是 batch["input_features"][0] 裡面是 tensor)
    input_features = [torch.tensor(f["input_features"]) if not isinstance(f["input_features"], torch.Tensor) else f["input_features"] for f in features]
    
    # labels 是 list[int]，需補齊長度
    label_features = [torch.tensor(f["labels"], dtype=torch.long) for f in features]
    
    # padding labels（補 -100，代表不計算 loss）
    labels_padded = torch.nn.utils.rnn.pad_sequence(label_features, batch_first=True, padding_value=-100)
    
    batch = {
        "input_features": torch.stack(input_features),
        "labels": labels_padded
    }
    return batch

processor = AutoProcessor.from_pretrained("Jobaula/whisper-medium-nan-tw-common-voice")
model = AutoModelForSpeechSeq2Seq.from_pretrained("Jobaula/whisper-medium-nan-tw-common-voice")

#Jobaula/whisper-medium-nan-tw-common-voice
# 加入 forced_decoder_ids，鎖定語言與任務
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
    task="transcribe"
)


# 8. 檢查 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
model.to(device)

# 9. 定義 preprocess 函數（加入檔案存在檢查）
def preprocess(batch):
    audio_path = batch["audio"]
    audio, _ = librosa.load(audio_path, sr=16000)

    # 特徵提取（Whisper expects log-mel spectrograms）
    features = processor.feature_extractor(audio, sampling_rate=16000).input_features[0]

    # 加上最大長度限制（Whisper 最大輸出長度為 448）
    labels = processor.tokenizer(
        batch["text"],
        max_length=448,
        padding="max_length",   # 如果你要用 Trainer，這樣對齊 padding
        truncation=True         # 重點：截斷過長的文字
    ).input_ids

    return {
        "input_features": features,
        "labels": labels
    }




# 10. 使用 map 預處理 Dataset
train_ds = train_ds.map(preprocess, remove_columns=train_ds.column_names)
val_ds = val_ds.map(preprocess, remove_columns=val_ds.column_names)

# 11. 設定訓練參數
training_args = Seq2SeqTrainingArguments(
  output_dir="./whisper-nan-tw",
    per_device_train_batch_size=1,            # 小一點，加快每步速度
    gradient_accumulation_steps=4,            # 這樣等於有效 batch size = 8
    learning_rate=5e-2,                       # 比 1e-2 小很多，對 Whisper 更穩
    warmup_steps=30,                          # 只預熱一小段即可
    max_steps=5,                            # 測試效果就好，先訓練 100 steps
    logging_steps=5,                          # 更密集觀察 loss
    save_steps=5,                           # 只存一次 checkpoint
    save_total_limit=1,
    fp16=torch.cuda.is_available(),           # 有 GPU 才開 fp16
    report_to="none"

)

# 12. 載入評估指標
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")
import logging
logging.basicConfig(filename='metrics.log', level=logging.INFO)   

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

   
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    logging.info(f"WER: {wer}, CER: {cer}")
    return {
        "wer": wer_metric.compute(predictions=pred_str, references=label_str),
        "cer": cer_metric.compute(predictions=pred_str, references=label_str),
    }

# 13. 建立 Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=data_collator,  
    compute_metrics=compute_metrics,
    tokenizer=processor.tokenizer
)

# 14. 開始訓練
# 開始訓練
trainer.train()

# ✅ 儲存訓練後的模型到指定資料夾
trainer.save_model("./whisper-nan-tw/final-checkpoint")

# ✅ 儲存 processor 和 tokenizer（這樣推論才不會出錯）
processor.save_pretrained("./whisper-nan-tw/final-checkpoint")
if hasattr(processor, "tokenizer"):
    processor.tokenizer.save_pretrained("./whisper-nan-tw/final-checkpoint")

    # 🔚 訓練結束後手動跑一次評估（並輸出結果）
metrics = trainer.evaluate()
print("Final Eval Metrics:", metrics)

