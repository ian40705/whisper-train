# 🗣️ Whisper-TW (台語語音辨識訓練)

本專案為基於 OpenAI Whisper 模型進行台語（閩南語）語音辨識的微調(fine-tune)實作

## 🧠 使用模型
訓練所使用的預訓練模型為：
- [`Jobaula/whisper-medium-nan-tw-common-voice`](https://huggingface.co/Jobaula/whisper-medium-nan-tw-common-voice)（模型）

## 📂 專案結構

whisper-train/
├── whisper.py # 訓練腳本
├── inference.py # 推論腳本
├── nan-tw/ # 台語音訊資料夾
│ ├── clips/ # 音訊資料
│ └── train.tsv # 標註文字檔
└── README.md # 專案說明文件


## 🛠️ 訓練步驟

```bash
# 安裝必要套件
pip install torch transformers datasets librosa evaluate gradio

# 執行訓練
python whisper.py
