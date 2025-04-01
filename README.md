# Podcast Retrieval Chatbot with ollama

## 安裝 Ollama
### 可以在 [Ollama官網](https://ollama.com/download) 依不同作業系統做安裝
1. 執行 Ollama 服務
```bash
ollama serve
```
2. 下載所需要的LLM
Example :
```bash
ollama pull llama3:8b
```
3. 可檢查是否有安裝成功
```bash
ollama list
```

## 安裝說明
以下安裝皆在 Python==3.8 測試

1. 安裝 PyTorch:
   請訪問 [PyTorch 官網](https://pytorch.org/get-started/locally/) 並根據您的系統和 CUDA 版本選擇適當的安裝指令。
   
2. 安装 ffmpeg (For Voice2Text)
   若ffmpeg已安裝則跳過。

### Ubuntu/Debian 系統
```bash
sudo apt update && sudo apt install ffmpeg
```
### Windows 系統

- 下載[ffmpeg.exe](https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffmpeg.exe)

3. 安裝其他依賴:
> 建議 conda 創建一個 python3.8 虛擬環境後，執行以下指令
```python
pip install -r requirements.txt
```
  FAISS 安裝 (Optional):
   - 對於 CPU 版本: pip install faiss-cpu
   - 對於 GPU 版本 (需要 CUDA 支持): pip install faiss-gpu

注意: 
- 確保安裝的 PyTorch 版本與 transformers 庫兼容。如果遇到兼容性問題，可能需要調整 transformers 的版本。
- 如果您使用 GPU 版本的 FAISS，請確保您的 CUDA 環境與 PyTorch 和 FAISS 兼容。

### 執行主程式
```python
python main.py
```







