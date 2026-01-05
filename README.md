# TCAI: A Domain-Specific AI Assistant for Turtle Care
![畫畫](https://github.com/user-attachments/assets/f37dd31d-4b8f-488c-a797-ee5aeea21a70)
TCAI (Turtle Cares AI) 是一個專門為烏龜養護與健康管理設計的領域特定型 AI 智能助手 。本專案結合了大型語言模型（LLM）架構與檢索增強生成（RAG）技術，旨在解決寵物烏龜資訊破碎且不準確的問題，提供專業、可靠且即時的照護建議 。

專案背景
在特殊寵物照護領域中，高品質的數據集非常稀缺 。烏龜的飼養資訊往往散見於論壇、手冊與片段的專業報告，導致飼主難以獲取正確知識，進而影響烏龜健康 。TCAI 透過系統化整理專業知識，有效降低 LLM 常見的「幻覺（Hallucination）」問題，提升專業性 。
<img width="1564" height="906" alt="image" src="https://github.com/user-attachments/assets/a5d06b17-53cc-452a-b9e5-7a75a388d776" />

🛠️ 核心技術架構
本系統基於 LLaMA 3.1-8B 基礎模型，並應用以下優化技術 ：
1.超級微調 (SFT)：使用手動標記的問答對進行訓練，使模型學習標準化的回答格式與專業邏輯 。
2.低秩適配 (LoRA)：實現參數高效的領域適配，讓模型能快速吸收烏龜專業知識而無需重訓整個網絡 。
3.4-bit 量化 (4-bit Quantization)：降低記憶體使用量與推論成本，提升在一般硬體設備上的部署可行性 。

📊 資料集 (目前有291筆)
我們構建了一個涵蓋四大領域的高質量烏龜照護資料庫 ：
1.品種特性：體型、生長速度、壽命與行為模式 。
2.飼養環境：水質管理、溫度控制、UVB 燈光需求與棲息地佈置 。
3.飲食習性：營養比例、推薦/禁忌食物及維生素補充 。
4.常見疾病：呼吸道感染、眼部發炎、軟殼症及寄生蟲防治建議 。



檢索增強生成 (RAG)：結合向量數據庫，即時檢索相關知識片段，確保回答具備事實依據 。

介面: Streamlit </br>
檢索技術:Embedding models mxbai-embed-large</br>
模型:llama3.1</br>

==建虛擬環境==</br>
python版本3.12</br>
cd C:\Users\YourName\YourProject</br>
python -m venv myenv(環境名稱)</br>
myenv(環境名稱)\Scripts\activate</br>

==安裝步驟== </br>
pip install requests </br>
ollama pull mxbai-embed-large (安裝嵌入模型) </br>
ollama pull meta-llama/Meta-Llama-3.1(安裝llama3模型) </br>
pip install chromadb安裝chromadb(向量庫) </br>
pip install streamlit </br>
pip install ollama </br>
pip install openpyxl </br>

==開啟== </br>
**記得先打開ollma </br>
進入環境:myenv\Scripts\activate </br>
啟動: streamlit run temp.py </br>


![image](https://github.com/user-attachments/assets/1c311b3f-7bfe-4e75-9742-4e9c2ba04f15)
