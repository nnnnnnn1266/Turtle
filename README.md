# TCAI
烏龜問答
![畫畫](https://github.com/user-attachments/assets/f37dd31d-4b8f-488c-a797-ee5aeea21a70)


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
