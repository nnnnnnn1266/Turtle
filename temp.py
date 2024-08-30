import streamlit as st  # 導入Streamlit庫，用於建立網頁應用
import ollama  # 導入ollama庫，用於自然語言處理
import chromadb  # 導入chromadb庫，用於數據存儲和查詢
import pandas as pd  # 導入pandas庫，用於數據分析和處理

# 定義一個初始化函數，用於設置Streamlit的會話狀態
def initialize():
    if "already_executed" not in st.session_state:
        st.session_state.already_executed = False

    if not st.session_state.already_executed:
        setup_database()

# 定義設置資料庫的函數
def setup_database():
    client = chromadb.Client()  # 創建一個chromadb的客戶端，用於與資料庫交互
    file_path = '烏龜問題集測試rag2.xlsx'  # 指定Excel文件的路徑和名稱
    documents = pd.read_excel(file_path, header=None)  # 使用pandas讀取Excel文件

    collection = client.get_or_create_collection(name="demodocs")

    for index, content in documents.iterrows():
        response = ollama.embeddings(prompt=content[0], model="llama3.1:latest")  # 確保模型名稱正確
        collection.add(ids=[str(index)], embeddings=[response["embedding"]], documents=[content[0]])

    st.session_state.already_executed = True
    st.session_state.collection = collection

# 定義創建新chromadb客戶端的函數，每次需要時創建新的連接
def create_chromadb_client():
    return chromadb.Client()

# 主函數，負責構建UI和處理用戶事件
def main():
    initialize()
    st.title("歡迎來到Turtle知識問答")
    user_input = st.text_area("您想問什麼？", "")

    if st.button("送出"):
        if user_input:
            handle_user_input(user_input, st.session_state.collection)
        else:
            st.warning("請輸入問題！")

# 定義處理用戶輸入的函數
def handle_user_input(user_input, collection):
    response = ollama.embeddings(prompt=user_input, model="llama3.1:latest")  # 確保模型名稱正確
    results = collection.query(query_embeddings=[response["embedding"]], n_results=3)
    data = results['documents'][0]
    output = ollama.generate(
        model="llama3.1:latest",  # 確保模型名稱正確
        prompt=f"Using this data: {data}. Respond to this prompt and use Chinese: {user_input}"
    )
    st.text("回答：")
    st.write(output['response'])

if __name__ == "__main__":
    main()
