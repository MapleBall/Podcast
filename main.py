from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import BaseRetriever
from langchain.llms.base import BaseLLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLMResult
from langchain_community.vectorstores import FAISS
from langchain.schema.runnable import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
import os
import torch
import gradio as gr
import requests
import json
import heapq
from typing import Any, List, Mapping, Optional
from pydantic import Field

k = 5
fetch_k = 100

def create_embeddings(use_cpu=False):
    device = "cpu" if use_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': device}
    )

def load_FAISS_vectorstore(vectorstore_path, embeddings):
    if os.path.exists(vectorstore_path):
        try:
            vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
            print(f"Loaded vector store from {vectorstore_path}")
            return vectorstore
        except Exception as e:
            print(f"Error loading vector store from {vectorstore_path}: {str(e)}")
    else:
        print(f"Vector store not found at {vectorstore_path}")
    return None

def load_vectorstores_from_directory(parent_directory, embeddings):
    vectorstores = []
    for root, dirs, files in os.walk(parent_directory):
        if 'index.faiss' in files and 'index.pkl' in files:
            vectorstore_path = root
            vs = load_FAISS_vectorstore(vectorstore_path, embeddings)
            if vs:
                vectorstores.append(vs)
    return vectorstores

def retrieve_from_multiple_stores(vectorstores, query, k=5, fetch_k=100):
    all_results = []
    for vs in vectorstores:
        results = vs.max_marginal_relevance_search(query, k=fetch_k, fetch_k=fetch_k)
        scored_results = vs.similarity_search_with_score(query, k=len(results))
        all_results.extend(scored_results)
    
    return [doc for doc, score in heapq.nsmallest(k, all_results, key=lambda x: x[1])]

### 定義ollama類別
class ChatOllama(BaseLLM):
    model_name: str = "llama3.1:latest"
    url: str = "http://localhost:11434/api/generate"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        data = {
            "model": self.model_name,
            "prompt": prompt
        }
        
        response = requests.post(self.url, json=data)
        if response.status_code == 200:
            full_response = ""
            for line in response.text.split('\n'):
                if line:
                    try:
                        json_response = json.loads(line)
                        full_response += json_response.get('response', '')
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON: {line}")
            return full_response
        else:
            raise RuntimeError(f"Error: {response.status_code}\n{response.text}")

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        generations = []
        for prompt in prompts:
            response = self._call(prompt, stop, run_manager, **kwargs)
            generations.append([{"text": response}])
        return LLMResult(generations=generations)

    @property
    def _llm_type(self) -> str:
        return "chat_ollama"

def setup_qa_chain(use_cpu=False):
    # groq_api_key = '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'
    # model = 'llama-3.1-8b-instant'
    # ollama_chat = ChatGroq(groq_api_key=groq_api_key, model_name=model)
    ollama_chat = ChatOllama(model_name='llama3:8b') #可換模型
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    parent_directory = r"D:\Podcast_mp3_save\VectoreStore_1"  # 更新為您的向量庫目錄
    embeddings = create_embeddings(use_cpu)
    vectorstores = load_vectorstores_from_directory(parent_directory, embeddings)
    
    class CustomRetriever(BaseRetriever):
        vectorstores: list = Field(default_factory=list)
    
        def __init__(self, vectorstores):
            super().__init__()
            self.vectorstores = vectorstores
    
        def _get_relevant_documents(self, query):
            results = retrieve_from_multiple_stores(self.vectorstores, query, k=k, fetch_k=fetch_k)
            return results

    custom_retriever = CustomRetriever(vectorstores)

    template = """您是一個專業的 Podcast 搜尋引擎助手。您的任務是使用 RAG（檢索增強生成）技術回答有關 Podcast 節目內容的問題。請仔細閱讀以下指南，並據此回答問題。

當前對話歷史：
{chat_history}

檢索到的相關資訊：
{context}

使用者當前問題：
{question}

回答指南：
1. **上下文理解**：
   - 仔細分析當前問題與之前對話的關聯。
   - 如果當前問題是對之前對話的延續，請確保利用先前的資訊來提供連貫的回答。

2. **資訊使用**：
   - 主要使用檢索到的資訊（{context}）來回答問題。
   - 如果檢索的資訊不足，可以參考之前的對話歷史來補充。
   - 若兩者都無法提供足夠資訊，請回答「抱歉，RAG 資料庫中沒有足夠的資訊來回答這個問題」。

3. **回答結構**：
   - 具體內容要點：提供明確、相關的內容要點。
   - 時間戳：為每個內容要點提供時間戳，格式為（MM:SS~MM:SS）或（MM:SS）。
   - 來源區分：清楚區分不同 Podcast 節目的觀點，並提供該集標題。
   - 節目標題：在回答末尾提供完整的節目標題，格式為（節目標題：[完整標題]）。

4. **回答格式示例**：
   「根據檢索資料和先前的對話，[內容摘要1]（時間戳）。此外，[內容摘要2]（時間戳）。[繼續列舉其他相關內容]。（節目標題：[完整標題]）」

5. **語言和風格**：
   - 使用繁體中文回答。
   - 保持清晰、詳細且專業的語氣。

6. **資訊限制**：
   - 僅使用檢索到的資料和對話歷史中的資訊。
   - 不要添加任何未在這些來源中提及的資訊。

7. **格式注意事項**：
   - 不使用刪除線或其他特殊格式標記。
   - 保持回答的結構清晰易讀。

請根據以上指南，回答使用者的問題：
"""

    document_prompt = PromptTemplate(
        input_variables=["page_content", "episode_name", "Podcast_name"],
        template="內容: {page_content}\n來源: {episode_name}, {Podcast_name}"
    )
    prompt = ChatPromptTemplate.from_template(template)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ollama_chat,
        retriever=custom_retriever,
        memory=memory,
        verbose=True,  # 添加這行來幫助調試
        combine_docs_chain_kwargs={
            "prompt": prompt,
            "document_variable_name": "context",
            "document_prompt": document_prompt
        }
    )

    return qa_chain, custom_retriever

def main(use_cpu=False):
    qa_chain, retriever = setup_qa_chain(use_cpu)

    def get_program_list(folder_path):
        try:
            programs = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
            program_list = "\n".join(f"{i + 1}: {program}" for i, program in enumerate(programs))
            return program_list
        except FileNotFoundError:
            return "指定的資料夾不存在。"
        except Exception as e:
            return f"發生錯誤: {e}"

    def chat_function(message, history):
        try:
            # 使用檢索器獲取相關文檔
            results = retriever._get_relevant_documents(message)

            # 使用鏈進行預測
            response = qa_chain({"question": message})
            answer = response['answer']

            # 使用集合來存儲唯一的 (episode_name, podcast_name) 組合
            unique_sources = set()
            for result in results:
                episode_name = result.metadata.get('episode_name', 'Unknown Episode')
                podcast_name = result.metadata.get('Podcast_name', 'Unknown Podcast')
                unique_sources.add((episode_name, podcast_name))

            # 格式化 sources 字符串
            sources_str = "\n可參考下方節目集數：\n"
            for idx, (episode_name, podcast_name) in enumerate(unique_sources, 1):
                sources_str += f"Result {idx}: {episode_name}, {podcast_name}\n"

            # 將答案和來源信息合併為一個字符串
            full_response = f"{answer}\n\n{sources_str}"
            # 打印記憶內容以進行調試
            print("Current memory contents:")
            print(qa_chain.memory.chat_memory.messages)

            return full_response

        except Exception as e:
            error_message = f"發生錯誤: {str(e)}\n很抱歉，我無法處理您的問題。請再試一次或換個問題。"
            return error_message

    with gr.Blocks() as iface:
        gr.Markdown(f"## 目前資料庫中的節目有：\n{get_program_list('/media/starklab/BACKUP/Podcast_project/轉錄文本存放區')}\n\n請在下方提問：")

        chatbot = gr.ChatInterface(
            chat_function,
            title="Podcast Q&A Assistant",
            description="Ask questions about podcast content, and I'll provide answers based on the retrieved information.",
            theme="soft",
            examples=[
                "林書豪這個賽季遇到了什麼困難？",
                "請告訴我這個節目討論了哪些主題？",
                "這集節目中有提到哪些重要的觀點？"
            ],
            retry_btn="重試",
            undo_btn="撤銷",
            clear_btn="清除"
        )

    iface.launch(share=True)

if __name__ == "__main__":
    use_cpu = False  # 設置為 True 以使用 CPU，False 則使用 GPU（如果可用）
    main(use_cpu)