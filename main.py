from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import BaseRetriever
from langchain.llms.base import BaseLLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLMResult
from langchain_community.vectorstores import FAISS
from langchain.schema.runnable import RunnablePassthrough
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import os
import torch
import gradio as gr
import requests
import json
import heapq
from typing import Any, List, Mapping, Optional
from pydantic import Field

# 設置環境變數以禁用 tokenizers 的並行處理
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
    model_name: str = "llama3:8b"
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
    
    ollama_chat = ChatOllama(model_name='llama3:8b') #可換模型
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # 使用新的FAISS檢索邏輯，並傳入 use_cpu 參數
    parent_directory = r"C:\Users\stanl\OneDrive\文件\GitHub\Podcast-\VectoreStore_1"  # 更新為您的向量庫目錄
    embeddings = create_embeddings(use_cpu)
    vectorstores = load_vectorstores_from_directory(parent_directory, embeddings)
    
    class CustomRetriever(BaseRetriever):
        vectorstores: list = Field(default_factory=list)
    
        def __init__(self, vectorstores):
            super().__init__()
            self.vectorstores = vectorstores
    
        def _get_relevant_documents(self, query):
            results = retrieve_from_multiple_stores(self.vectorstores, query, k=k, fetch_k=fetch_k)
            return results  # 這裡應該直接返回文檔列表，而不是元組

    custom_retriever = CustomRetriever(vectorstores)



    template = """我將作為您的Podcast搜尋引擎。當您向我詢問有關特定Podcast節目或內容的問題時，我將使用RAG（檢索增強生成）技術來回答您的問題。請注意，如果RAG檢索庫中沒有您所需的內容，我將告知您「RAG資料庫內沒有您所需的內容」。我希望您根據這些條件提問。

您的第一句話是「嗨」。

檢索資料信息（包括節目標題）：
{context}

聊天歷史：
{chat_history}

當前問題：
{question}

回答指南：
1. **問題處理**：首先對當前問題進行清晰的 prompt engineering，確保理解問題的核心需求。
2. **信息使用**：僅使用檢索資料中的信息來回答問題。如果資料不足以回答問題，請直接回答「RAG 資料庫沒有您想要的資料」。
3. **回答內容**：
   - **具體內容要點**：回答應包括具體的內容要點。
   - **時間戳**：每個內容要點應附上對應的時間戳。請使用完整的格式，例如（MM:SS~MM:SS）。如果只有一個時間點，則使用（MM:SS）。
   - **節目標題**：最後應提供節目標題（格式：（節目標題：[完整標題]））。
4. **回答格式示例**：
   - 「根據檢索資料，[內容摘要1]（時間戳）。此外，[內容摘要2]（時間戳）。[如有更多內容，繼續列舉]。（節目標題：[完整標題]）」
5. **回答語言和風格**：回答要清楚詳細，使用繁體中文。
6. **資訊限制**：不要添加任何檢索資料中沒有的信息。
7. **格式問題**: 請不要使用刪除線或任何其他特殊格式標記在你的回答中。
8. **記憶**: 如果使用者希望接續前面的問答再次提問，系統應該能夠檢索並提供對話紀錄（chat_history），並根據這些紀錄回答使用者的問題。
請根據上述指南回答問題：
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
        combine_docs_chain_kwargs={
            "prompt": prompt,
            "document_variable_name": "context",
            "document_prompt": document_prompt
        }
    )

    return qa_chain, custom_retriever

# 修改主函數以接受 use_cpu 參數
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
            # 檢查新的問題是否與對話歷史相關
            is_related = check_relevance(message, history)

            if not is_related:
                # 不相關則清除
                qa_chain.memory.clear()

            results = retriever.invoke(message)
            response = qa_chain.invoke({"question": message, "chat_history": history if is_related else []})
            answer = response['answer']

            unique_sources = set()
            for result in results:
                episode_name = result.metadata.get('episode_name', 'Unknown Episode')
                podcast_name = result.metadata.get('Podcast_name', 'Unknown Podcast')
                unique_sources.add((episode_name, podcast_name))

            sources_str = "\n可參考下方節目集數：\n"
            for idx, (episode_name, podcast_name) in enumerate(unique_sources, 1):
                sources_str += f"Result {idx}: {episode_name}, {podcast_name}\n"

            full_response = f"{answer}\n\n{sources_str}"

            return full_response

        except Exception as e:
            error_message = f"發生錯誤: {str(e)}\n很抱歉，我無法處理您的問題。請再試一次或換個問題。"
            return error_message
        
    # 檢查新對話與對話歷史是否相關 (關鍵字相似度比較)
    def check_relevance(new_question, history):
        
        if not history:
            return False

        last_question = history[-1][0]  # 獲取最後一個問題
        
        # 關鍵字重疊性比較是否相關
        keywords_last = set(last_question.lower().split())
        keywords_new = set(new_question.lower().split())
        
        overlap = len(keywords_last.intersection(keywords_new))
        threshold = min(len(keywords_last), len(keywords_new)) * 0.3  # 閾值:30%
        
        return overlap >= threshold

    with gr.Blocks() as iface:
        gr.Markdown(f"## 目前資料庫中的節目有：\n{get_program_list('C:/Users/stanl/OneDrive/文件/GitHub/Podcast-/VectoreStore_sample')}\n\n請在下方提問：")

        chatbot = gr.ChatInterface(
            chat_function,
            title="Podcast Q&A Assistant",
            description="Ask questions about podcast content, and I'll provide answers based on the retrieved information.",
            theme="soft",
            # 設定一些快捷問題供使用者選擇
            examples=[
                "還有甚麼節目與這個主題相關",
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