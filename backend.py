# ==============================================================================
# 智能文档问答助手 —— 后端核心
# 技术栈：LangChain + Chroma + HuggingFace Embeddings + OpenAI LLM
# ==============================================================================

import os
import sys
from typing import Optional
from dotenv import load_dotenv
from langchain_classic.chains.retrieval_qa.base import RetrievalQA

# ------------------------------------------------------------------------------
# 1. 加载环境变量（API Key 等）
# ------------------------------------------------------------------------------
load_dotenv(override=True)  # override=True 强制覆盖系统环境变量中的同名项

# ==============================================================================
# 2. 导入 LangChain 各组件
# ==============================================================================

# ---------- 文档加载器（支持多种格式） ----------
from langchain_community.document_loaders import (
    PyPDFLoader,  # PDF
    TextLoader,  # TXT
    UnstructuredWordDocumentLoader,  # Word (.docx)
    UnstructuredMarkdownLoader,  # Markdown (.md)
)

# ---------- 文本切分 ----------
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------- 嵌入模型 ----------
from langchain_huggingface import HuggingFaceEmbeddings

# ---------- 向量库 ----------
from langchain_chroma import Chroma

# ---------- LLM ----------
from langchain_openai import ChatOpenAI

# ---------- RAG 链 ----------


# ---------- Prompt 模板（自定义，更精准） ----------
from langchain_core.prompts import PromptTemplate

# ==============================================================================
# 3. 自定义 Prompt 模板
#    目的：让 LLM 更严格地基于上下文回答，减少幻觉
# ==============================================================================
CUSTOM_PROMPT = """你是一个专业的文档问答助手。请严格根据提供的上下文内容回答用户的问题。

规则：
1. 只使用上下文中出现的信息回答，不要编造
2. 如果上下文不足以回答问题，请直接说"根据当前文档内容，我无法回答这个问题"
3. 回答时引用原文要点，保持准确、简洁
4. 如果答案涉及数字、日期、金额等，务必与原文一致

上下文：
{context}

问题：{question}

回答："""


# ==============================================================================
# 4. DocQA 核心类
# ==============================================================================
class DocQA:
    """
    智能文档问答助手

    完整 RAG 流水线：
        文档 → 加载 → 切分 → 嵌入 → Chroma向量库
                                      ↑
        提问 → 嵌入 → 检索TopK → 拼接上下文 → LLM生成 → 答案
    """

    # 支持的文件格式及其加载器映射
    SUPPORTED_FORMATS = {
        ".pdf": PyPDFLoader,
        ".txt": lambda p: TextLoader(p, encoding="utf-8"),
        ".docx": UnstructuredWordDocumentLoader,
        ".md": UnstructuredMarkdownLoader,
    }

    def __init__(
            self,
            persist_dir: str = "./chroma_db", # persist_dir: Chroma 向量库持久化目录
            embedding_model: str = "BAAI/bge-base-zh-v1.5", # embedding_model: HuggingFace 嵌入模型名或本地路径
            llm_model: str = "gpt-6-luna", # llm_model: OpenAI LLM 模型名
            chunk_size: int = 500, # chunk_size: 切分块大小（字符数）
            chunk_overlap: int = 50, # chunk_overlap: 相邻块重叠字符数
            top_k: int = 4, # top_k: 检索返回的 Top-K 相关块数量
            temperature: float = 0.0, # temperature: LLM 温度（0=确定性最强）
            device: str = "cpu", # device: 嵌入模型运行设备（"cpu" 或 "cuda"）
    ):
        """
        初始化 RAG 流水线
        Args:
        """
        self.persist_dir = persist_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

        # ----------------------------------------------------------
        # ① 嵌入模型
        # ----------------------------------------------------------
        print(f"🔧 加载嵌入模型: {embedding_model} (device={device})")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )

        # ----------------------------------------------------------
        # ② Chroma 向量库
        # ----------------------------------------------------------
        print(f"🔧 初始化向量库: {persist_dir}")
        os.makedirs(persist_dir, exist_ok=True)  # 确保目录存在
        self.vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embeddings,
        )

        # ----------------------------------------------------------
        # ③ LLM
        # ----------------------------------------------------------
        print(f"🔧 加载 LLM: {llm_model}")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "未检测到 OPENAI_API_KEY！请在 .env 文件中配置。"
            )
        base_url = os.getenv("OPENAI_BASE_URL")  # 可选，支持第三方 API
        llm_kwargs = {
            "model": llm_model,
            "temperature": temperature,
            "api_key": api_key,
        }
        if base_url:
            llm_kwargs["base_url"] = base_url
        self.llm = ChatOpenAI(**llm_kwargs)

        # ----------------------------------------------------------
        # ④ 自定义 Prompt
        # ----------------------------------------------------------
        self.prompt = PromptTemplate(
            template=CUSTOM_PROMPT,
            input_variables=["context", "question"],
        )

        # ----------------------------------------------------------
        # ⑤ RAG 链：RetrievalQA
        # ----------------------------------------------------------
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": top_k}
        )
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": self.prompt},
            return_source_documents=True,
        )

        print("✅ DocQA 初始化完成！\n")

    # ==================================================================
    # 工具方法：选择文档加载器
    # ==================================================================
    def _get_loader(self, file_path: str):
        """根据文件扩展名选择对应的文档加载器"""
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"不支持的文件格式: {ext}\n"
                f"当前支持: {list(self.SUPPORTED_FORMATS.keys())}"
            )
        loader_class = self.SUPPORTED_FORMATS[ext]
        return loader_class(file_path)

    # ==================================================================
    # Step 1+2：文档加载 + 切分
    # ==================================================================
    def _load_and_split(self, file_path: str):
        """
        加载文档并切分为文本块

        Returns:
            list[Document]: 切分后的文档块列表
        """
        # 加载
        loader = self._get_loader(file_path)
        docs = loader.load()
        print(f"📄 加载文档: {file_path} → {len(docs)} 个 Document")

        # 切分
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", "：", " ", ""],
            length_function=len,
        )
        chunks = splitter.split_documents(docs)
        print(f"✂️  切分为 {len(chunks)} 个文本块 "
              f"(chunk_size={self.chunk_size}, overlap={self.chunk_overlap})")

        # 打印每个块的大小（调试用）
        for i, chunk in enumerate(chunks[:3]):  # 只看前3个
            content_len = len(chunk.page_content)
            print(f"   块#{i}: {content_len} 字符")
        if len(chunks) > 3:
            print(f"   ... 共 {len(chunks)} 块")

        return chunks

    # ==================================================================
    # Step 3：嵌入 + 存入 Chroma
    # ==================================================================
    def _embed_and_store(self, chunks, source_file: str):
        """
        为文档块添加来源元数据，然后嵌入并存入 Chroma
        """
        # 为每个 chunk 添加来源文件元数据（方便溯源）
        for chunk in chunks:
            chunk.metadata["source_file"] = os.path.basename(source_file)

        # Chroma.add_documents 内部会自动：
        #   1. 调 self.embeddings 算出每个 chunk 的向量
        #   2. 向量 + 文本 + 元数据 一起存入
        self.vectorstore.add_documents(chunks)
        print(f"💾 已存入 Chroma 向量库（{len(chunks)} 条）")

    # ==================================================================
    # 对外方法：上传并索引文档
    # ==================================================================
    def upload(self, file_path: str) -> dict:
        """
        完整流程：加载 → 切分 → 嵌入 → 存储

        Args:
            file_path: 本地文件路径

        Returns:
            dict: 索引结果 {"total_pages": N, "total_chunks": M, "source_file": "..."}
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        print(f"\n{'=' * 50}")
        print(f"📦 开始处理: {os.path.basename(file_path)}")
        print(f"{'=' * 50}")

        # Step 1+2
        chunks = self._load_and_split(file_path)

        # Step 3
        self._embed_and_store(chunks, file_path)

        result = {
            "total_pages": len(set(
                c.metadata.get("page", 0) for c in chunks
            )) if hasattr(chunks[0], "metadata") else len(chunks),
            "total_chunks": len(chunks),
            "source_file": os.path.basename(file_path),
        }

        print(f"\n✅ 索引完成！共 {result['total_chunks']} 个文本块已入库\n")
        return result

    # ==================================================================
    # 对外方法：查询向量库中的文档数量
    # ==================================================================
    def count_documents(self) -> int:
        """返回向量库中的文档（块）总数"""
        try:
            data = self.vectorstore.get()
            return len(data.get("ids", []))
        except Exception:
            return 0

    # ==================================================================
    # 对外方法：清空向量库
    # ==================================================================
    def clear(self) -> int:
        """
        清空向量库中所有数据

        Returns:
            int: 被删除的文档数量
        """
        try:
            data = self.vectorstore.get()
            ids = data.get("ids", [])
            if ids:
                self.vectorstore.delete(ids=ids)
            print(f"🗑️  已清空向量库（删除 {len(ids)} 条）")
            return len(ids)
        except Exception as e:
            print(f"⚠️ 清空失败: {e}")
            return 0

    # ==================================================================
    # 对外方法：问答（同步）
    # ==================================================================
    def ask(self, question: str) -> dict:
        """
        基于已索引文档进行问答

        Args:
            question: 用户问题

        Returns:
            dict: {
                "answer": "LLM 生成的答案",
                "sources": [{"content": "...", "page": 1, "source_file": "..."}],
                "retrieval_count": 检索到的文档数量,
            }
        """
        # 先检查向量库是否为空
        if self.count_documents() == 0:
            return {
                "answer": "⚠️ 知识库为空！请先上传并索引文档。",
                "sources": [],
                "retrieval_count": 0,
            }

        # 调用 RAG 链
        result = self.qa_chain.invoke({"query": question})

        # 提取来源信息
        sources = []
        for doc in result.get("source_documents", []):
            sources.append({
                "content": doc.page_content,
                "page": doc.metadata.get("page", "未知"),
                "source_file": doc.metadata.get("source_file", "未知"),
            })

        return {
            "answer": result["result"],
            "sources": sources,
            "retrieval_count": len(sources),
        }

    # ==================================================================
    # 对外方法：问答（流式输出）
    # ==================================================================
    def ask_stream(self, question: str):
        """
        流式问答，逐字返回。用于前端打字机效果。

        Yields:
            str: 每次 LLM 返回的增量文本
        """
        if self.count_documents() == 0:
            yield "⚠️ 知识库为空！请先上传并索引文档。"
            return

        # 使用 LangChain 的 stream 接口
        for event in self.qa_chain.stream({"query": question}):
            # RetrievalQA.stream 返回的是 {"answer": "...", "sources": [...]} 的字典
            # 我们只增量 yield answer 部分
            if "result" in event:
                yield event["result"]
            elif isinstance(event, dict) and "answer" in event:
                yield event["answer"]

    # ==================================================================
    # 对外方法：仅检索（不生成答案，调试用）
    # ==================================================================
    def retrieve(self, question: str) -> list:
        """
        仅执行向量检索，返回最相关的文档块（不调用 LLM）

        Returns:
            list[dict]: 检索到的文档块列表
        """
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.top_k}
        )
        docs = retriever.invoke(question)
        return [
            {
                "content": doc.page_content,
                "page": doc.metadata.get("page", "未知"),
                "source_file": doc.metadata.get("source_file", "未知"),
            }
            for doc in docs
        ]


# ==============================================================================
# 5. 命令行测试入口
# ==============================================================================
if __name__ == "__main__":
    print("=" * 50)
    print("📄 智能文档问答助手 —— 命令行测试")
    print("=" * 50)

    try:
        qa = DocQA()
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        sys.exit(1)

    # 检查是否有待索引的文件
    import glob

    test_files = glob.glob("assets/*.pdf") + glob.glob("assets/*.txt")

    if test_files:
        print(f"\n📂 发现 {len(test_files)} 个待处理文件")
        for f in test_files:
            try:
                qa.upload(f)
            except Exception as e:
                print(f"  ⚠️ 处理 {f} 失败: {e}")
    else:
        print("\n💡 提示：请把测试文件放到 assets/ 目录下，或直接在交互模式上传")

    # 交互式问答循环
    print("\n💬 进入问答模式（输入 q 退出，输入 r 清空知识库）")
    while True:
        try:
            q = input("\n请输入问题: ").strip()
            if q.lower() == "q":
                print("👋 再见！")
                break
            elif q.lower() == "r":
                qa.clear()
                continue
            elif not q:
                continue

            print("\n🧠 思考中...\n")
            res = qa.ask(q)

            print(f"💡 答案：\n{res['answer']}")

            if res["sources"]:
                print(f"\n📌 来源（共 {res['retrieval_count']} 条）：")
                for i, src in enumerate(res["sources"], 1):
                    print(f"  {i}. [{src['source_file']} 第{src['page']}页] "
                          f"{src['content'][:100]}...")
        except KeyboardInterrupt:
            print("\n👋 再见！")
            break
        except Exception as e:
            print(f"❌ 问答出错: {e}")