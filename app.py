import streamlit as st
import os
from backend import DocQA

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="智能文档问答助手",
    page_icon="📄",
    layout="wide"
)

# ============================================================
# 会话状态：在整个页面生命周期中保持 DocQA 实例
# 避免每次交互都重新加载模型（很慢！）
# ============================================================
if "qa" not in st.session_state:
    with st.spinner("正在加载嵌入模型（首次约 1-2 分钟）..."):
        st.session_state.qa = DocQA()
qa = st.session_state.qa

# ============================================================
# 标题
# ============================================================
st.title("📄 智能文档问答助手")
st.caption("基于 RAG 的本地文档问答系统 —— 上传 PDF，提问即可")

# ============================================================
# 侧边栏：文档管理
# ============================================================
with st.sidebar:
    st.header("📚 文档管理")

    # 上传 PDF
    uploaded_file = st.file_uploader(
        "选择 PDF 文件",
        type=["pdf"],
        help="支持 PDF 格式，上传后点击索引"
    )

    if uploaded_file:
        # 保存到临时文件
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        # 索引按钮
        if st.button("🔍 索引文档", type="primary"):
            with st.spinner("正在加载 → 切分 → 嵌入 → 存储..."):
                try:
                    msg = qa.upload(temp_path)
                    st.success(msg)
                except Exception as e:
                    st.error(f"索引失败：{e}")
                finally:
                    # 清理临时文件
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

    # 清空按钮
    st.divider()
    if st.button("🗑️  清空知识库"):
        qa.clear()
        st.warning("已清空所有文档")

# ============================================================
# 主区域：问答
# ============================================================
question = st.text_input(
    "💬 基于已索引文档提问：",
    placeholder="例如：这份合同的违约金是多少？"
)

if question:
    with st.spinner("🧠 检索 + 生成中..."):
        try:
            result = qa.ask(question)

            # 显示答案
            st.markdown(f"### 💡 答案\n\n{result['answer']}")

            # 显示来源（可折叠）
            if result["sources"]:
                with st.expander("📌 答案来源（可点击展开）"):
                    for i, src in enumerate(result["sources"], 1):
                        st.markdown(f"**来源 {i}** — 第 {src['page']} 页")
                        st.caption(src["content"])

        except Exception as e:
            st.error(f"问答失败：{e}")

# ============================================================
# 底部提示
# ============================================================
st.divider()
st.caption("💡 提示：首次加载模型较慢，请耐心等待。答案基于你上传的文档生成，若文档中没有相关内容，回答可能不准确。")