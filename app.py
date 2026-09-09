import os
import gradio as gr
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import ZhipuAIEmbeddings
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

chat = ChatZhipuAI(model="glm-4-flash", temperature=0.7)
embedding = ZhipuAIEmbeddings(model="embedding-2")

ROLES = {
    "🤣 幽默助手": "你是一个幽默风趣的AI助手，喜欢讲冷笑话，说话带点调皮",
    "👨‍💼 专业顾问": "你是一个专业的商业顾问，回答严谨、逻辑清晰、数据驱动",
    "✍️ 创意文案": "你是一个创意文案写手，语言优美、富有想象力、擅长比喻",
    "💻 代码助手": "你是一个编程专家，回答简洁、直接给代码示例、解释清楚"
}

class ConversationMemory:
    def __init__(self):
        self.history = []

    def add_message(self, role, content):
        self.history.append({"role": role, "content": content})

    def get_messages(self):
        messages = []
        for msg in self.history[-10:]:  # 只保留最近10轮，防止token超限
            if msg["role"] == "system":
                messages.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        return messages

    def clear(self):
        self.history = []

    def export(self):

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        content = f"# 对话记录 - {timestamp}\n\n"
        for msg in self.history:
            if msg["role"] == "user":
                content += f"用户: {msg['content']}\n\n"
            elif msg["role"] == "assistant":
                content += f"AI: {msg['content']}\n\n"
        return content


memory = ConversationMemory()
vector_db = None  # 全局向量库，上传文档后赋值

def upload_file(file_obj):
    global vector_db
    if file_obj is None:
        return "未选择文件，请上传txt文档"
    with open(file_obj.name, "r", encoding="utf-8") as f:
        raw_text = f.read()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    chunks = splitter.split_text(raw_text)
    vector_db = FAISS.from_texts(chunks, embedding)
    return f"✅ 文档加载成功！共切分 {len(chunks)} 个文本块，可以开始提问知识库内容"

def ai_chat(user_input, temperature, role_key, history_visible):
    global vector_db
    if not user_input.strip():
        return "请输入有效内容。", history_visible, ""

    system_prompt = ROLES[role_key]

    rag_context = ""
    if vector_db is not None:
        # 相似度检索，拿top3相关片段
        docs = vector_db.similarity_search(user_input, k=3)
        rag_context = "\n\n".join([d.page_content for d in docs])
        # 把检索到的知识库内容加到system提示词
        system_prompt += f"""
【参考知识库内容，回答必须基于下面材料，如果材料没有答案，直接说明知识库没有相关信息，不要编造】
{rag_context}
"""

    # 构建消息列表
    messages = [
        SystemMessage(content=system_prompt)
    ]
    # 从记忆中加载历史对话
    messages.extend(memory.get_messages())
    # 添加当前用户输入
    messages.append(HumanMessage(content=user_input))

    # 更新Temperature
    chat.temperature = temperature

    try:
        response = chat.invoke(messages)
        reply = response.content

        # 保存到记忆
        memory.add_message("user", user_input)
        memory.add_message("assistant", reply)

        # 构建历史显示
        history_display = ""
        for msg in memory.history[-20:]:
            if msg["role"] == "user":
                history_display += f"🧑 你: {msg['content']}\n\n"
            elif msg["role"] == "assistant":
                history_display += f"🤖 AI: {msg['content']}\n\n"

        return reply, history_display, ""

    except Exception as e:
        return f"调用出错：{str(e)}", history_visible, ""

# 导出对话功能
def export_conversation():
    content = memory.export()
    if content.strip() == "# 对话记录 - \n\n":
        return "暂无对话记录可导出"
    return content

# 清空对话
def clear_conversation():
    memory.clear()
    return "✅ 对话已清空", ""

# 清空知识库
def clear_knowledge_base():
    global vector_db
    vector_db = None
    return "🗑️ 知识库已清空，不再读取文档内容"


with gr.Blocks(title="智能对话助手") as demo:
    gr.Markdown("""
    # 🤖 智能对话助手 Pro（RAG知识库版本）
    > 基于 智谱AI GLM-4-Flash + LangChain + FAISS向量库
    > 支持：私有文档问答RAG、多轮记忆、角色切换、参数调节
    """)

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("## 📚 RAG知识库（仅支持txt文件）")
            file_upload = gr.File(label="上传文档", file_types=[".txt"])
            upload_msg = gr.Textbox(label="知识库状态", interactive=False)
            clear_kb_btn = gr.Button("清空知识库")

            gr.Markdown("## 💬 对话设置")
            user_input = gr.Textbox(
                label="💬 输入你的问题",
                placeholder="输入你想问的...\n如果上传文档，AI会优先基于文档回答",
                lines=3
            )

            # 功能4：角色切换
            role_selector = gr.Dropdown(
                choices=list(ROLES.keys()),
                value="🤣 幽默助手",
                label="🎭 选择AI角色"
            )

            temp_slider = gr.Slider(
                minimum=0.0,
                maximum=1.5,
                value=0.7,
                step=0.1,
                label="🌡️ Temperature（随机性）",
                info="0.0 = 严谨保守 | 0.7 = 平衡 | 1.5 = 创意发散"
            )

            with gr.Row():
                submit_btn = gr.Button("🚀 发送", variant="primary")
                clear_btn = gr.Button("🗑️ 清空对话", variant="stop")

            # 功能3：导出按钮
            export_btn = gr.Button("📥 导出对话记录")

        # 右侧面板
        with gr.Column(scale=3):
            # 输出区域
            output = gr.Textbox(
                label="🤖 AI 回复",
                lines=6
            )

            # 对话历史显示
            history_display = gr.Textbox(
                label="📜 对话历史",
                lines=10,
                interactive=False,
                placeholder="对话记录将在这里显示..."
            )

    # 导出结果显示（隐藏）
    export_output = gr.Textbox(label=" 导出内容", lines=10, visible=False)


    file_upload.change(fn=upload_file, inputs=[file_upload], outputs=[upload_msg])
    clear_kb_btn.click(fn=clear_knowledge_base, inputs=[], outputs=[upload_msg])

    submit_btn.click(
        fn=ai_chat,
        inputs=[user_input, temp_slider, role_selector, history_display],
        outputs=[output, history_display, export_output]
    )

    clear_btn.click(
        fn=clear_conversation,
        inputs=[],
        outputs=[history_display, output]
    )

    export_btn.click(fn=export_conversation, inputs=[], outputs=[export_output])

    gr.Examples(
        examples=[
            ["讲个冷笑话"],
            ["用一句话介绍你自己"],
            ["写一段 Python 代码实现快速排序"],
            ["今天天气不错，用诗意的语言描述一下"],
            ["根据文档内容，总结要点"]
        ],
        inputs=[user_input],
        label=" 快速示例"
    )

if __name__ == "__main__":
    demo.launch(share=False, theme=gr.themes.Soft())
