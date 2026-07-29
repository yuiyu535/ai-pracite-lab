import os
import gradio as gr
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

# 加载.env中的API Key
load_dotenv()

# 创建模型客户端（与之前完全一样）
chat = ChatZhipuAI(
    model="glm-4-flash",
    temperature=0.5,
)

# 定义对话函数
def ai_chat(user_input):
    if not user_input.strip():
        return "请输入有效内容。"
    messages = [
        SystemMessage(content="你是一个幽默风趣的AI助手"),
        HumanMessage(content=user_input)
    ]
    response = chat.invoke(messages)
    return response.content

# 构建Gradio界面
demo = gr.Interface(
    fn=ai_chat,
    inputs=gr.Textbox(label="💬 输入你的问题", placeholder="例如：讲个冷笑话", lines=2),
    outputs=gr.Textbox(label="🤖 AI回答", lines=6),
    title="🤖 智能对话助手",
    description="基于 **智谱AI GLM-4-Flash** 模型 + **LangChain** 框架构建",
    theme="soft",
    examples=[
        ["请用一句话介绍你自己"],
        ["给我讲一个关于程序员的笑话"],
        ["用三个词形容今天的天气"]
    ]
)

if __name__ == "__main__":
    demo.launch()