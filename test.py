import gradio as gr

# 定义 JS 脚本
js = """
function someFunction() {
  return "选项3"
}
"""
with gr.Blocks() as interface:
    # 创建 Gradio 组件
    dropdown = gr.Dropdown(choices=["选项 1", "选项 2", "选项 3"], value="选项 1",_js=js)
    # 默认选项为1
    # 在 Python 脚本中接收来自 JS 脚本的值
    # dropdown.change(lambda x: gr.update(value=x), inputs=dropdown, outputs=dropdown)
    @dropdown.change
    def on_change(value):
        # 修改下拉栏的值
        dropdown.value = someFunction()

# 启动 Gradio 服务器
interface.launch(server_port=5055)