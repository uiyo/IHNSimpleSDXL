from people import name_list
import os
import gradio as gr
from pathlib import Path
# 定义根目录
ROOT_DIR = "outputs"  # 替换为你的根目录路径

# 获取所有用户名（文件夹名称）
def get_usernames():
    # return sorted([d for d in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, d))])
    return list(name_list.keys())

# 获取某个用户名下的所有日期文件夹
def get_dates(username_chi):
    username = name_list[username_chi]
    user_dir = os.path.join(ROOT_DIR, username)
    if not os.path.exists(user_dir):
        os.mkdir(user_dir)
    return sorted([d for d in os.listdir(user_dir) if os.path.isdir(os.path.join(user_dir, d))], reverse=True)

# 获取某个日期文件夹下的 log.html 文件内容
def get_log_html(username_chi, date):
    username = name_list[username_chi]
    log_path = os.path.join(ROOT_DIR, username, date, "log.html")
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            
            html = f.read()
            # html = html.replace("onerror=\"this.closest('.image-container').style.display='none';\"", "")
            
            path = os.path.join(ROOT_DIR, username, date)
            modified_html = html.replace("<img src='", f"<img src='/file={path}/")

            return modified_html
    else:
        return "log.html 文件不存在！"

# Gradio 界面

# 初始化用户名选择
usernames = get_usernames()

# 更新日期的回调函数
def update_dates(username):
    dates = get_dates(username)
    return gr.Dropdown(choices=dates,label="选择日期")

# 更新 HTML 的回调函数
def update_html(username, date):
    return get_log_html(username, date)

# 创建 Gradio 界面
with gr.Blocks() as demo:
    gr.Markdown("## Fooocus历史图片展示系统（仅限美术组）")            
    with gr.Row():
        with gr.Column(scale=1):
            # 用户选择
            username_dropdown = gr.Dropdown(choices=usernames, label="选择用户名")
        with gr.Column(scale=4):
            # 日期选择
            date_dropdown = gr.Dropdown(label="选择日期")

    with gr.Row():    
        # HTML 展示
        html_output = gr.HTML(label="log.html 内容")
    username_dropdown.change(update_dates, inputs=username_dropdown, outputs=date_dropdown)
    date_dropdown.change(update_html, inputs=[username_dropdown, date_dropdown], outputs=html_output)

demo.launch(server_name="0.0.0.0", server_port=8188, allowed_paths=[Path.cwd().absolute()])

# if debug, flux environment
# demo.launch(allowed_paths=[Path.cwd().absolute()])