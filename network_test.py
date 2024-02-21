import gradio as gr
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

def predict(text, request: gr.Request):
    try:
        headers = request.headers
        host = request.client.host
        user_agent = request.headers["user-agent"]
        return {
            "ip": host,
            "user_agent": user_agent,
            "headers": headers,
        }
    except Exception as e:
        logging.error("An error occurred: %s", e, exc_info=True)
        raise e  # Re-raise the exception to handle it as usual (or you can return an error message)


gr.Interface(predict, "text", "json").queue().launch(server_name='0.0.0.0', server_port=6666)
