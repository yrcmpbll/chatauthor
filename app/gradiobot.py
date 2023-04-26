import gradio as gr
import openai
import random
import time
import dotenv
import os

# Load secrets into environment variables
dotenv.load_dotenv()

# Set org ID and API key
openai.organization = os.environ['OPENAI_ORG'] # "<YOUR_OPENAI_ORG_ID>"
openai.api_key = os.environ['OPENAI_KEY'] # "<YOUR_OPENAI_API_KEY>"

system_message = {"role": "system", "content": "You are a helpful assistant."}

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    state = gr.State([])

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history, messages_history):
        user_message = history[-1][0]
        bot_message, messages_history = ask_gpt(user_message, messages_history)
        messages_history += [{"role": "assistant", "content": bot_message}]
        history[-1][1] = bot_message
        time.sleep(1)
        return history, messages_history

    def ask_gpt(message, messages_history):
        messages_history += [{"role": "user", "content": message}]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages_history
        )
        return response['choices'][0]['message']['content'], messages_history

    def init_history(messages_history):
        messages_history = []
        messages_history += [system_message]
        return messages_history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, [chatbot, state], [chatbot, state]
    )

    clear.click(lambda: None, None, chatbot, queue=False).success(init_history, [state], [state])

demo.launch()