# Import necessary libraries
import gradio as gr
import openai
import time
import dotenv
import os

# Load environment variables from .env file
dotenv.load_dotenv()

# Set OpenAI organization ID and API key from environment variables
openai.organization = os.environ["OPENAI_ORG"]
openai.api_key = os.environ["OPENAI_KEY"]

# Define the initial system message
system_message = {"role": "system", "content": "You are a helpful assistant."}

# Initialize messages_history with the system message
messages_history = [system_message]

# Add an initial assistant message
initial_assistant_message = {
    "role": "assistant",
    "content": "Hello! I am a helpful assistant. How can I help you today?",
}
messages_history.append(initial_assistant_message)


# Function to send user message and conversation history to OpenAI API and get a response
def ask_gpt(message):
    global messages_history
    messages_history.append({"role": "user", "content": message})

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=messages_history
    )

    bot_message = response["choices"][0]["message"]["content"]
    messages_history.append({"role": "assistant", "content": bot_message})

    return bot_message


# Function to reset chat history
def reset_history():
    global messages_history
    messages_history = [system_message, initial_assistant_message]


# Function to generate the message pairs for the chat component
def generate_chat_pairs():
    global messages_history
    chat_history_in_tuples = list()
    for message_pair in grouped(messages_history[2:], 2):
        m_user, m_bot = message_pair
        chat_history_in_tuples.append((m_user["content"], m_bot["content"]))
    return chat_history_in_tuples


# Function to group elements of an iterable
def grouped(iterable, n):
    "s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ..."
    return zip(*[iter(iterable)] * n)


# Create Gradio Blocks interface
with gr.Blocks() as interface:
    # Create interface elements (Chatbot, Textbox, Clear Button)
    chatbot = gr.Chatbot()
    textbox = gr.Textbox()
    clear_button = gr.Button("Clear")

    # Function to handle user message input and bot response
    def process_message(user_message):
        # bot_message = ask_gpt(user_message)
        # return [(user_message, bot_message)]  # Return as a list of tuples
        ask_gpt(user_message)
        return gr.update(value=""), generate_chat_pairs()

    # Set up event listeners for Textbox submit and Clear Button click events
    # textbox.submit(lambda x: gr.update(value=''), [],[textbox])
    # https://discuss.huggingface.co/t/unable-to-clear-input-after-submit/33543/4
    textbox_submit = textbox.submit(
        process_message, inputs=[textbox], outputs=[textbox, chatbot], queue=False
    )  # .then(lambda x: gr.update(value=''), inputs=None, outputs=[textbox])
    clear_button_click = clear_button.click(
        reset_history, inputs=[], outputs=[chatbot], queue=False
    )

# Launch the Gradio interface
interface.launch(server_name="0.0.0.0", server_port=7860)
