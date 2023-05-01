import openai


class Translator:

    def __init__(self) -> None:
        # Define the initial system message
        system_message = {"role": "system", "content": "You are a helpful assistant."}

        # Initialize messages_history with the system message
        messages_history = [system_message]

        # Add an initial assistant message
        initial_assistant_message = {
            "role": "assistant",
            "content": "Hello! I am a helpful assistant. How can I help you today?"
        }
        messages_history.append(initial_assistant_message)

        # Define ground zero history
        self.zero_prompt = messages_history

    def __api_call(self, message):
        api_call_history = self.zero_prompt + [{"role": "user", "content": message}]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=api_call_history
        )
        return response['choices'][0]['message']['content']

    def to_german(self, message):
        return self.__api_call('Translate this to German: '+ message)
    
    def to_english(self, message):
        return self.__api_call('Translate this to English: '+ message)