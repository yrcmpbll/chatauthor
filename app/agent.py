import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.agents import Tool
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.agents import initialize_agent


class Agent:
    def __init__(self, faiss, author_names) -> None:
        self.faiss = faiss
        self.author_names = author_names

        self.llm = ChatOpenAI(
            openai_api_key=os.environ["OPENAI_KEY"],
            temperature=0.15,
            model_name="gpt-3.5-turbo",
        )

        self.retriever = RetrievalQA.from_chain_type(
            llm=self.llm, chain_type="stuff", retriever=self.faiss.as_retriever()
        )

        self.tools = list()
        self.__load_tools()

        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",  # important to align with agent prompt (below)
            k=5,
            return_messages=True,
        )

        self.conversational_engine = initialize_agent(
            agent="chat-conversational-react-description",
            tools=self.tools,
            llm=self.llm,
            verbose=True,
            max_iterations=5,
            early_stopping_method="generate",
            memory=self.memory,
        )

        sys_msg = """You are a helpful chatbot that answers the user's questions."""

        system_prompt = self.conversational_engine.agent.create_prompt(
            system_message=sys_msg, tools=self.tools
        )
        self.conversational_engine.agent.llm_chain.prompt = system_prompt

    def prompt(self, message):
        try:
            engine_output = self.conversational_engine(message)
            answer = engine_output["output"]
        except e:
            answer = "Problems with my chain."

        return answer

    def __load_tools(self):
        self.__load_author_db_tool()

    def __load_author_db_tool(self):
        tool_desc = f"""Use this tool to answer user questions refering to "{self.author_names[0]} {self.author_names[1]}" or just "{self.author_names[1]}". 
                        If the user mentions "{self.author_names[0]} {self.author_names[1]}", use this tool to get the answer. 
                        Use this tool also for follow up questions from the user."""

        author_db_tool = Tool(
            func=self.retriever.run,
            description=tool_desc,
            name=f"{self.author_names[0]} {self.author_names[1]} DB",
        )

        self.tools.append(author_db_tool)
