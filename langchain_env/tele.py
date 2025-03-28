import os
import openai
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI  # Correct import for Chat Model
from langchain.chains import TransformChain  # New method for handling requests
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory  # Import memory
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes
from langchain_openai import ChatOpenAI


# Load environment variables from .env file
load_dotenv()

# Get API keys from the .env file
openai_api_key = os.getenv("OPENAI_API_KEY")
telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")

print(f"OpenAI API Key: {openai_api_key}")

# Initialize the ChatOpenAI model (Using GPT-4)
llm = ChatOpenAI(model_name="gpt-4", openai_api_key=openai_api_key)

# Initialize memory for conversation
memory = ConversationBufferMemory()

# Define the prompt for generating responses
prompt_template = PromptTemplate(
    input_variables=["message", "history"],
    template=(
        "You are a helpful assistant responding to Telegram messages. Here is the conversation history:\n"
        "{history}\nUser: {message}\nRespond with a helpful and engaging answer."
    )
)

# Define the transformation process (New LangChain method)
def generate_response(inputs):
    user_message = inputs["message"]
    conversation_history = memory.load_memory_variables({}).get("history", "")
    prompt = prompt_template.format(message=user_message, history=conversation_history)
    
    # Generate the response using invoke()
    response = llm.invoke(prompt)  # Removed list brackets to get the actual response
    response_text = response.content  # Correctly access the content attribute

    # Save conversation in memory
    memory.save_context({"message": user_message}, {"response": response_text})
    
    return {"response": response_text}

# Create a LangChain TransformChain with memory
response_chain = TransformChain(
    input_variables=["message"],
    output_variables=["response"],
    transform=generate_response
)

# Define the message handler function
async def handle_telegram_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    result = response_chain.invoke({"message": user_message})
    response = result["response"]
    await update.message.reply_text(response)

# Initialize the Telegram Bot application
app = ApplicationBuilder().token(telegram_bot_token).build()

# Add the message handler to the dispatcher
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_telegram_message))

# Start the bot
print("✅ Telegram AI Bot with Memory is now running...")
app.run_polling()
