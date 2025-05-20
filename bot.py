from discord.ext import commands
import discord

import PyPDF2
import io

import os
import nltk

from ids import BOT_TOKEN, CHANNEL_ID
from load_model import load_model
from settings import *
from utils import *

#give bot all the permissions, probably questionable
bot = commands.Bot(command_prefix="!", intents=discord.Intents.all())

@bot.event
async def on_ready():
    print("Startup Complete")
    channel = bot.get_channel(CHANNEL_ID)
    await channel.send("Hello! I am here to assist")

    # Download NLTK resources if not already present
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt')
        nltk.download('punkt_tab')
        nltk.download('stopwords')


@bot.command()
async def info(ctx):
    await ctx.send("Hey, I am here to help students connect through their common questions and topics of interest. I will automatically record "
    "your notes and notify you if I found someone with similar questions. \nIf you need help with commands, use !help")


@bot.command(brief = "set up the topics for a lecture",
             description = "Provide a PDF file after this command and the bot will set up a list of potential topics that appear in the lecture")
async def setup(ctx):
    if len(ctx.message.attachments) == 0:
        await ctx.send("Please attach a PDF file with your command.")
        return
    
    attachment = ctx.message.attachments[0]
    
    # Check if the attachment is a PDF
    if not attachment.filename.endswith('.pdf'):
        await ctx.send("The attached file is not a PDF. Please upload a PDF file.")
        return
    
    try:
        # Show processing message
        process_msg = await ctx.send(f"Processing {attachment.filename} for topic analysis...")

        # Download the PDF file
        pdf_bytes = await attachment.read()
        pdf_file = io.BytesIO(pdf_bytes)
        
        # Read PDF content
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        num_pages = len(pdf_reader.pages)
        
        # Extract text from all pages
        all_text = ""
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            all_text += text + "\n"
        
        # Update processing message
        await process_msg.edit(content=f"Extracted text from {num_pages} pages. Analyzing topics...")

        # Analyze text for topics
        topics = extract_topics(all_text)
        
        # Create output directory if it doesn't exist
        os.makedirs('topic_results', exist_ok=True)
        
        # Generate filename based on original PDF name
        base_filename = os.path.splitext(attachment.filename)[0]
        output_filename = f"topic_results/{base_filename}_topics.txt"
        
        # Save topics to file
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(f"Topics extracted from {attachment.filename}:\n\n")
            for i, topic in enumerate(topics, 1):
                f.write(f"Topic {i}: {topic}\n")
        
        # Format and send the topics
        topic_message = f"**Key Topics in {attachment.filename}:**\n"
        for i, topic in enumerate(topics, 1):
            topic_message += f"{i}. {topic}\n"
        
        await ctx.send(topic_message)
        print(f"\nTopics saved to `{output_filename}`")
        
            
    except Exception as e:
        print(f"An error occurred while processing the PDF: {str(e)}")
        await ctx.send("An error occurred while processing the PDF")

@bot.event
async def on_message(message: discord.Message):

    if message.author == bot.user or message.channel.id != CHANNEL_ID:
        return

    await bot.process_commands(message)

    print(message.content)
    #TODO in here we will read the message and call the backend stuff

#before we do anything, we load the model
model = load_model()

bot.run(BOT_TOKEN)