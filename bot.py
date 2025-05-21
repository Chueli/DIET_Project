import discord.context_managers
from discord.ext import commands, tasks
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

global model, topic_strings, topic_embeddings, similarities

bot_data = {
    "topic_embeddings": None,
    "topic_strings": None,
    "note_strings": [],
    "note_embeddings": [],
    "note_topic_similarities": None,
    "model": None
}

@bot.event
async def on_ready():
    #before we do anything, we load the model
    bot_data["model"] = load_model()

    # Download NLTK resources if not already present
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt')
        nltk.download('punkt_tab')
        nltk.download('stopwords')

    print("Startup Complete")
    channel = bot.get_channel(CHANNEL_ID)
    await channel.send("Hello! I am here to assist")

@bot.command()
async def info(ctx):
    await ctx.send("Hey, I am here to help students connect through their common questions and topics of interest. I will automatically record "
    "your notes and notify you if I found someone with similar questions. \nIf you need help with commands, use !help")


@bot.command(brief = "Automatically set up the topics for a lecture",
             description = "Provide a PDF or txt after this command and the bot will set up a list of potential topics that appear in the lecture")
async def extract(ctx):
    process_msg = await ctx.send(f"Processing topic analysis...")
    
    try:
        all_text = await extract_text(ctx.message)
        print(all_text)
        # Update processing message
        await process_msg.edit(content=f"Extracted text. Analyzing topics...")

        # Analyze text for topics
        topics = extract_topics(all_text)

        # Generate embeddings for the topics using sentence-transformers
        try:
            # Create embeddings
            await process_msg.edit(content=f"Generating embeddings for {len(topics)} topics...")
            bot_data["topic_embeddings"] = bot_data["model"].encode(topics, normalize_embeddings=True)
            bot_data["topic_strings"] = topics
                   
            # Update message with embedding info
            await ctx.send(f"\nGenerated {len(topics)} embeddings of dimension {bot_data['topic_embeddings'].shape[1]}")
        except Exception as e:
            await ctx.send(f"Error generating embeddings: {str(e)}")
        
        # Format and send the topics
        topic_message = f"**Key Topics:**\n"
        for i, topic in enumerate(topics, 1):
            topic_message += f"{i}. {topic}\n"
        
        await ctx.send(topic_message)
            
    except Exception as e:
        print(f"An error occurred while processing the PDF: {str(e)}")
        await ctx.send("An error occurred while processing the PDF")

@bot.command(brief = "Manually set up the topics for a lecture",
             description = "Provide a List of topics after this command, use a newline after each topic")
async def topics(ctx):
    text = ctx.message.content

    text = await extract_text(ctx.message)

    if text == "":
        await ctx.send("Please provide the topics for this lecture.")
        return

    topics = [line.strip() for line in text.split('\n') if line.strip()]

    await ctx.send(content=f"Generating embeddings for {len(topics)} topics...")
    bot_data["topic_embeddings"] = bot_data["model"].encode(topics, normalize_embeddings=True)
    bot_data["topic_strings"] = topics
                   
    # Update message with embedding info
    await ctx.send(f"\nGenerated {len(topics)} embeddings of dimension {bot_data['topic_embeddings'].shape[1]}")

@bot.command()
async def dbg(ctx):
    print(bot_data["topic_embeddings"])
    print(bot_data["topic_strings"])
    print(bot_data["note_embeddings"])
    print(bot_data["note_strings"])

@bot.event
async def on_message(message: discord.Message):

    if message.author == bot.user or message.channel.id != CHANNEL_ID:
        return

    await bot.process_commands(message)

    if message.content.startswith("!"):
        return

    text = await extract_text(message)

    bot_data["note_strings"].append((message.author.id, text))
    note_embeds = bot_data["model"].encode(text, normalize_embeddings=True)
    bot_data["note_embeddings"].append((message.author.id, note_embeds))
    
bot.run(BOT_TOKEN)