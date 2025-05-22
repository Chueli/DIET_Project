import discord.context_managers
from discord.ext import commands, tasks
import discord

import PyPDF2
import io

import os
import nltk
from collections import defaultdict

from ids import BOT_TOKEN, CHANNEL_ID, CHANNEL_CATEGORY_ID
from settings import *
from settings import diet_topics as topics
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

    # # Download NLTK resources if not already present
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt')
        nltk.download('punkt_tab')
        nltk.download('stopwords')

    # add mock user data
    mock_user_id = 302145648759668737  # Replace with your mock user ID
    mock_notes = [
        "i think for me to be motivated i often need to have a lot of structure and less self determination",
        "I'm struggling with cognitive load theory and how it relates to UI design",
        "Really interested in how social learning works in online environments",
        "my experience of learning has had very little to do with things like what i am expecting to gain, and instead i just see what i end up learning"
    ]
    
    # Convert notes to embeddings and add to bot_data
    for note in mock_notes:
        bot_data["note_strings"].append((mock_user_id, note))
        note_embeds = torch.from_numpy(bot_data["model"].encode(note, normalize_embeddings=True))
        bot_data["note_embeddings"].append((mock_user_id, note_embeds))
    
    print(f"Added {len(mock_notes)} mock notes for testing")

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

        # Generate embeddings for the topics using sentence-transformers
        try:
            # Create embeddings
            await process_msg.edit(content=f"Generating embeddings for {len(topics)} topics...")
            bot_data["topic_embeddings"] = torch.from_numpy(bot_data["model"].encode(topics, normalize_embeddings=True))
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
    bot_data["topic_embeddings"] = torch.from_numpy(bot_data["model"].encode(topics, normalize_embeddings=True))
    bot_data["topic_strings"] = topics
                   
    # Update message with embedding info
    await ctx.send(f"\nGenerated {len(topics)} embeddings of dimension {bot_data['topic_embeddings'].shape[1]}")

@bot.command()
async def dbg(ctx):
    print(bot_data["topic_embeddings"])
    print(bot_data["topic_strings"])
    print(bot_data["note_embeddings"])
    print(bot_data["note_strings"])

@bot.command()
async def match(ctx):
    '''
    Match students with similar topics based on their notes and show their common interests
    '''
    # ensure that we have topics and notes embedded
    if (bot_data["topic_embeddings"] is None or 
        len(bot_data["topic_embeddings"]) == 0 or 
        bot_data["topic_strings"] is None or 
        len(bot_data["topic_strings"]) == 0):
        await ctx.send("No topics have been set up yet. Use !extract or !topics first.")
        return
        
    if not bot_data["note_embeddings"]:
        await ctx.send("No notes have been recorded yet.")
        return
    
    # group notes and embeddings by author
    author_notes = defaultdict(list)
    author_embeddings = defaultdict(list)
    
    for (author_id, note), (_, embedding) in zip(bot_data["note_strings"], bot_data["note_embeddings"]):
        author_notes[author_id].append(note)
        author_embeddings[author_id].append(embedding)
    
    # compute topics for each author
    author_topics = {}
    for author_id in author_notes.keys():
        # stack all embeddings for given author
        stacked_embeddings = torch.stack(author_embeddings[author_id])
        # compute topics for all notes
        topics = compute_user_topics(stacked_embeddings, bot_data["topic_embeddings"], bot_data["topic_strings"], top_n=5)
        author_topics[author_id] = topics
    
    # compute similarities between all pairs of authors
    similarities = []
    authors = list(author_topics.keys())
    for i, author1 in enumerate(authors):
        for author2 in authors[i+1:]:  # avoid comparing an author with themselves
            sim, common_topics = compute_author_similarity(author_topics[author1], author_topics[author2])
            similarities.append((author1, author2, sim, common_topics))
    
    # sort by similarity score
    similarities.sort(key=lambda x: x[2], reverse=True)
    
    # format and send results
    result_msg = "**Student Matches Based on Similar Topics:**\n\n"
    for author1_id, author2_id, sim, common_topics in similarities[:5]:  # Show top 5 matches
        author1 = await bot.fetch_user(author1_id)
        author2 = await bot.fetch_user(author2_id)
        
        # Add the pair of authors and their similarity score
        result_msg += f"**{author1.name} & {author2.name}** (Similarity: {sim:.2f})\n"
        
        # Add their common topics
        if common_topics:
            result_msg += "Common interests:\n"
            for topic, score1, score2 in common_topics[:3]:  # Show top 3 common topics
                avg_score = (score1 + score2) / 2
                result_msg += f"- {topic} (Interest level: {avg_score:.2f})\n"
        else:
            result_msg += "No exactly matching topics, but similar interests in:\n"
            # Show top topics for each author
            author1_top = set(topic for topics in author_topics[author1_id] for topic, _ in topics[:3])
            author2_top = set(topic for topics in author_topics[author2_id] for topic, _ in topics[:3])
            result_msg += f"- {author1.name}: {', '.join(list(author1_top)[:3])}\n"
            result_msg += f"- {author2.name}: {', '.join(list(author2_top)[:3])}\n"
        
        result_msg += "\n"
    
    await ctx.send(result_msg)

@bot.event
async def on_message(message: discord.Message):

    if message.author == bot.user or message.channel.category_id != CHANNEL_CATEGORY_ID:
        return

    await bot.process_commands(message)

    if message.content.startswith("!"):
        return

    text = await extract_text(message)

    bot_data["note_strings"].append((message.author.id, text))
    note_embeds = torch.from_numpy(bot_data["model"].encode(text, normalize_embeddings=True))
    bot_data["note_embeddings"].append((message.author.id, note_embeds))
    
bot.run(BOT_TOKEN)