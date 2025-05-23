from datetime import datetime, timezone
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
    "model": None,
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

    print("Startup Complete")

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
    
    # channel for thread creation
    channel = bot.get_channel(CHANNEL_ID)
    
    # format and send results
    result_msg = "**Student Matches Based on Similar Topics:**\n\n"
    for author1_id, author2_id, sim, common_topics in similarities[:5]:  # Show top 5 matches
        author1 = await bot.fetch_user(author1_id)
        author2 = await bot.fetch_user(author2_id)
        
        # Add the pair of authors and their similarity score
        thread_msg = f"**{author1.name} & {author2.name}** (Similarity: {sim:.2f})\n"
        
        # Add their common topics
        if common_topics:
            thread_msg += "Common interests:\n"
            for topic, score1, score2 in common_topics[:3]:  # Show top 3 common topics
                avg_score = (score1 + score2) / 2
                thread_msg += f"- {topic} (Interest level: {avg_score:.2f})\n"
        else:
            thread_msg += "No exactly matching topics, but similar interests in:\n"
            # Show top topics for each author
            author1_top = set(topic for topics in author_topics[author1_id] for topic, _ in topics[:3])
            author2_top = set(topic for topics in author_topics[author2_id] for topic, _ in topics[:3])
            thread_msg += f"- {author1.name}: {', '.join(list(author1_top)[:3])}\n"
            thread_msg += f"- {author2.name}: {', '.join(list(author2_top)[:3])}\n"
        
        result_msg += thread_msg + "\n"

        at = f"<@{author1_id}> <@{author2_id}>\n"
        greeting = (f"Hey {at}! You've been grouped together in this thread because you've shown similar interests "
           "in your notes and questions. Introduce yourselves to one another if you're meeting for the first time! "
           "I encourage you to share your notes and questions with one another to start this conversation, and you "
           "can always @ the TA if you need help! Here are your common topics:\n")
        thread_msg = greeting + thread_msg

        await channel.create_thread(
            name=common_topics[0][0],
            content=thread_msg
        )
   
    await ctx.send(result_msg)


# Entirely generated by copilot to compute similarity scores for a specified group size
# can be called with `!matchgroup n` for n >= 3
@bot.command()
async def matchgroup(ctx, group_size: int = 3):
    '''
    Match students in groups of specified size based on their notes and common interests
    
    Args:
        group_size: Number of students per group (default 3)
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
    
    # Find groups of similar authors
    authors = list(author_topics.keys())
    from itertools import combinations
    
    if len(authors) < group_size:
        await ctx.send(f"Not enough users to form groups of size {group_size}. Only have {len(authors)} users.")
        return
    
    # Get all possible groups of the specified size
    possible_groups = list(combinations(authors, group_size))
    group_similarities = []
    
    for group in possible_groups:
        group_topics = [author_topics[author_id] for author_id in group]
        sim, common_topics = compute_group_similarity(group_topics)
        group_similarities.append((group, sim, common_topics))
    
    # Sort by similarity score
    group_similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Format and send results
    result_msg = f"**Student Groups (Size: {group_size}) Based on Similar Topics:**\n\n"
    for group, sim, common_topics in group_similarities[:3]:  # Show top 3 groups
        # Get usernames
        usernames = []
        for author_id in group:
            user = await bot.fetch_user(author_id)
            usernames.append(user.name)
        
        result_msg += f"**Group: {', '.join(usernames)}** (Similarity: {sim:.2f})\n"
        if common_topics:
            result_msg += "Common interests:\n"
            for topic, score in common_topics[:3]:  # Show top 3 common topics
                result_msg += f"- {topic} (Group interest: {score:.2f})\n"
        else:
            result_msg += "No exactly matching topics, but group members are interested in:\n"
            for author_id in group:
                user = await bot.fetch_user(author_id)
                author_top = set(topic for topics in author_topics[author_id] for topic, _ in topics[:3])
                result_msg += f"- {user.name}: {', '.join(list(author_top)[:3])}\n"
        
        result_msg += "\n"
    
    await ctx.send(result_msg)

@bot.command()
async def dbg(ctx):
    print(bot_data["topic_embeddings"])
    print(bot_data["topic_strings"])
    print(bot_data["note_embeddings"])
    print(bot_data["note_strings"])

@bot.command()
async def purgeforum(ctx):
    forum_channel = bot.get_channel(CHANNEL_ID)
    for thread in forum_channel.threads:
        await thread.delete()

@bot.command()
async def purgechannel(ctx, timestamp: str):
    try:
        target_time = datetime.fromisoformat(timestamp).replace(tzinfo=timezone.utc)

        def is_after(message):
            return message.created_at > target_time

        await ctx.channel.purge(limit=1000, check=is_after)
    except ValueError:
        await ctx.send("❌ Invalid timestamp format. Use ISO format: `YYYY-MM-DDTHH:MM:SS`")

@bot.command()
async def resetnotes(ctx):
    bot_data["note_embeddings"] = []
    bot_data["note_strings"] = []

@bot.command()
async def addmock(ctx):
    add_notes()

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

def add_notes():
    # Andy 256144780977766400
    # Clarasecondary  384138646560702475
    # Davidos 163279380565458944
    mock_user_ids = [256144780977766400, 384138646560702475, 163279380565458944]
    mock_user_notes = [
        [ 
        "i think for me to be motivated i often need to have a lot of structure and less self determination",
        "I'm struggling with cognitive load theory and how it relates to UI design",
        "Really interested in how social learning works in online environments, especially in classroom or seminar settings",
        "my experience of learning has had very little to do with things like what i am expecting to gain, and instead i just see what i end up learning"
        ],
        [
        "The idea that you can’t just tell someone something and expect them to know it — you have to build the right experience around it. That's hard.",
        "We’re supposed to build a tool that adapts to learners and supports group work? How are we even modeling individual vs. group behavior?",
        "I use duolingo very frequently and it has definitely helped me build a habit but i think i do it mostly for the points and not really for the learning",
        "I have noticed that many learners' motivation comes from whether or not they are interested in the topic and have an innate sense of enthusiasm for it, its very rare that a textbook created enthusiasm in anyone around me."
        ],
        [
        "The class covered how kids learn by watching others and trying it themselves. Makes sense, but how do we build tech that supports that?",
        "so is autonomy central for a student to be motivated to engage in the hard task of learning?",
        "Still not sure how giving students choices actually helps them learn better. Isn’t more freedom sometimes just overwhelming?",
        "i think for me to be motivated i often need to have a lot of structure and less self determination"
        ]
    ]
    count = 0
    for i in range(len(mock_user_ids)):
        # Convert notes to embeddings and add to bot_data
        for note in mock_user_notes[i]:
            bot_data["note_strings"].append((mock_user_ids[i], note))
            note_embeds = torch.from_numpy(bot_data["model"].encode(note, normalize_embeddings=True))
            bot_data["note_embeddings"].append((mock_user_ids[i], note_embeds))
            count += 1
           
    print(f"Added {count} mock notes for testing")

    
bot.run(BOT_TOKEN)