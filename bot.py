import datetime
from discord.ext import commands, tasks
import discord
from dataclasses import dataclass
from ids import BOT_TOKEN, CHANNEL_ID
from load_model import load_model

bot = commands.Bot(command_prefix="!", intents=discord.Intents.all())

@bot.event
async def on_ready():
    print("Startup Complete")
    channel = bot.get_channel(CHANNEL_ID)
    await channel.send("Hello! I am here to assist")


@bot.command(brief = "set up the topics for a lecture",
             description = "Provide a PDF file after this command and the bot will set up a list of potential topics that appear in the lecture")
async def setup(ctx):
    await ctx.send("Not yet implemented")

    #TODO: Ideally we can add the topics we want the bot to learn here, e.g provide a file and it will extract the main topics

@bot.command()
async def info(ctx):
    await ctx.send("Hey, I am here to help students connect through their common questions and topics of interest. I will automatically record "
    "your notes and notify you if I found someone with similar questions. \nIf you need help with commands, use !help")

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