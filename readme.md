# Getting Started

This discord bot is not public. If you want to use it you will have to first set up your own bot at
https://discord.com/developers/applications/
and add it to a server.

In the ids.py file replace:

- channel_id: the channel_id of the channel you want the bot to create threads in
- channel_category_id: the id of the channel category you want the bot to read (ie. where students will put their notes and lecturers create the lecture topics)
- bot_token:  your bots token (provided on the above website)

In a later build this should be replaced with a .env file

## Install the required modules
```
pip install requirements.txt
```
## You can now run the bot, it should automatically download all required models and files on the first startup.
```
py bot.py
```