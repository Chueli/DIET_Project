# Getting Started

This discord bot is not public. If you want to use it you will have to first set up your own bot at
https://discord.com/developers/applications/
and add it to a server.

In the ids.py file replace:

- channel_id: the channel_id of the channel you want the bot to create threads in
- channel_category_id: the id of the channel category you want the bot to read (ie. where students will put their notes and lecturers create the lecture topics)
- bot_token:  your bots token (provided on the above website)

## Install the required modules
```
pip install requirements.txt
```
## Download the model, you only need to do this once
```
py download_model.py
```
## You can now run the toy example or the bot
```
py extract_topics.py
py bot.py
```