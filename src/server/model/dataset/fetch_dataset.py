import os

import discord

client = discord.Client(intents=discord.Intents.all())


@client.event
async def on_ready():
    server = discord.utils.get(client.guilds, id=803282468798201927)
    channel = discord.utils.get(server.channels, id=857811318195486740)

    emojis = list(server.emojis)

    with open("dataset.txt", "w", encoding="utf-8") as f:
        async for message in channel.history(limit=None):
            if message.author.bot:
                continue

            content = message.clean_content

            if not content.strip():
                continue

            for emoji in emojis:
                if str(emoji) in content:
                    content = content.replace(str(emoji), f":{emoji.name}:")

            f.write(f"<{message.author.name}>: {content}\n-----\n")

    await client.close()


client.run(os.environ["DISCORD_TOKEN"])
