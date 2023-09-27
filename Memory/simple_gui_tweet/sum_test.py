from extensions.long_term_memory.core.memory_database import LtmDatabase
import pathlib
memory_database = LtmDatabase(
    directory=pathlib.Path("./extensions/long_term_memory/user_data/bot_memories/"),
    force_use_legacy_db=True,
    num_memories_to_fetch=4
)
memory_database._load_db()
sums = [
    "Member2 went on a trip to Japan last summer. He liked the food, temples and shopping in Harajuku. He didn't go to any other countries while he was there.",
    "It's Member1's birthday next week. He's having a dinner party at a fancy restaurant with about 10 people. He'll have a chocolate cake with raspberry filling.",
    "Member4 has been to Japan, but he went to Kyoto instead. He visited shrines and temples and saw some geishas. He had some amazing sushi and tried some matcha sweets. He recommends the bamboo forest in Kyoto.",
    "Member2 will bring a bottle of wine for the celebration. Member3 will help with decorations."
    ]
# for s in sums:
#     print('zmy',[{'summary_text':s}])
#     memory_database.add([{'summary_text':s}])

question = "What museum did Member2 visit in Japan?"
fetched_memories = memory_database.query(
        question
    )
print('fetched_memories',fetched_memories)