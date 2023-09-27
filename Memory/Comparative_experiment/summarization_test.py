from transformers import pipeline
# print('Please input conversation:')
conversation = """
Person B: Have you heard about the new restaurant that opened downtown? It's supposed to have amazing food.

Person A: Yes, I have! I actually went there last week, and I can confirm that the food is incredible. The menu is diverse, the flavors are unique, and the presentation is top-notch. It's definitely worth a visit if you're a food enthusiast.

Person B: That's awesome! I'm always on the lookout for new places to try. I'll make sure to add it to my list. By the way, have you been following the latest sports tournament? There have been some intense matches.

Person A: Oh yes, I've been keeping up with it. The level of competition has been amazing. Did you catch the match between Team A and Team B? It was a nail-biter till the very end.

Person B: Unfortunately, I missed that one. But I heard it was a thrilling match. I'll have to catch the highlights. On a different topic, have you seen the latest exhibition at the art gallery? I heard it's quite unique.

Person A: Yes, I went there last weekend. It was unlike any exhibition I've seen before. The artist's work was thought-provoking, and the way they explored different mediums was fascinating. It's definitely worth visiting if you appreciate contemporary art.
"""

summarizer1 = pipeline("summarization", model="philschmid/distilbart-cnn-12-6-samsum")
summarizer2 = pipeline("summarization", model="facebook/bart-large-cnn")
summarizer3 = pipeline("summarization", model="morenolq/bart-base-xsum")
summarizer4 = pipeline("summarization", model="facebook/bart-large-xsum")
summarizer5 = pipeline("summarization", model="Samuel-Fipps/t5-efficient-large-nl36_fine_tune_sum_V2")
summarizer6 = pipeline("summarization", model="lidiya/bart-large-xsum-samsum")
summarizer7 = pipeline("summarization", model="knkarthick/bart-large-xsum-samsum")
summarizer8 = pipeline("summarization", model="knkarthick/meeting-summary-samsum")





from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch
model_name = "google/pegasus-xsum"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
batch = tokenizer(conversation, truncation=True, padding="longest", return_tensors="pt").to(device)
translated = model.generate(**batch, max_new_tokens=2048)
tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)

print('='*60)
print('MODEL:philschmid/distilbart-cnn-12-6-samsum')
print('-'*60)
print(summarizer1(conversation))
print('='*60)

print('MODEL:facebook/bart-large-cnn')
print('-'*60)
print(summarizer2(conversation))
print('='*60)

print('MODEL:google/pegasus-xsum')
print('-'*60)
print(tgt_text)
print('='*60)

print('MODEL:facebook/bart-large-xsum')
print('-'*60)
print(summarizer4(conversation))
print('='*60)

print('MODEL:morenolq/bart-base-xsum')
print('-'*60)
print(summarizer3(conversation))
print('='*60)

print('MODEL:Samuel-Fipps/t5-efficient-large-nl36_fine_tune_sum_V2')
print('-'*60)
print(summarizer5(conversation))
print('='*60)

print('MODEL:lidiya/bart-large-xsum-samsum')
print('-'*60)
print(summarizer6(conversation))
print('='*60)

print('MODEL:knkarthick/bart-large-xsum-samsum')
print('-'*60)
print(summarizer7(conversation))
print('='*60)

print('MODEL:knkarthick/meeting-summary-samsum')
print('-'*60)
print(summarizer8(conversation))
print('='*60)