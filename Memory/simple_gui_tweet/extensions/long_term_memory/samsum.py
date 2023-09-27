from transformers import pipeline
def samsum(conversation):
    # summarizer = pipeline("summarization", model="knkarthick/bart-large-xsum-samsum")
    # summarizer = pipeline("summarization", model="lidiya/bart-large-xsum-samsum")
    summarizer = pipeline("summarization", model="philschmid/distilbart-cnn-12-6-samsum")


    return summarizer(conversation)
