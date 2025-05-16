# install pip install transformers torch gradio
import gradio as gr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import pipeline, AutoModelForImageClassification, AutoImageProcessor
from sklearn.metrics import accuracy_score
from collections import defaultdict
import re
from PIL import Image

# Load dataset
df = pd.read_csv('sentimentdataset.csv')

# Clean the text
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower()

df['Cleaned_Text'] = df['Text'].apply(clean_text)

# Add manual labels (you should adjust these based on actual tweet contents)
df['True_Emotion'] = None
df.loc[0, 'True_Emotion'] = 'joy'
df.loc[1, 'True_Emotion'] = 'sadness'
df.loc[2, 'True_Emotion'] = 'anger'
df.loc[3, 'True_Emotion'] = 'fear'
df.loc[4, 'True_Emotion'] = 'love'

# Load emotion classification model
emotion_analyzer = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)

# Load image classification model (example: Vision Transformer)
image_model_name = "google/vit-base-patch16-224-in21k"
image_processor = AutoImageProcessor.from_pretrained(image_model_name)
image_model = AutoModelForImageClassification.from_pretrained(image_model_name)

# Emoji emotion mapping
emoji_emotions = {
    '😀': 'joy',
    '😂': 'joy',
    '😢': 'sadness',
    '😭': 'sadness',
    '😠': 'anger',
    '😡': 'anger',
    '😨': 'fear',
    '😱': 'fear',
    '😍': 'love',
    '😘': 'love',
    '😐': 'neutral',
    '😴': 'tiredness',
    '😎': 'confidence',
    '🤔': 'curiosity',
    '😇': 'peace',
    '😤': 'frustration',
    '🤯': 'shock',
    '😳': 'embarrassment',
}

# Define the functions for Gradio UI

# Emotion classification for text input
def classify_text_emotion(text):
    cleaned_text = clean_text(text)
    try:
        result = emotion_analyzer(cleaned_text)[0][0]
        emotion = result['label']
        if emotion == 'anger':
            return f"Wow, you sound MAD! Calm down. Predicted Emotion: {emotion}. What’s got you so mad, huh?"
        else:
            return f"Predicted Emotion: {emotion}"
    except Exception as e:
        return f"Error: {str(e)}"

# Emotion classification for emoji input
def classify_emoji(emoji_input):
    emotion = emoji_emotions.get(emoji_input, 'Unknown or not in database')
    if emotion == 'anger':
        return f"Ugh, I can feel the rage! 😡 Emoji: {emoji_input} => Emotion: {emotion}. Are you yelling at your screen right now?"
    else:
        return f"Emoji: {emoji_input} => Emotion: {emotion}"

# Image classification function
def classify_image(image):
    # Preprocess image
    inputs = image_processor(images=image, return_tensors="pt")
    
    # Run image through the model
    outputs = image_model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    
    # Get the label for the predicted class
    class_names = image_model.config.id2label
    predicted_class = class_names[predicted_class_idx]
    
    return f"Predicted Image Label: {predicted_class}"

# Gradio interface

# Text-based input interface
text_input = gr.Interface(
    fn=classify_text_emotion,
    inputs=gr.Textbox(label="Enter Text for Emotion Classification"),
    outputs=gr.Textbox(label="Predicted Emotion")
)

# Emoji-based input interface
emoji_input = gr.Interface(
    fn=classify_emoji,
    inputs=gr.Textbox(label="Enter Emoji for Emotion Classification"),
    outputs=gr.Textbox(label="Predicted Emotion")
)

# Image-based input interface
image_input = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(label="Upload Image for Classification"),
    outputs=gr.Textbox(label="Predicted Image Label")
)

# Combine all three interfaces into one
demo = gr.TabbedInterface([text_input, emoji_input, image_input], tab_names=["Text Classification", "Emoji Classification", "Image Classification"])

# Launch the UI
demo.launch()