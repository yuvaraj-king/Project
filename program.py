import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import pipeline
from sklearn.metrics import accuracy_score
from collections import defaultdict
import re

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

# Extract top hashtags for grouping
df['Primary_Hashtag'] = df['Hashtags'].apply(lambda x: x.split()[0] if isinstance(x, str) and len(x.split()) > 0 else 'NoHashtag')

# Predict emotions
predicted_emotions = []

for index, row in df.iterrows():
    try:
        result = emotion_analyzer(row['Cleaned_Text'])[0][0]
        predicted_emotions.append(result['label'])
    except:
        predicted_emotions.append(None)

df['Predicted_Emotion'] = predicted_emotions


# Prepare for plotting
emotion_results = defaultdict(list)
for index, row in df.iterrows():
    if pd.notna(row['Predicted_Emotion']):
        emotion_results['Hashtag'].append(row['Primary_Hashtag'])
        emotion_results['Emotion'].append(row['Predicted_Emotion'])

emotion_df = pd.DataFrame(emotion_results)

# Plot the emotion distribution per hashtag
plt.figure(figsize=(14, 7))
sns.countplot(data=emotion_df, x='Emotion', hue='Hashtag')
plt.title('Emotion Distribution by Hashtag Topic')
plt.xlabel('Emotion')
plt.ylabel('Tweet Count')
plt.legend(title='Hashtag', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

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

# Ask user for emoji input
emoji_input = input("Enter an emoji to classify: ").strip()

# Classify the emoji
emotion = emoji_emotions.get(emoji_input, 'Unknown or not in database')

# Display result
print(f"Emoji: {emoji_input} => Emotion: {emotion}")

# Accuracy calculation
valid_df = df.dropna(subset=['True_Emotion', 'Predicted_Emotion'])
accuracy = accuracy_score(valid_df['True_Emotion'], valid_df['Predicted_Emotion'])
print(f"Emotion Classification Accuracy (on labeled subset): {accuracy:.2f}")