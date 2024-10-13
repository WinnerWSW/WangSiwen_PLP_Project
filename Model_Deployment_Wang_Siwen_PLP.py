import re
import os
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, opinion_lexicon
from nltk.util import ngrams
from wordcloud import WordCloud
from collections import Counter
import emoji
import squarify
import matplotlib.pyplot as plt
import streamlit as st
from math import pi
import gdown
import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer
import logging
logging.basicConfig(level=logging.ERROR)

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('opinion_lexicon')

model_directory = "finetuned_bert_twitter"
os.makedirs(model_directory, exist_ok=True)

files_to_download = {
    "tf_model.h5": "13Xe_TiDmkjpFof2AO9rDTzwtdpTKFO-m",
    "vocab.txt": "1ruQPJSyCbuECkNZ8y0i8keELBsNjd6mF",
    "config.json": "1nPth_UtG18YGMKX_4MAxCJ5IrsHpIeZ6",
    "tokenizer_config.json": "1WQ5mbmlMPYTQLbh8F5jm-Jg9crDLHMt2",
    "special_tokens_map.json": "196GwvOcgDgK2b_8e91krExWz76LTR4nK"
}

for file_name, file_id in files_to_download.items():
    output = os.path.join(model_directory, file_name)
    gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)

if all(os.path.exists(os.path.join(model_directory, file)) for file in files_to_download):
    st.success("All model files downloaded successfully!")
else:
    st.error("Some files failed to download. Please check the file IDs.")

try:
    model = TFBertForSequenceClassification.from_pretrained('./finetuned_bert_twitter', from_tf=True)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop() 
try:
    tokenizer = BertTokenizer.from_pretrained('./finetuned_bert_twitter')
    st.success("Tokenizer loaded successfully!")
except Exception as e:
    st.error(f"Error loading the tokenizer: {e}")

def load_emoji_sentiment_mapping():
    try:
        csv_url = "https://raw.githubusercontent.com/WinnerWSW/WangSiwen_PLP_Project/master/Emoji_Sentiment_Data_v1.0.csv"
        emoji_df = pd.read_csv(csv_url)        
        emoji_sentiment_mapping = {}
        for _, row in emoji_df.iterrows():
            emoji_char = row['Emoji']
            positive_score = row['Positive']
            negative_score = row['Negative']
            neutral_score = row['Neutral']

            if positive_score > negative_score and positive_score > neutral_score:
                emoji_sentiment_mapping[emoji_char] = 'positive'
            elif negative_score > positive_score and negative_score > neutral_score:
                emoji_sentiment_mapping[emoji_char] = 'negative'
            else:
                emoji_sentiment_mapping[emoji_char] = 'neutral'
        return emoji_sentiment_mapping
    except Exception as e:
        st.error(f"Error loading emoji sentiment data: {e}")
        return {}

emoji_sentiment_mapping = load_emoji_sentiment_mapping()

if not emoji_sentiment_mapping:
    st.error("Failed to load emoji sentiment data.")
    
def sentiment_analysis_with_emoji(comment):
    max_length = 512
    label_mapping = {
        'LABEL_0': 1,
        'LABEL_1': 2,
        'LABEL_2': 3
    }

    emoji_sentiment = detect_emoji_sentiment(comment)
    if emoji_sentiment:
        return emoji_sentiment

    if len(comment) > max_length:
        sentiment_scores = []
        for i in range(0, len(comment), max_length):
            chunk = comment[i:i + max_length]
            sentiment_result = absa(chunk)[0]
            sentiment_label = sentiment_result['label']
            sentiment_value = label_mapping.get(sentiment_label, 2)
            sentiment_scores.append(sentiment_value)

        avg_sentiment_value = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 2
    else:
        sentiment_result = absa(comment)[0]
        sentiment_label = sentiment_result['label']
        avg_sentiment_value = label_mapping.get(sentiment_label, 2)

    if avg_sentiment_value > 2:
        return 'positive'
    elif avg_sentiment_value == 2:
        return 'neutral'
    else:
        return 'negative'

def detect_emoji_sentiment(comment):
    for char in comment:
        if char in emoji_sentiment_mapping:
            return emoji_sentiment_mapping[char]
    return None

st.title("YouTube Comment Sentiment Analysis")
st.success("Emoji sentiment data loaded successfully!")

@st.cache_resource
def get_youtube_client(api_key):
    from googleapiclient.discovery import build
    return build("youtube", "v3", developerKey=api_key)

video_url = st.text_input("Please enter the YouTube video URL")

def extract_video_id(video_url):
    video_id_match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", video_url)
    if video_id_match:
        return video_id_match.group(1)
    else:
        st.error("Invalid YouTube URL")
        return None

def remove_stop_words(text):
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'<.*?>', '', text)
    words = word_tokenize(text.lower())
    filtered_text = [word for word in words if word.isalpha() and word not in stop_words]
    return filtered_text

def get_all_youtube_comments_with_timestamps(video_id):
    comments = []
    timestamps = []
    next_page_token = None

    while True:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=next_page_token
        )
        response = request.execute()

        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            timestamp = item['snippet']['topLevelComment']['snippet']['publishedAt']
            comments.append(comment)
            timestamps.append(timestamp)

        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break

    return comments, timestamps

def classify_sentiments_with_emoji(comments):
    positive_comments, neutral_comments, negative_comments = [], [], []
    for comment in comments:
        sentiment = sentiment_analysis_with_emoji(comment)
        if sentiment == "positive":
            positive_comments.append(comment)
        elif sentiment == "neutral":
            neutral_comments.append(comment)
        else:
            negative_comments.append(comment)
    return positive_comments, neutral_comments, negative_comments

if video_url:
    video_id = extract_video_id(video_url)
    if video_id:
        st.write(f"Extracted Video ID: {video_id}")
        api_key = "AIzaSyCd-cjihctHlrQd-G7iNn2G3Un8awSXI34"
        youtube = get_youtube_client(api_key)

        comments, timestamps = get_all_youtube_comments_with_timestamps(video_id)
        st.write(f"Total number of comments fetched: {len(comments)}")

        if comments:
            positive_comments, neutral_comments, negative_comments = classify_sentiments_with_emoji(comments)
            st.write("Sentiment Distribution")
            positive_count, neutral_count, negative_count = len(positive_comments), len(neutral_comments), len(
                negative_comments)

            def generate_word_cloud(comments, sentiment_type):
                cleaned_comments = [" ".join(remove_stop_words(comment)) for comment in comments]
                text = " ".join(cleaned_comments)
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
                plt.figure(figsize=(10, 6))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis("off")
                plt.title(f"{sentiment_type} Comments Word Cloud", fontsize=20)
                st.pyplot(plt)

            if positive_comments:
                st.write("Positive Comments Word Cloud")
                generate_word_cloud(positive_comments, "Positive")

            if neutral_comments:
                st.write("Neutral Comments Word Cloud")
                generate_word_cloud(neutral_comments, "Neutral")

            if negative_comments:
                st.write("Negative Comments Word Cloud")
                generate_word_cloud(negative_comments, "Negative")

            positive_comments, neutral_comments, negative_comments = classify_sentiments_with_emoji(comments)

            st.write("Sentiment Distribution")
            positive_count = len(positive_comments)
            neutral_count = len(neutral_comments)
            negative_count = len(negative_comments)

            def plot_sentiment_distribution(positive_count, neutral_count, negative_count):
                labels = ['Positive', 'Neutral', 'Negative']
                sizes = [positive_count, neutral_count, negative_count]
                colors = ['green', 'gray', 'red']
                plt.figure(figsize=(7, 7))
                plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                plt.title("Sentiment Distribution", fontsize=20)
                st.pyplot(plt)

            plot_sentiment_distribution(positive_count, neutral_count, negative_count)

            def generate_word_cloud(comments, sentiment_type):
                cleaned_comments = [" ".join(remove_stop_words(comment)) for comment in comments]
                text = " ".join(cleaned_comments)
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
                plt.figure(figsize=(10, 6))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis("off")
                plt.title(f"{sentiment_type} Comments Word Cloud", fontsize=20)
                st.pyplot(plt)

            def detect_negation_phrases(comment):
                comment = comment.lower()

                negation_patterns = [
                    r"not \w+",
                    r"don't \w+",
                    r"dont \w+",
                    r"didn't \w+",
                    r"didnt \w+",
                    r"doesn't \w+"
                    r"never \w+",
                    r"can't \w+",
                    r"no \w+"
                    r"seldom \w+"
                    r"rarely \w+",
                    r"hardly \w+",
                    r"barely \w+",
                    r"won't \w+",
                    r"shouldn't \w+",
                    r"couldn't \w+",
                    r"wouldn't \w+",
                    r"nobody \w+",
                    r"nothing \w+",
                    r"nowhere \w+",
                    r"neither \w+",
                    r"lack of \w+",
                    r"without \w+",
                    r"least \w+",
                    r"few \w+",
                ]
                combined_pattern = "|".join(negation_patterns)
                matches = re.findall(combined_pattern, comment)

                return matches

            def filter_sentiment_words_with_negation(comments):
                all_words = []

                positive_words = set(opinion_lexicon.positive())
                negative_words = set(opinion_lexicon.negative())

                for comment in comments:
                    # Detect negation phrases and treat them as single tokens
                    negation_phrases = detect_negation_phrases(comment)

                    # Add negation phrases to word list
                    all_words.extend(
                        [phrase for phrase in negation_phrases if
                         any(word in phrase for word in positive_words | negative_words)])

                    # Clean and tokenize the rest of the comment
                    words = remove_stop_words(comment)

                    # Add regular words that are sentiment-related to word list
                    sentiment_words = [word for word in words if word in positive_words or word in negative_words]
                    all_words.extend(sentiment_words)

                return all_words

            def get_top_words_and_plot_with_negation(comments, sentiment_type, top_n=10):
                all_words = filter_sentiment_words_with_negation(comments)
                word_freq = Counter(all_words)
                top_words = word_freq.most_common(top_n)
                words, frequencies = zip(*top_words)
                plt.figure(figsize=(10, 6))
                plt.bar(words, frequencies, color='blue')
                plt.title(f"Top {top_n} Words in {sentiment_type} Comments (With Negation Handling)")
                plt.xticks(rotation=45)
                plt.ylabel("Frequency")
                st.pyplot(plt)

            if positive_comments:
                st.write("Top Words in Positive Comments")
                get_top_words_and_plot_with_negation(positive_comments, "Positive", top_n=15)

            if neutral_comments:
                st.write("Top Words in Neutral Comments")
                get_top_words_and_plot_with_negation(neutral_comments, "Neutral", top_n=15)

            if negative_comments:
                st.write("Top Words in Negative Comments")
                get_top_words_and_plot_with_negation(negative_comments, "Negative", top_n=15)

            # Function to generate bigrams from a comment
            def generate_bigrams(text):
                words = remove_stop_words(text)
                bigrams = ngrams(words, 2)
                return [' '.join(gram) for gram in bigrams]

            # Function to get top bigrams
            def get_top_bigrams(comments, top_n=10):
                all_bigrams = []

                for comment in comments:
                    bigrams = generate_bigrams(comment)
                    all_bigrams.extend(bigrams)

                bigram_freq = Counter(all_bigrams)
                top_bigrams = bigram_freq.most_common(top_n)

                return top_bigrams

            def plot_top_bigrams(comments, sentiment_type, top_n=10):
                top_bigrams = get_top_bigrams(comments, top_n=top_n)
                if not top_bigrams:
                    st.write(f"No top bigrams found for {sentiment_type} comments.")
                    return
                bigrams, frequencies = zip(*top_bigrams)
                plt.figure(figsize=(10, 6))
                plt.bar(bigrams, frequencies, width=0.6, color='blue')
                plt.title(f"Top {top_n} Bigrams in {sentiment_type} Comments")
                plt.xticks(rotation=45)
                plt.ylabel("Frequency")
                st.pyplot(plt)

            if positive_comments:
                st.write("Top Bigrams in Positive Comments")
                plot_top_bigrams(positive_comments, "Positive", top_n=15)

            if neutral_comments:
                st.write("Top Bigrams in Neutral Comments")
                plot_top_bigrams(neutral_comments, "Neutral", top_n=15)

            if negative_comments:
                st.write("Top Bigrams in Negative Comments")
                plot_top_bigrams(negative_comments, "Negative", top_n=15)

            # Clean each comment, filtering out empty strings
            cleaned_comments = [remove_stop_words(comment) for comment in comments if comment]

            def classify_comment_aspect(comment):
                aspects = []

                if any(word in comment for word in
                       ["plot", "story", "narrative", "script", "writing", "storyline", "arc", "twist", "dialogue",
                        "premise",
                        "theme", "structure", "pace", "subplot"]):
                    aspects.append("plot")
                if any(word in comment for word in
                       ["acting", "performance", "cast", "character", "portrayal", "role", "chemistry", "emotion",
                        "delivery",
                        "dialogue", "characterization", "dubbing", "voice-over", "voice", "actor", "actress",
                        "screenplay",
                        "monologue", "rehearsal", "staging", "blocking", "cue", "stage presence", "improv",
                        "understudy"]):
                    aspects.append("acting")
                if any(word in comment for word in
                       ["visual", "effects", "cinematography", "graphics", "animation", "CGI", "lighting", "aesthetics",
                        "art",
                        "scenery", "imagery", "sight", "SFX", "VFX", "filmmaking", "camera", "shot", "photography",
                        "design", "2D",
                        "3D", "cartooning", "stop-motion", "illumination", "style", "scenography", "backdrop",
                        "landscaping"]):
                    aspects.append("visual effects")
                if any(word in comment for word in
                       ["sound", "music", "score", "soundtrack", "audio", "song", "track", "composition", "melody",
                        "orchestration",
                        "tune", "vocals", "BGM", "OST", "harmony", "rhythm", "instrumentation", "acoustics", "beats",
                        "arrangement",
                        "lyrics", "chorus", "symphony", "jingle", "motif", "mixing"]):
                    aspects.append("soundtrack")

                # If no specific aspect is found, classify as "overall"
                if not aspects:
                    aspects.append("overall")

                return aspects

            # Assume LABEL_0 = Negative, LABEL_1 = Neutral, LABEL_2 = Positive
            label_mapping = {
                'LABEL_0': 1,
                'LABEL_1': 2,
                'LABEL_2': 3
            }

            def analyze_aspects_with_absa(comments, max_length=512):
                aspect_sentiment_scores = {
                    "plot": [],
                    "acting": [],
                    "visual effects": [],
                    "soundtrack": [],
                    "overall": []
                }

                for comment in comments:
                    comment_text = " ".join(comment)

                    if len(comment_text) > max_length:
                        sentiment_scores = []
                        for i in range(0, len(comment_text), max_length):
                            chunk = comment_text[i:i + max_length]
                            sentiment_result = absa(chunk)[0]
                            label = sentiment_result['label']
                            sentiment_value = label_mapping.get(label, 2)
                            sentiment_scores.append(sentiment_value)

                        # Calculate the average sentiment score for long texts
                        if len(sentiment_scores) > 0:
                            avg_sentiment_value = sum(sentiment_scores) / len(sentiment_scores)
                        else:
                            avg_sentiment_value = 2

                    else:
                        sentiment_result = absa(comment_text)[0]
                        label = sentiment_result['label']
                        avg_sentiment_value = label_mapping.get(label, 2)

                    # Detect the aspects involved in the comments
                    aspects = classify_comment_aspect(comment_text)
                    if aspects:
                        for aspect in aspects:
                            if aspect in aspect_sentiment_scores:
                                aspect_sentiment_scores[aspect].append(avg_sentiment_value)

                # Return the sentiment score for each aspect
                return aspect_sentiment_scores

            def plot_aspect_distribution(aspect_sentiment_scores):
                aspect_counts = {
                    "plot": len(aspect_sentiment_scores["plot"]),
                    "acting": len(aspect_sentiment_scores["acting"]),
                    "visual effects": len(aspect_sentiment_scores["visual effects"]),
                    "soundtrack": len(aspect_sentiment_scores["soundtrack"]),
                    "overall": len(aspect_sentiment_scores["overall"]),
                }

                labels = list(aspect_counts.keys())
                sizes = list(aspect_counts.values())
                colors = ['gold', 'lightcoral', 'lightskyblue', 'lightgreen', 'violet']

                fig, ax = plt.subplots(figsize=(8, 8))
                ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
                ax.axis('equal')
                plt.title("Aspect Sentiment Distribution", fontsize=16)

                st.pyplot(fig)

            aspect_sentiment_scores = analyze_aspects_with_absa(cleaned_comments)
            plot_aspect_distribution(aspect_sentiment_scores)

            def count_aspects_and_scores(comments, sentiment_scores):
                aspect_scores = {
                    "plot": [],
                    "acting": [],
                    "visual effects": [],
                    "soundtrack": [],
                    "overall": []
                }

                for comment, score in zip(comments, sentiment_scores):
                    aspects = classify_comment_aspect(comment.lower())
                    for aspect in aspects:
                        if aspect in aspect_scores:
                            aspect_scores[aspect].append(score)

                aspect_counts_and_scores = {}
                for aspect, scores in aspect_scores.items():
                    if scores:
                        avg_score = sum(scores) / len(scores)
                        aspect_counts_and_scores[aspect] = (len(scores), avg_score)
                    else:
                        aspect_counts_and_scores[aspect] = (0, 0)

                return aspect_counts_and_scores

            # Ensure the lengths of comments, sentiment scores, and timestamps are consistent
            sentiment_scores = []
            for comment in cleaned_comments:
                comment_text = " ".join(comment)
                sentiment_result = absa(comment_text)[0]
                label = sentiment_result['label']
                sentiment_value = label_mapping.get(label, 2)
                sentiment_scores.append(sentiment_value)

            aspect_scores = count_aspects_and_scores(comments, sentiment_scores)

            labels = [f"{aspect}\n{count} comments\nAvg. score: {score:.2f}"
                      for aspect, (count, score) in aspect_scores.items()]
            sizes = [count for count, score in aspect_scores.values()]
            colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0']

            fig = plt.figure(figsize=(10, 6))
            squarify.plot(sizes=sizes, label=labels, color=colors, alpha=0.7)
            plt.title('Comment Distribution and Sentiment Scores by Aspect', fontsize=20)
            plt.axis('off')

            st.pyplot(fig)

            positive_aspect_scores = count_aspects_and_scores(positive_comments, sentiment_scores)
            neutral_aspect_scores = count_aspects_and_scores(neutral_comments, sentiment_scores)
            negative_aspect_scores = count_aspects_and_scores(negative_comments, sentiment_scores)

            # Prepare the function for plotting the treemap
            def plot_treemap(aspect_scores, title):
                labels = [f"{aspect}\n{count} comments\nAvg. score: {score:.2f}"
                          for aspect, (count, score) in aspect_scores.items()]
                sizes = [count for count, score in aspect_scores.values()]
                colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0']

                # Plot the treemap
                plt.figure(figsize=(10, 6))
                squarify.plot(sizes=sizes, label=labels, color=colors, alpha=0.7)
                plt.title(title, fontsize=20)
                plt.axis('off')
                st.pyplot(fig)

            st.write("### Positive Comments - Aspect Sentiment Distribution")
            plot_treemap(positive_aspect_scores, 'Positive Comments - Aspect Sentiment Distribution')

            st.write("### Neutral Comments - Aspect Sentiment Distribution")
            plot_treemap(neutral_aspect_scores, 'Neutral Comments - Aspect Sentiment Distribution')

            st.write("### Negative Comments - Aspect Sentiment Distribution")
            plot_treemap(negative_aspect_scores, 'Negative Comments - Aspect Sentiment Distribution')

            # Prepare the data for the stacked bar chart
            def get_aspect_comment_counts(aspect_scores):
                return {aspect: count for aspect, (count, _) in aspect_scores.items()}

            # Get the number of comments for each sentiment category in each aspect
            positive_counts = get_aspect_comment_counts(positive_aspect_scores)
            neutral_counts = get_aspect_comment_counts(neutral_aspect_scores)
            negative_counts = get_aspect_comment_counts(negative_aspect_scores)

            # Define each aspect
            aspects = list(positive_counts.keys())

            # Number of comments in each sentiment category
            positive_values = [positive_counts[aspect] for aspect in aspects]
            neutral_values = [neutral_counts[aspect] for aspect in aspects]
            negative_values = [negative_counts[aspect] for aspect in aspects]

            # Calculate the total number of comments for each aspect
            total_comments = [positive_values[i] + neutral_values[i] + negative_values[i] for i in range(len(aspects))]

            # Calculate the percentage of each sentiment category
            positive_percents = [positive_values[i] / total_comments[i] for i in range(len(aspects))]
            neutral_percents = [neutral_values[i] / total_comments[i] for i in range(len(aspects))]
            negative_percents = [negative_values[i] / total_comments[i] for i in range(len(aspects))]

            # Generate the stacked bar chart
            r = np.arange(len(aspects))
            bar_width = 0.6

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(r, negative_percents, color='#ff9999', edgecolor='grey', label='Negative')
            ax.bar(r, neutral_percents, bottom=negative_percents, color='#66b3ff', edgecolor='grey', label='Neutral')
            ax.bar(r, positive_percents, bottom=[i + j for i, j in zip(negative_percents, neutral_percents)],
                   color='#99ff99', edgecolor='grey', label='Positive')

            ax.set_xticks(r)
            ax.set_xticklabels(aspects, rotation=45)

            for i in range(len(r)):
                ax.text(r[i], negative_percents[i] / 2, f'{negative_percents[i]:.0%}', ha='center', va='center',
                        color='black')
                ax.text(r[i], negative_percents[i] + neutral_percents[i] / 2, f'{neutral_percents[i]:.0%}', ha='center',
                        va='center', color='black')
                ax.text(r[i], negative_percents[i] + neutral_percents[i] + positive_percents[i] / 2,
                        f'{positive_percents[i]:.0%}', ha='center', va='center', color='black')

            ax.set_title('Stacked Bar Chart - Sentiment Distribution by Aspect (Percentage)', fontsize=16)
            ax.set_xlabel('Aspect', fontsize=12)
            ax.set_ylabel('Percentage of Comments', fontsize=12)
            ax.legend()

            plt.tight_layout()

            st.pyplot(fig)

            # Prepare the proportion of each aspect in different sentiment categories
            def get_aspect_percentage(aspect_scores, total_scores):
                return {aspect: [aspect_scores['Negative'][aspect] / total_scores[aspect],
                                 aspect_scores['Neutral'][aspect] / total_scores[aspect],
                                 aspect_scores['Positive'][aspect] / total_scores[aspect]] for aspect in aspects}

            # Define the number of aspect comments for each sentiment category
            aspect_scores = {
                'Negative': negative_counts,
                'Neutral': neutral_counts,
                'Positive': positive_counts
            }

            # Calculate the total number of comments for each aspect
            total_scores = {aspect: positive_counts[aspect] + neutral_counts[aspect] + negative_counts[aspect] for
                            aspect in aspects}

            # Calculate the proportion of each aspect within each sentiment category
            aspect_percentages = get_aspect_percentage(aspect_scores, total_scores)

            # Prepare the data: the proportion of each sentiment category for each aspect
            aspect_categories = aspects
            num_aspects = len(aspect_categories)

            # Calculate the average proportion for each aspect within each sentiment category
            negative_values = [negative_counts[aspect] / total_scores[aspect] for aspect in aspects]
            neutral_values = [neutral_counts[aspect] / total_scores[aspect] for aspect in aspects]
            positive_values = [positive_counts[aspect] / total_scores[aspect] for aspect in aspects]

            # Close the loop of the data to fit the radar chart
            negative_values += negative_values[:1]
            neutral_values += neutral_values[:1]
            positive_values += positive_values[:1]

            # Create angles for the radar chart
            angles = [n / float(num_aspects) * 2 * pi for n in range(num_aspects)]
            angles += angles[:1]

            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

            ax.plot(angles, negative_values, linewidth=2, linestyle='solid', label='Negative')
            ax.fill(angles, negative_values, color='#ff9999', alpha=0.4)

            ax.plot(angles, neutral_values, linewidth=2, linestyle='solid', label='Neutral')
            ax.fill(angles, neutral_values, color='#66b3ff', alpha=0.4)

            ax.plot(angles, positive_values, linewidth=2, linestyle='solid', label='Positive')
            ax.fill(angles, positive_values, color='#99ff99', alpha=0.4)

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(aspects)

            plt.title('Aspect Sentiment Distribution - Radar Chart', size=16, color='black', y=1.1)
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

            plt.tight_layout()

            st.pyplot(fig)

            # Extract all emojis from the comments
            def extract_emojis(comment):
                return [char for char in comment if char in emoji.EMOJI_DATA]

            # Count the frequency of emoji usage in all comments and calculate their percentage
            def calculate_emoji_percentage(comments):
                all_emojis = []

                # Extract emojis from each comment and add them to a list
                for comment in comments:
                    all_emojis.extend(extract_emojis(comment))

                # Count the occurrences of each emoji
                emoji_counter = Counter(all_emojis)

                # Calculate the total number of emojis
                total_emojis = sum(emoji_counter.values())

                # Calculate and return the percentage of each emoji
                emoji_percentage = {emoji_char: (count / total_emojis) * 100 for emoji_char, count in
                                    emoji_counter.items()}

                # Sort the emojis by percentage in descending order
                emoji_percentage_sorted = dict(sorted(emoji_percentage.items(), key=lambda item: item[1], reverse=True))

                return emoji_percentage_sorted

            def display_emoji_percentage(comments):
                emoji_percentage = calculate_emoji_percentage(comments)

                st.write("### Emoji Usage Percentage in the Comments (sorted by percentage):")
                if emoji_percentage:
                    for emoji_char, percentage in emoji_percentage.items():
                        st.write(f"**Emoji**: {emoji_char}, **Percentage**: {percentage:.2f}%")
                else:
                    st.write("No emojis found in the comments.")


            display_emoji_percentage(comments)
