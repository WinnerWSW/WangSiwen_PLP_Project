from datasets import load_dataset
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import KFold
import seaborn as sns
import numpy as np
import random
from transformers import set_seed
from nltk.corpus import wordnet

# Set random seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
set_seed(SEED)

# Load the TweetEval dataset
dataset = load_dataset("tweet_eval", "sentiment")

# Load the BERT tokenizer and model
model = TFBertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=3,
    hidden_dropout_prob=0.3,
    attention_probs_dropout_prob=0.3,
    num_hidden_layers=4
)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Save the cached model locally
model.save_pretrained('./cached_bert_twitter')
tokenizer.save_pretrained('./cached_bert_twitter')

# Synonym replacement function
def synonym_replacement(text):
    words = text.split()
    new_words = words.copy()
    random_word_list = list(set([word for word in words if wordnet.synsets(word)]))
    if len(random_word_list) > 0:
        word_to_replace = random.choice(random_word_list)
        synonyms = wordnet.synsets(word_to_replace)[0].lemma_names()
        synonym = random.choice(synonyms)
        index_to_replace = new_words.index(word_to_replace)
        new_words[index_to_replace] = synonym
    return ' '.join(new_words)

# Preprocess the data
def encode_data(comments, labels, max_length=512):
    encodings = tokenizer(comments, truncation=True, padding=True, max_length=max_length)
    dataset = tf.data.Dataset.from_tensor_slices((dict(encodings), labels))
    return dataset

# Apply data augmentation to the training set
train_texts = dataset['train']['text']
train_labels = dataset['train']['label']
augmented_texts = [synonym_replacement(text) for text in train_texts]

# Merge the original and augmented datasets
combined_texts = train_texts + augmented_texts
combined_labels = train_labels + train_labels

# K-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
fold_no = 1
all_fold_histories = []

for train_index, val_index in kf.split(combined_texts):
    print(f'Training fold {fold_no}...')

    train_texts_fold = [combined_texts[i] for i in train_index]
    train_labels_fold = [combined_labels[i] for i in train_index]
    val_texts_fold = [combined_texts[i] for i in val_index]
    val_labels_fold = [combined_labels[i] for i in val_index]

    train_dataset = encode_data(train_texts_fold, train_labels_fold)
    val_dataset = encode_data(val_texts_fold, val_labels_fold)

    # Load the model from the local cache
    model = TFBertForSequenceClassification.from_pretrained('./cached_bert_twitter')

    # Compile the model
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-5,
        decay_steps=10000,
        decay_rate=0.9)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, weight_decay=1e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    # Model Training
    history = model.fit(
        train_dataset.batch(16),
        validation_data=val_dataset.batch(16),
        epochs=15
    )

    # Save the training history for each fold
    all_fold_histories.append(history)

    # Predict the validation set
    y_pred_probs = model.predict(val_dataset.batch(16))
    y_pred_classes = np.argmax(y_pred_probs.logits, axis=1)
    y_true = val_labels_fold

    # Compute and plot the confusion matrix heatmap
    conf_matrix = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Negative", "Neutral", "Positive"], yticklabels=["Negative", "Neutral", "Positive"])
    plt.title(f'Fold {fold_no} Confusion Matrix Heatmap')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

    fold_no += 1

# Plot the training and validation loss and accuracy curves for each fold
def plot_kfold_training_history(histories):
    for i, history in enumerate(histories):
        plt.figure(figsize=(12, 5))

        # Plot the loss curve
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label=f'Fold {i + 1} Train Loss')
        plt.plot(history.history['val_loss'], label=f'Fold {i + 1} Validation Loss')
        plt.title(f'Fold {i + 1} Loss over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Plot the accuracy curve
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label=f'Fold {i + 1} Train Accuracy')
        plt.plot(history.history['val_accuracy'], label=f'Fold {i + 1} Validation Accuracy')
        plt.title(f'Fold {i + 1} Accuracy over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()

# Call function to plot K-fold training history
plot_kfold_training_history(all_fold_histories)

# Save the model
model.save_pretrained('./finetuned_bert_twitter')
tokenizer.save_pretrained('./finetuned_bert_twitter')
