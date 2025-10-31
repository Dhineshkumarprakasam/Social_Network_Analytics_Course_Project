import os
import numpy as np
import json
import uuid
from datetime import datetime
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify
import mysql.connector
from mysql.connector import Error
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve

nlp = spacy.load("en_core_web_md")

# ------------------ MySQL Database Configuration ------------------ #
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'root',
    'database': 'vector_db'
}

# ------------------ Utility Functions ------------------ #
def keep_nouns_adjs(text):
    doc = nlp(text)
    tokens = [
        token.lemma_.lower()
        for token in doc
        if token.pos_ in ["NOUN", "PROPN", "ADJ"] and not token.is_stop and not token.is_punct
    ]
    return " ".join(tokens)

def text_to_vector(text):
    clean_text = keep_nouns_adjs(text)
    doc = nlp(clean_text)
    return doc.vector

# ------------------ Vector Database Class ------------------ #
class VectorDatabase:
    def __init__(self, db_config, similarity_threshold=0.7):
        self.similarity_threshold = similarity_threshold
        self.db_config = db_config
        self._init_db()
    
    def _connect(self):
        return mysql.connector.connect(**self.db_config)
    
    def _init_db(self):
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vector_database (
                id VARCHAR(255) PRIMARY KEY,
                text TEXT NOT NULL,
                vector TEXT NOT NULL,
                timestamp VARCHAR(50) NOT NULL
            )
        """)
        conn.commit()
        cursor.close()
        conn.close()
    
    def add_or_find_duplicate(self, text):
        new_vector = text_to_vector(text)
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute("SELECT id, text, vector, timestamp FROM vector_database")
        rows = cursor.fetchall()
        
        for row in rows:
            entry_id, entry_text, entry_vector_json, timestamp = row
            entry_vector = np.array(json.loads(entry_vector_json))
            sim = cosine_similarity(new_vector.reshape(1, -1), entry_vector.reshape(1, -1))[0][0]
            
            if sim >= self.similarity_threshold:
                cursor.close()
                conn.close()
                return {
                    'status': 'duplicate',
                    'similarity': sim,
                    'id': entry_id,
                    'text': entry_text,
                    'timestamp': timestamp
                }
        
        entry_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().isoformat()
        cursor.execute(
            "INSERT INTO vector_database (id, text, vector, timestamp) VALUES (%s, %s, %s, %s)",
            (entry_id, text, json.dumps(new_vector.tolist()), timestamp)
        )
        conn.commit()
        cursor.close()
        conn.close()
        
        return {
            'status': 'added',
            'id': entry_id,
            'text': text,
            'timestamp': timestamp
        }
    
    def calculate_similarity(self, text1, text2):
        """Calculate similarity between two texts without adding to database"""
        vec1 = text_to_vector(text1)
        vec2 = text_to_vector(text2)
        sim = cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]
        return sim
    
    def clear_database(self):
        """Clear all entries from the database"""
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM vector_database")
        conn.commit()
        cursor.close()
        conn.close()

# ------------------ Evaluation Functions ------------------ #
def evaluate_duplicate_detection(df, similarity_threshold=0.5):
    """
    Evaluate the duplicate detection system on the dataset
    
    Parameters:
    - df: DataFrame with columns 'text1', 'text2', 'label'
    - similarity_threshold: threshold for duplicate detection
    
    Returns:
    - Dictionary with evaluation metrics and predictions
    """
    db = VectorDatabase(DB_CONFIG, similarity_threshold=similarity_threshold)
    
    # Clear database before evaluation
    db.clear_database()
    
    predictions = []
    true_labels = []
    similarity_scores = []
    
    print(f"Evaluating with threshold: {similarity_threshold}")
    print("-" * 50)
    
    for idx, row in df.iterrows():
        text1 = row['text1']
        text2 = row['text2']
        true_label = row['label']  # 1 = duplicate, 0 = not duplicate
        
        # Calculate similarity between text1 and text2
        similarity = db.calculate_similarity(text1, text2)
        similarity_scores.append(similarity)
        
        # Predict: if similarity >= threshold, it's a duplicate (1), else not (0)
        predicted_label = 1 if similarity >= similarity_threshold else 0
        
        predictions.append(predicted_label)
        true_labels.append(true_label)
        
        if idx < 5:  # Print first 5 examples
            print(f"Example {idx + 1}:")
            print(f"  Text1: {text1[:50]}...")
            print(f"  Text2: {text2[:50]}...")
            print(f"  Similarity: {similarity:.4f}")
            print(f"  True Label: {true_label}, Predicted: {predicted_label}")
            print()
    
    return {
        'predictions': predictions,
        'true_labels': true_labels,
        'similarity_scores': similarity_scores
    }

def calculate_metrics(true_labels, predictions, similarity_scores):
    """Calculate evaluation metrics"""
    
    # Confusion Matrix
    cm = confusion_matrix(true_labels, predictions)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'specificity': specificity,
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn
    }
    
    return metrics, cm

def plot_evaluation_results(true_labels, predictions, similarity_scores, metrics, cm, threshold):
    """Create visualization of evaluation results"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Duplicate Detection Evaluation (Threshold: {threshold})', fontsize=16, fontweight='bold')
    
    # 1. Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                xticklabels=['Not Duplicate', 'Duplicate'],
                yticklabels=['Not Duplicate', 'Duplicate'])
    axes[0, 0].set_title('Confusion Matrix')
    axes[0, 0].set_ylabel('True Label')
    axes[0, 0].set_xlabel('Predicted Label')
    
    # 2. Metrics Bar Chart
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity']
    metric_values = [metrics['accuracy'], metrics['precision'], metrics['recall'], 
                     metrics['f1_score'], metrics['specificity']]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    bars = axes[0, 1].bar(metric_names, metric_values, color=colors, alpha=0.7)
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].set_title('Performance Metrics')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom')
    
    # 3. Similarity Score Distribution
    duplicate_scores = [similarity_scores[i] for i in range(len(similarity_scores)) if true_labels[i] == 1]
    non_duplicate_scores = [similarity_scores[i] for i in range(len(similarity_scores)) if true_labels[i] == 0]
    
    axes[0, 2].hist(duplicate_scores, bins=20, alpha=0.6, label='Duplicates (Label=1)', color='green', edgecolor='black')
    axes[0, 2].hist(non_duplicate_scores, bins=20, alpha=0.6, label='Non-Duplicates (Label=0)', color='red', edgecolor='black')
    axes[0, 2].axvline(threshold, color='blue', linestyle='--', linewidth=2, label=f'Threshold={threshold}')
    axes[0, 2].set_title('Similarity Score Distribution')
    axes[0, 2].set_xlabel('Similarity Score')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].legend()
    
    # 4. ROC Curve
    fpr, tpr, _ = roc_curve(true_labels, similarity_scores)
    roc_auc = auc(fpr, tpr)
    axes[1, 0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    axes[1, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    axes[1, 0].set_xlim([0.0, 1.0])
    axes[1, 0].set_ylim([0.0, 1.05])
    axes[1, 0].set_xlabel('False Positive Rate')
    axes[1, 0].set_ylabel('True Positive Rate')
    axes[1, 0].set_title('ROC Curve')
    axes[1, 0].legend(loc="lower right")
    axes[1, 0].grid(alpha=0.3)
    
    # 5. Precision-Recall Curve
    precision_curve, recall_curve, _ = precision_recall_curve(true_labels, similarity_scores)
    axes[1, 1].plot(recall_curve, precision_curve, color='blue', lw=2)
    axes[1, 1].set_xlabel('Recall')
    axes[1, 1].set_ylabel('Precision')
    axes[1, 1].set_title('Precision-Recall Curve')
    axes[1, 1].set_xlim([0.0, 1.0])
    axes[1, 1].set_ylim([0.0, 1.05])
    axes[1, 1].grid(alpha=0.3)
    
    # 6. Classification Summary Table
    axes[1, 2].axis('tight')
    axes[1, 2].axis('off')
    
    table_data = [
        ['Metric', 'Value'],
        ['Accuracy', f"{metrics['accuracy']:.3f}"],
        ['Precision', f"{metrics['precision']:.3f}"],
        ['Recall', f"{metrics['recall']:.3f}"],
        ['F1-Score', f"{metrics['f1_score']:.3f}"],
        ['Specificity', f"{metrics['specificity']:.3f}"],
        ['', ''],
        ['True Positives', f"{metrics['true_positives']}"],
        ['True Negatives', f"{metrics['true_negatives']}"],
        ['False Positives', f"{metrics['false_positives']}"],
        ['False Negatives', f"{metrics['false_negatives']}"],
        ['', ''],
        ['ROC AUC', f"{roc_auc:.3f}"]
    ]
    
    table = axes[1, 2].table(cellText=table_data, cellLoc='left', loc='center',
                            colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(2):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    axes[1, 2].set_title('Classification Summary')
    
    plt.tight_layout()
    plt.savefig(f'evaluation_results_threshold_{threshold}.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved as 'evaluation_results_threshold_{threshold}.png'")
    plt.show()

def find_optimal_threshold(df, thresholds=np.arange(0.3, 0.9, 0.05)):
    """Find the optimal threshold by testing multiple values"""
    
    results = []
    
    print("Testing different thresholds...")
    print("=" * 70)
    
    for threshold in thresholds:
        eval_results = evaluate_duplicate_detection(df, similarity_threshold=threshold)
        metrics, cm = calculate_metrics(
            eval_results['true_labels'],
            eval_results['predictions'],
            eval_results['similarity_scores']
        )
        
        results.append({
            'threshold': threshold,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score']
        })
        
        print(f"Threshold: {threshold:.2f} | Accuracy: {metrics['accuracy']:.3f} | "
              f"Precision: {metrics['precision']:.3f} | Recall: {metrics['recall']:.3f} | "
              f"F1: {metrics['f1_score']:.3f}")
    
    # Plot threshold comparison
    results_df = pd.DataFrame(results)
    
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['threshold'], results_df['accuracy'], marker='o', label='Accuracy', linewidth=2)
    plt.plot(results_df['threshold'], results_df['precision'], marker='s', label='Precision', linewidth=2)
    plt.plot(results_df['threshold'], results_df['recall'], marker='^', label='Recall', linewidth=2)
    plt.plot(results_df['threshold'], results_df['f1_score'], marker='d', label='F1-Score', linewidth=2)
    
    plt.xlabel('Similarity Threshold', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Performance Metrics vs Similarity Threshold', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('threshold_comparison.png', dpi=300, bbox_inches='tight')
    print("\nThreshold comparison saved as 'threshold_comparison.png'")
    plt.show()
    
    # Find best threshold based on F1-score
    best_idx = results_df['f1_score'].idxmax()
    best_threshold = results_df.loc[best_idx, 'threshold']
    best_f1 = results_df.loc[best_idx, 'f1_score']
    
    print("\n" + "=" * 70)
    print(f"OPTIMAL THRESHOLD: {best_threshold:.2f} (F1-Score: {best_f1:.3f})")
    print("=" * 70)
    
    return best_threshold, results_df

# ------------------ Main Execution ------------------ #
if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv("dataset.csv")
    
    print("=" * 70)
    print("DUPLICATE DETECTION SYSTEM EVALUATION")
    print("=" * 70)
    print(f"Dataset size: {len(df)} text pairs")
    print(f"Duplicates (label=1): {sum(df['label'] == 1)}")
    print(f"Non-duplicates (label=0): {sum(df['label'] == 0)}")
    print("=" * 70)
    print()
    
    # Step 1: Find optimal threshold
    print("STEP 1: Finding Optimal Threshold")
    print("-" * 70)
    best_threshold, threshold_results = find_optimal_threshold(df)
    print()
    
    # Step 2: Evaluate with optimal threshold
    print("STEP 2: Detailed Evaluation with Optimal Threshold")
    print("-" * 70)
    eval_results = evaluate_duplicate_detection(df, similarity_threshold=best_threshold)
    
    # Calculate metrics
    metrics, cm = calculate_metrics(
        eval_results['true_labels'],
        eval_results['predictions'],
        eval_results['similarity_scores']
    )
    
    # Print detailed results
    print("\n" + "=" * 70)
    print("FINAL EVALUATION RESULTS")
    print("=" * 70)
    print(f"Threshold: {best_threshold:.2f}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1-Score: {metrics['f1_score']:.3f}")
    print(f"Specificity: {metrics['specificity']:.3f}")
    print()
    print("Confusion Matrix:")
    print(f"  True Positives: {metrics['true_positives']}")
    print(f"  True Negatives: {metrics['true_negatives']}")
    print(f"  False Positives: {metrics['false_positives']}")
    print(f"  False Negatives: {metrics['false_negatives']}")
    print("=" * 70)
    
    # Step 3: Visualize results
    print("\nSTEP 3: Creating Visualizations...")
    plot_evaluation_results(
        eval_results['true_labels'],
        eval_results['predictions'],
        eval_results['similarity_scores'],
        metrics,
        cm,
        best_threshold
    )
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE!")
    print("=" * 70)
