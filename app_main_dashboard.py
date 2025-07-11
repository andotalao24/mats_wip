from flask import Flask, jsonify, request
import json
import os
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import base64
import io
from functools import lru_cache
from analyze_checkpoint_files import load_checkpoint_data
from utils_eval import compare_answers

app = Flask(__name__)

# Global cache for data
data_cache = {}
examples_cache = {}  # Cache for examples data by record ID
OUTPUT_DIR = "data/ckpt_test"


def read_jsonl(filepath):
    """Read JSONL file and return list of JSON objects"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data



def compute_answer_top_prediction_checkpoint(record):
    """Compute the checkpoint position where the answer becomes the top prediction"""
    checkpoints = list(record['checkpoint_analysis'].values())
    total_checkpoints = len(checkpoints)
    for checkpoint_data in checkpoints:
        checkpoint_idx = int(checkpoint_data['checkpoint_idx']) 
        top_preds = checkpoint_data['top_predictions']
        if top_preds and len(top_preds) > 0 and top_preds[0]['token_id'] == checkpoint_data['ans_token_id']:
            return checkpoint_idx, (checkpoint_idx/ (total_checkpoints-1)) * 100

    if checkpoints and checkpoints[-1]['top_predictions'] and len(checkpoints[-1]['top_predictions']) > 0:
        print(checkpoints[-1]['top_predictions'][0]['probability'],checkpoints[-1]['top_predictions'][0]['token_text'],checkpoints[-1]['ans_token_id'],checkpoints[-1]['ans_probability'])
    return total_checkpoints,100

def track_top_predictions_base64(record):
    """Generate top predictions plot as base64"""
    try:
        checkpoints = {}
        ans_token_id = record['checkpoint_analysis']['checkpoint_1']['ans_token_id']
      
        for checkpoint_key, checkpoint_data in record['checkpoint_analysis'].items():
            checkpoint_idx = checkpoint_data['checkpoint_idx']
            top_preds = checkpoint_data['top_predictions']
            if top_preds and len(top_preds) > 0:
                checkpoints[checkpoint_idx] = {
                    'top_predictions': top_preds,
                    'top_token': top_preds[0]['token_text'],
                    'top_probability': top_preds[0]['probability'],
                    'top_token_id': top_preds[0].get('token_id')
                }

        if not checkpoints:
            return None, None

        sorted_checkpoints = sorted(checkpoints.keys())
        top_tokens = [checkpoints[cp]['top_token'] for cp in sorted_checkpoints]
        top_probs = [checkpoints[cp]['top_probability'] for cp in sorted_checkpoints]
        top_token_ids = [checkpoints[cp]['top_token_id'] for cp in sorted_checkpoints]
        
        # Find checkpoint where top prediction matches answer token
        answer_checkpoint = None
        for i, (cp, token_id) in enumerate(zip(sorted_checkpoints, top_token_ids)):
            if token_id == ans_token_id:
                answer_checkpoint = cp
                break
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot 1: Top prediction probability evolution
        ax1.plot(sorted_checkpoints, top_probs, 'o-', linewidth=2, markersize=8, color='red')
        ax1.set_xlabel('Checkpoint Index')
        ax1.set_ylabel('Probability')
        ax1.set_title('Top Prediction Probability Evolution (first generated token)')
        ax1.grid(True, alpha=0.3)

        # Add token annotations and highlight answer checkpoint
        for i, (cp, token, prob) in enumerate(zip(sorted_checkpoints, top_tokens, top_probs)):
            if cp == answer_checkpoint:
                ax1.annotate(f"'{token}' ✓", xy=(cp, prob), xytext=(0, 15),
                           textcoords='offset points', ha='center', fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='lime', alpha=0.8),
                           arrowprops=dict(arrowstyle='->', color='green', lw=2))
                ax1.axvline(x=cp, color='green', linestyle='--', alpha=0.7, linewidth=2)
            else:
                ax1.annotate(f"'{token}'", xy=(cp, prob), xytext=(0, 10),
                           textcoords='offset points', ha='center', fontsize=8,
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='lightblue', alpha=0.7))

        # Plot 2: Token distribution heatmap
        all_tokens = []
        for cp in sorted_checkpoints:
            all_tokens.extend([pred['token_text'] for pred in checkpoints[cp]['top_predictions']])
        
        unique_tokens = list(set(all_tokens))
        token_to_idx = {token: idx for idx, token in enumerate(unique_tokens)}

        freq_matrix = np.zeros((len(unique_tokens), len(sorted_checkpoints)))
        for cp_idx, cp in enumerate(sorted_checkpoints):
            cp_data = checkpoints[cp]
            for pred in cp_data['top_predictions']:
                token = pred['token_text']
                if token in token_to_idx:
                    freq_matrix[token_to_idx[token], cp_idx] = pred['probability']

        im = ax2.imshow(freq_matrix, cmap='YlOrRd', aspect='auto')
        ax2.set_xlabel('Checkpoint Index')
        ax2.set_ylabel('Token')
        ax2.set_title('Token Probability Distribution Across Checkpoints')
        ax2.set_yticks(range(len(unique_tokens)))
        ax2.set_yticklabels(unique_tokens, fontsize=8)
        ax2.set_xticks(range(len(sorted_checkpoints)))
        ax2.set_xticklabels(sorted_checkpoints)

        if answer_checkpoint is not None:
            answer_cp_idx = sorted_checkpoints.index(answer_checkpoint)
            ax2.axvline(x=answer_cp_idx, color='green', linestyle='--', alpha=0.8, linewidth=3)
            ax2.text(answer_cp_idx, -0.5, f'ANSWER\n(CP {answer_checkpoint})', ha='center', va='top', 
                    fontsize=10, fontweight='bold', color='green',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

        plt.colorbar(im, ax=ax2, label='Probability')
        
        if answer_checkpoint is not None:
            fig.suptitle(f'Answer Found at Checkpoint {answer_checkpoint}', 
                        fontsize=14, fontweight='bold', color='green')
        else:
            fig.suptitle('Answer Token Not Found in Top Predictions', 
                        fontsize=14, fontweight='bold', color='red')
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        return image_base64, answer_checkpoint
        
    except Exception as e:
        print(f"Error in track_top_predictions_base64: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def track_answer_probability_base64(record):
    """Generate answer probability plot as base64"""
    checkpoints = {}
    for checkpoint_key, checkpoint_data in record['checkpoint_analysis'].items():
        checkpoint_idx = checkpoint_data['checkpoint_idx']
        ans_prob = checkpoint_data.get('ans_probability')
        if ans_prob is not None:
            checkpoints[checkpoint_idx] = ans_prob

    if not checkpoints:
        return None

    sorted_checkpoints = sorted(checkpoints.keys())
    probabilities = [checkpoints[cp] for cp in sorted_checkpoints]
    
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_checkpoints, probabilities, 'o-', linewidth=2, markersize=8, color='blue')
    plt.xlabel('Checkpoint Index')
    plt.ylabel('Answer Probability')
    plt.title(f'Answer Probability Evolution - Answer: {record.get("extracted_ans", "N/A")}')
    plt.grid(True, alpha=0.3)

    max_prob_idx = np.argmax(probabilities)
    min_prob_idx = np.argmin(probabilities)

    plt.annotate(f'Max: {probabilities[max_prob_idx]:.4f}',
                xy=(sorted_checkpoints[max_prob_idx], probabilities[max_prob_idx]),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.annotate(f'Min: {probabilities[min_prob_idx]:.4f}',
                xy=(sorted_checkpoints[min_prob_idx], probabilities[min_prob_idx]),
                xytext=(10, -20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.tight_layout()
    
    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    return image_base64

def generate_checkpoint_text_evolution_base64(record):
    """Generate text evolution plot as base64"""
    checkpoints = {}
    for checkpoint_key, checkpoint_data in record['checkpoint_analysis'].items():
        checkpoint_idx = checkpoint_data['checkpoint_idx']
        text = checkpoint_data.get('text_up_to_checkpoint', '')
        if text:
            checkpoints[checkpoint_idx] = len(text)

    if not checkpoints:
        return None

    sorted_checkpoints = sorted(checkpoints.keys())
    text_lengths = [checkpoints[cp] for cp in sorted_checkpoints]
    
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_checkpoints, text_lengths, 'o-', linewidth=2, markersize=8, color='green')
    plt.xlabel('Checkpoint Index')
    plt.ylabel('Text Length (Characters)')
    plt.title(f'Text Length Evolution Across Checkpoints')
    plt.grid(True, alpha=0.3)

    # Add annotations for min and max lengths
    if len(text_lengths) > 1:
        max_length_idx = np.argmax(text_lengths)
        min_length_idx = np.argmin(text_lengths)

        plt.annotate(f'Max: {text_lengths[max_length_idx]}',
                    xy=(sorted_checkpoints[max_length_idx], text_lengths[max_length_idx]),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        plt.annotate(f'Min: {text_lengths[min_length_idx]}',
                    xy=(sorted_checkpoints[min_length_idx], text_lengths[min_length_idx]),
                    xytext=(10, -20), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.tight_layout()
    
    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    return image_base64

def generate_text_evolution_html(record):
    """Generate HTML for text evolution across checkpoints"""
    checkpoints = {}
    for checkpoint_key, checkpoint_data in record['checkpoint_analysis'].items():
        checkpoint_idx = checkpoint_data['checkpoint_idx']
        text = checkpoint_data.get('text_up_to_checkpoint', '')
        full_tokens = checkpoint_data['full_generated_text']
        
        if text:
            checkpoints[checkpoint_idx] = {
                'text': text,
                'full_tokens': full_tokens
            }

    if not checkpoints:
        return """
        <div class="text-evolution-section">
            <h4>Text Evolution Across Checkpoints</h4>
            <p class="no-data">No text data available for this example.</p>
        </div>
        """

    sorted_checkpoints = sorted(checkpoints.keys())
    
    # Create HTML for text evolution - display all checkpoints line by line
    html = """
    <div class="text-evolution-section">
        <h4>Text Evolution Across Checkpoints</h4>
        <div class="checkpoint-texts-container">
    """
    
    for i, cp in enumerate(sorted_checkpoints):
        checkpoint_data = checkpoints[cp]
        text = checkpoint_data['text']
        full_tokens = checkpoint_data['full_tokens']
        
        # Ensure text is a valid string
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        
        # Format full tokens for display
        tokens_display = ""
        if full_tokens and len(full_tokens) > 0:
            tokens_display = f"{full_tokens}"
        else:
            tokens_display = "No tokens available"
        
        # Highlight the checkpoint where answer was found
        is_answer_cp = False
        if hasattr(record, 'answer_checkpoint') and record.answer_checkpoint == cp:
            is_answer_cp = True
        
        checkpoint_class = "checkpoint-text-block"
        if is_answer_cp:
            checkpoint_class += " answer-checkpoint"
        
        # Properly escape the text content for HTML and handle problematic characters
        try:
            # Remove or replace problematic characters that might cause parsing issues
            cleaned_text = text
            # Replace common problematic characters
            cleaned_text = cleaned_text.replace('\x00', '')  # Remove null bytes
            cleaned_text = cleaned_text.replace('\r', '\n')  # Normalize line endings
            # Escape HTML entities
            escaped_text = cleaned_text.replace('<', '&lt;').replace('>', '&gt;').replace('&', '&amp;')
        except Exception as e:
            print(f"Error processing text for checkpoint {cp}: {e}")
            escaped_text = f"[Error processing text: {str(e)}]"
        
        html += f"""
        <div class="{checkpoint_class}">
            <div class="checkpoint-header">
                <h5>Checkpoint {cp}</h5>
                <span class="text-length">{len(text)} characters</span>
            </div>
            <div class="checkpoint-tokens">
                <strong>Generated Tokens:</strong> {tokens_display}
            </div>
            <div class="checkpoint-text">
                <pre>{escaped_text}</pre>
            </div>
        </div>
        """
    
    html += """
        </div>
    </div>
    """
    
    return html

def track_extracted_answer_probability_base64(record):
    """Generate extracted answer probability plot as base64 with correctness markers"""
    try:
        checkpoints = {}
        raw_answer = record.get('data_point', {})['answer']
        # Extract answer from \boxed{} format if present
        try:
            correct_answer = str(raw_answer).split("boxed{")[1].split("}")[0]
        except (IndexError, AttributeError):
            correct_answer = str(raw_answer)
    
        for checkpoint_key, checkpoint_data in record['checkpoint_analysis'].items():
            checkpoint_idx = checkpoint_data['checkpoint_idx']
            extracted_ans = checkpoint_data.get('extracted_generated_answer')
            extracted_ans_prob = checkpoint_data.get('extracted_generated_answer_probability')
            extracted_ans = str(extracted_ans).split("$")[0].split("\n")[0]
        
            if extracted_ans is not None and extracted_ans_prob is not None:
                # Check if this extracted answer is correct
                is_correct = compare_answers(extracted_ans, correct_answer)
                
                checkpoints[checkpoint_idx] = {
                    'probability': extracted_ans_prob,
                    'extracted_answer': extracted_ans,
                    'is_correct': is_correct
                }

        if not checkpoints:
            return None, None

        sorted_checkpoints = sorted(checkpoints.keys())
        probabilities = [checkpoints[cp]['probability'] for cp in sorted_checkpoints]
        extracted_answers = [checkpoints[cp]['extracted_answer'] for cp in sorted_checkpoints]
        correctness = [checkpoints[cp]['is_correct'] for cp in sorted_checkpoints]
        
        # Find first checkpoint where answer becomes correct
        first_correct_checkpoint = None
        for i, (cp, is_correct) in enumerate(zip(sorted_checkpoints, correctness)):
            if is_correct:
                first_correct_checkpoint = cp
                break
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot 1: Extracted Answer Probability Evolution
        ax1.plot(sorted_checkpoints, probabilities, 'o-', linewidth=2, markersize=8, color='blue', alpha=0.7)
        ax1.set_xlabel('Checkpoint Index')
        ax1.set_ylabel('Extracted Answer Probability')
        ax1.set_title(f'Extracted Answer Probability Evolution\nCorrect Answer: {correct_answer}')
        ax1.grid(True, alpha=0.3)

        # Highlight correct answers
        correct_indices = [i for i, is_correct in enumerate(correctness) if is_correct]
        if correct_indices:
            correct_probs = [probabilities[i] for i in correct_indices]
            correct_cps = [sorted_checkpoints[i] for i in correct_indices]
            ax1.scatter(correct_cps, correct_probs, color='green', s=100, zorder=5, 
                       label='Correct Answer', marker='*')
            
            # Mark first correct checkpoint
            if first_correct_checkpoint is not None:
                first_correct_prob = probabilities[sorted_checkpoints.index(first_correct_checkpoint)]
                ax1.annotate(f'FIRST CORRECT\nCP {first_correct_checkpoint}', 
                            xy=(first_correct_checkpoint, first_correct_prob),
                            xytext=(10, 20), textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='lime', alpha=0.8),
                            arrowprops=dict(arrowstyle='->', color='green', lw=2),
                            fontsize=10, fontweight='bold')
                ax1.axvline(x=first_correct_checkpoint, color='green', linestyle='--', alpha=0.7, linewidth=2)

        # Add annotations for max and min probabilities
        max_prob_idx = np.argmax(probabilities)
        min_prob_idx = np.argmin(probabilities)

        ax1.annotate(f'Max: {probabilities[max_prob_idx]:.4f}',
                    xy=(sorted_checkpoints[max_prob_idx], probabilities[max_prob_idx]),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        ax1.annotate(f'Min: {probabilities[min_prob_idx]:.4f}',
                    xy=(sorted_checkpoints[min_prob_idx], probabilities[min_prob_idx]),
                    xytext=(10, -20), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        # Plot 2: Extracted Answers vs Checkpoints
        # Convert extracted answers to numeric values for plotting, with fallback to indices
        try:
            # Try to convert to numeric values
            numeric_answers = []
            for ans in extracted_answers:
                try:
                    # Remove common formatting and convert to float
                    clean_ans = str(ans).replace('$', '').replace(',', '').strip()
                    numeric_answers.append(clean_ans)
                except (ValueError, TypeError):
                    # If conversion fails, use the index as a fallback
                    numeric_answers.append(len(numeric_answers))
            
            ax2.plot(sorted_checkpoints, numeric_answers, 'o-', linewidth=2, markersize=8, color='red', alpha=0.7)
            ax2.set_ylabel('Extracted Answer (Numeric)')
        except Exception:
            # Fallback: plot indices instead of actual values
            ax2.plot(sorted_checkpoints, range(len(sorted_checkpoints)), 'o-', linewidth=2, markersize=8, color='red', alpha=0.7)
            ax2.set_ylabel('Answer Index')
        
        ax2.set_xlabel('Checkpoint Index')
        ax2.set_title('Extracted Answer Evolution')
        ax2.grid(True, alpha=0.3)

        # Highlight correct answers in second plot
        if correct_indices:
            try:
                correct_answers_plot = [numeric_answers[i] for i in correct_indices]
                correct_cps_plot = [sorted_checkpoints[i] for i in correct_indices]
                ax2.scatter(correct_cps_plot, correct_answers_plot, color='green', s=100, zorder=5, 
                           label='Correct Answer', marker='*')
                
                # Mark first correct checkpoint
                if first_correct_checkpoint is not None:
                    first_correct_idx = sorted_checkpoints.index(first_correct_checkpoint)
                    first_correct_ans = extracted_answers[first_correct_idx]  # Use original answer for annotation
                    first_correct_numeric = numeric_answers[first_correct_idx]
                    ax2.annotate(f'FIRST CORRECT\nAnswer: {first_correct_ans}', 
                                xy=(first_correct_checkpoint, first_correct_numeric),
                                xytext=(10, 20), textcoords='offset points',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='lime', alpha=0.8),
                                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                                fontsize=10, fontweight='bold')
                    ax2.axvline(x=first_correct_checkpoint, color='green', linestyle='--', alpha=0.7, linewidth=2)
            except (NameError, IndexError):
                # Fallback if numeric_answers is not available
                correct_cps_plot = [sorted_checkpoints[i] for i in correct_indices]
                correct_indices_plot = [i for i in correct_indices]
                ax2.scatter(correct_cps_plot, correct_indices_plot, color='green', s=100, zorder=5, 
                           label='Correct Answer', marker='*')
                
                # Mark first correct checkpoint
                if first_correct_checkpoint is not None:
                    first_correct_idx = sorted_checkpoints.index(first_correct_checkpoint)
                    first_correct_ans = extracted_answers[first_correct_idx]
                    ax2.annotate(f'FIRST CORRECT\nAnswer: {first_correct_ans}', 
                                xy=(first_correct_checkpoint, first_correct_idx),
                                xytext=(10, 20), textcoords='offset points',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='lime', alpha=0.8),
                                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                                fontsize=10, fontweight='bold')
                    ax2.axvline(x=first_correct_checkpoint, color='green', linestyle='--', alpha=0.7, linewidth=2)

        # Add horizontal line for correct answer
        """
        if correct_answer != 'N/A':
            try:
                correct_ans_float = float(correct_answer)
                ax2.axhline(y=correct_ans_float, color='green', linestyle=':', alpha=0.5, linewidth=2, label=f'Correct: {correct_answer}')
            except (ValueError, TypeError):
                pass
        """
        ax2.legend()

        # Add summary statistics
        if first_correct_checkpoint is not None:
            fig.suptitle(f'Extracted Answer Analysis - First Correct at CP {first_correct_checkpoint}', 
                        fontsize=14, fontweight='bold', color='green')
        else:
            fig.suptitle('Extracted Answer Analysis - No Correct Answer Found', 
                        fontsize=14, fontweight='bold', color='red')
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64, first_correct_checkpoint
        
    except Exception as e:
        print(f"Error in track_extracted_answer_probability_base64: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def filter_correct_answers(records, dataset=""):
    """Filter records to only include those with correct answers."""
    d_ref = None
    ref_lookup = {}
    if dataset == "deepmath":
        deepmath_file = "data/deepmath_103k.jsonl"
        if os.path.exists(deepmath_file):
            d_ref = read_jsonl(deepmath_file)
            ref_lookup = {entry['question']: entry['final_answer'] for entry in d_ref}
    
    correct_records = []
    for record in records:
        try:
            extracted_ans = record['extracted_ans']
            
            if dataset == "deepmath" and ref_lookup:
                question = record['data_point'].get('question', '')
                correct_ans = ref_lookup.get(question)
            else:
                correct_ans = record['data_point']['answer']
                if "boxed{" in str(correct_ans):
                    correct_ans = correct_ans.split("boxed{")[1].split("}")[0]
            

            if extracted_ans is not None and correct_ans is not None:
                try:
                    if str(extracted_ans).strip() == str(correct_ans).strip():
                        correct_records.append(record)
                        continue
                except:
                    pass
                
                try:
                    if str(extracted_ans).split() == str(correct_ans).split():
                        correct_records.append(record)
                        continue
                except:
                    pass
                
                try:
                    if float(extracted_ans) == float(correct_ans):
                        correct_records.append(record)
                        continue
                except:
                    pass
                    
        except Exception as e:
            print(e)
            continue
    
    return correct_records

def load_dataset_data(checkpoint_dir=OUTPUT_DIR):
    """Load all data organized by dataset, then by model"""
    if not os.path.exists(checkpoint_dir):
        print(f"Directory {checkpoint_dir} not found")
        return None
    
    files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.jsonl')]
    datasets_data = {}
    
    print("Loading checkpoint analysis data...")
    
    for filename in sorted(files):
        filepath = os.path.join(checkpoint_dir, filename)
        file_size = os.path.getsize(filepath)
        
        print(f"Processing {filename}...")
        
        dataset = None
        model_name = None
        
        if filename.startswith('checkpoint_analysis_results_'):
            parts = filename.replace('checkpoint_analysis_results_', '').replace('.jsonl', '').split('_')
            
            if 'hendrycks' in filename:
                dataset = 'hendrycks_math'
                model_start_idx = 2
            elif 'deepmath' in filename:
                dataset = 'deepmath'
                model_start_idx = 1
            elif 'amc' in filename:
                dataset = 'amc'
                model_start_idx = 1
            else:
                print(f"  Unknown dataset pattern in {filename}, skipping...")
                continue
            
            model_parts = parts[model_start_idx:]
            model_name = '_'.join(model_parts)
            
            print(f"  Dataset: {dataset}, Model: {model_name}")
            
            try:
                all_records = load_checkpoint_data(filepath)
                correct_records = filter_correct_answers(all_records, dataset)
                print(f"  Found {len(correct_records)} correct answers out of {len(all_records)} total records")
                
                if dataset not in datasets_data:
                    datasets_data[dataset] = {
                        'name': dataset,
                        'display_name': dataset.replace('_', ' ').title(),
                        'models': {},
                        'total_records': 0,
                        'total_correct': 0
                    }
                
                datasets_data[dataset]['models'][model_name] = {
                    'filename': filename,
                    'file_size_mb': round(file_size / (1024*1024), 2),
                    'total_records': len(all_records),
                    'correct_records': len(correct_records),
                    'accuracy': len(correct_records) / len(all_records) * 100 if all_records else 0,
                    'avg_chunks': sum(r.get('num_chunks', 0) for r in all_records) / len(all_records) if all_records else 0,
                    'correct_examples': correct_records
                }
                
                datasets_data[dataset]['total_records'] += len(all_records)
                datasets_data[dataset]['total_correct'] += len(correct_records)
                
            except UnicodeDecodeError as e:
                print(f"Unicode encoding error loading {filename}: {e}")
                continue
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
        else:
            print(f"  Skipping {filename} (doesn't match expected pattern)")
    
    # Calculate dataset-level accuracy
    for dataset_name, dataset_data in datasets_data.items():
        if dataset_data['total_records'] > 0:
            dataset_data['accuracy'] = dataset_data['total_correct'] / dataset_data['total_records'] * 100
        else:
            dataset_data['accuracy'] = 0
    
    return datasets_data

@lru_cache(maxsize=1)
def get_datasets_data():
    """Cached function to load all datasets data"""
    return load_dataset_data()

def create_main_page_html(datasets_data):
    """Create main overview page HTML"""
    
    # Generate dataset sections with model cards
    dataset_sections = ""
    total_datasets = len(datasets_data)
    total_models = sum(len(dataset['models']) for dataset in datasets_data.values())
    
    for dataset_name, dataset_data in datasets_data.items():
        # Generate model cards for this dataset
        model_cards = ""
        for model_name, model_data in dataset_data['models'].items():
            model_cards += f"""
            <div class="model-card" onclick="window.location.href='/model/{dataset_name}/{model_name}'">
                <h4><i class="fas fa-microchip"></i> {model_name.upper()}</h4>
                <div class="stat-row">
                    <span>Total Records:</span>
                    <span>{model_data['total_records']}</span>
                </div>
                <div class="stat-row">
                    <span>Correct Records:</span>
                    <span>{model_data['correct_records']}</span>
                </div>
                <div class="stat-row">
                    <span>Accuracy:</span>
                    <span>{model_data['accuracy']:.1f}%</span>
                </div>
                <div class="stat-row">
                    <span>File Size:</span>
                    <span>{model_data['file_size_mb']} MB</span>
                </div>
                <div class="click-hint">
                    <i class="fas fa-mouse-pointer"></i> Click to analyze
                </div>
            </div>
            """
        
        dataset_sections += f"""
        <div class="dataset-section">
            <div class="dataset-header">
                <h2><i class="fas fa-database"></i> {dataset_data['display_name']}</h2>
                <div class="dataset-stats">
                    <span class="dataset-stat">
                        <strong>{len(dataset_data['models'])}</strong> Models
                    </span>
                    <span class="dataset-stat">
                        <strong>{dataset_data['total_records']}</strong> Total Records
                    </span>
                    <span class="dataset-stat">
                        <strong>{dataset_data['total_correct']}</strong> Correct
                    </span>
                    <span class="dataset-stat">
                        <strong>{dataset_data['accuracy']:.1f}%</strong> Accuracy
                    </span>
                </div>
            </div>
            <div class="models-grid">
                {model_cards}
            </div>
        </div>
        """
    
    # Generate comparison table
    comparison_rows = ""
    for dataset_name, dataset_data in datasets_data.items():
        for model_name, model_data in dataset_data['models'].items():
            comparison_rows += f"""
            <tr onclick="window.location.href='/model/{dataset_name}/{model_name}'" style="cursor: pointer;">
                <td>{dataset_data['display_name']}</td>
                <td>{model_name.upper()}</td>
                <td>{model_data['total_records']}</td>
                <td>{model_data['correct_records']}</td>
                <td>{model_data['accuracy']:.1f}%</td>
                <td>{model_data['file_size_mb']} MB</td>
                <td>N/A</td>
            </tr>
            """
    
    total_records = sum(dataset['total_records'] for dataset in datasets_data.values())
    total_correct = sum(dataset['total_correct'] for dataset in datasets_data.values())
    overall_accuracy = (total_correct / total_records * 100) if total_records > 0 else 0
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Checkpoint Analysis Dashboard</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }}
        
        .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
        
        .header {{
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }}
        
        .header h1 {{
            font-size: 3rem;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        
        .stats-overview {{
            background-color: rgba(255,255,255,0.1);
            color: white;
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 30px;
            text-align: center;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 25px;
            margin-top: 20px;
        }}
        
        .stat-item {{
            padding: 20px;
            background-color: rgba(255,255,255,0.1);
            border-radius: 10px;
        }}
        
        .stat-value {{
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 8px;
        }}
        
        .stat-label {{
            font-size: 1rem;
            opacity: 0.9;
        }}
        
        .dataset-section {{
            background-color: white;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 30px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        }}
        
        .dataset-header {{
            margin-bottom: 25px;
            border-bottom: 2px solid #f1f3f5;
            padding-bottom: 20px;
        }}
        
        .dataset-header h2 {{
            color: #495057;
            margin-bottom: 15px;
            font-size: 1.8rem;
        }}
        
        .dataset-stats {{
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }}
        
        .dataset-stat {{
            padding: 8px 15px;
            background-color: #f8f9fa;
            border-radius: 20px;
            font-size: 0.9rem;
            color: #495057;
        }}
        
        .models-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }}
        
        .model-card {{
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 20px;
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        }}
        
        .model-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.25);
        }}
        
        .model-card h4 {{
            font-size: 1.4rem;
            margin-bottom: 15px;
            text-align: center;
        }}
        
        .stat-row {{
            display: flex;
            justify-content: space-between;
            margin: 12px 0;
            font-size: 1.1rem;
        }}
        
        .click-hint {{
            text-align: center;
            margin-top: 20px;
            font-size: 0.95rem;
            opacity: 0.8;
        }}
        
        .comparison-section {{
            background-color: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        }}
        
        .comparison-section h2 {{
            color: #333;
            margin-bottom: 20px;
            font-size: 1.8rem;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        
        th {{
            background-color: #f8f9fa;
            padding: 15px;
            text-align: left;
            border-bottom: 2px solid #dee2e6;
            font-weight: bold;
        }}
        
        td {{
            padding: 15px;
            border-bottom: 1px solid #dee2e6;
        }}
        
        tr:hover {{
            background-color: #f8f9fa;
        }}
        
        .intro-section {{
            background-color: rgba(255,255,255,0.1);
            color: white;
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 30px;
            text-align: center;
        }}
        
        @media (max-width: 768px) {{
            .container {{ padding: 15px; }}
            .header h1 {{ font-size: 2.2rem; }}
            .models-grid {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1><i class="fas fa-chart-line"></i> Checkpoint Analysis Dashboard</h1>
            <p>Interactive multi-model analysis dashboard</p>
        </div>

        <div class="intro-section">
            <h3><i class="fas fa-info-circle"></i> Welcome to the Dashboard</h3>
            <p>This dashboard analyzes model performance across multiple mathematical datasets. Each dataset section shows different models tested on that dataset. Click on any model card to explore detailed analysis of correct answers, including visualizations and checkpoint text evolution.</p>
        </div>

        <div class="stats-overview">
            <h2><i class="fas fa-chart-bar"></i> Overall Statistics</h2>
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-value">{total_datasets}</div>
                    <div class="stat-label">Datasets</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{total_models}</div>
                    <div class="stat-label">Models Analyzed</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{total_records}</div>
                    <div class="stat-label">Total Records</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{total_correct}</div>
                    <div class="stat-label">Correct Answers</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{overall_accuracy:.1f}%</div>
                    <div class="stat-label">Overall Accuracy</div>
                </div>
            </div>
        </div>

        {dataset_sections}

        <div class="comparison-section">
            <h2><i class="fas fa-balance-scale"></i> Model Comparison</h2>
            <p>Click on any row to view detailed analysis for that model.</p>
            <table>
                <thead>
                    <tr>
                        <th>Dataset</th>
                        <th>Model</th>
                        <th>Total Records</th>
                        <th>Correct Records</th>
                        <th>Accuracy</th>
                        <th>File Size</th>
                        <th>Avg Top Position</th>
                    </tr>
                </thead>
                <tbody>
                    {comparison_rows}
                </tbody>
            </table>
        </div>
    </div>

    <script>
        console.log('Main dashboard loaded');
        
        // Add click animations
        document.querySelectorAll('.model-card').forEach(card => {{
            card.addEventListener('click', function() {{
                this.style.transform = 'scale(0.95)';
                setTimeout(() => {{
                    this.style.transform = '';
                }}, 150);
            }});
        }});
    </script>
</body>
</html>"""

    return html_content

def create_model_page_html(dataset_name, model_name, model_data):
    """Create individual model analysis page HTML"""
    
    examples = model_data['correct_examples']
    
    # Handle case where there are no correct examples
    if not examples:
        return create_empty_model_page_html(dataset_name, model_name, model_data)
    
    print(f"Creating model page for {len(examples)} examples in {model_name}...")
    
    # Store examples data for AJAX loading (no visualizations generated yet)
    examples_data = []
    for i, record in enumerate(examples):
        # Calculate basic metadata without generating visualizations
        answer_checkpoint, _ = compute_answer_top_prediction_checkpoint(record)
        
        # Extract basic info for display
        question = record['data_point'].get('question', record['data_point'].get('problem', 'Question not available'))
        correct_answer = record.get('data_point', {}).get('answer', 'N/A')
        extracted_answer = record.get('extracted_ans', 'N/A')
        
        examples_data.append({
            'index': i,
            'question': question,
            'correct_answer': correct_answer,
            'extracted_answer': extracted_answer,
            'answer_checkpoint': answer_checkpoint,
            'record_id': id(record)  # Use record ID for AJAX requests
        })

    # Create example sections (visualizations loaded on demand)
    examples_html = ""
    for example_data in examples_data:
        i = example_data['index']
        question = example_data['question']
        correct_answer = example_data['correct_answer']
        extracted_answer = example_data['extracted_answer']
        answer_checkpoint = example_data['answer_checkpoint']
        record_id = example_data['record_id']

        examples_html += f"""
        <div class="example-section" data-record-id="{record_id}" data-example-index="{i}">
            <div class="example-header">
                <h3>Example #{i+1}</h3>
                <div class="example-metadata">
                    <span class="metadata-item">Answer Checkpoint: {answer_checkpoint if answer_checkpoint != 'N/A' else 'Not found'}</span>
                    <span class="metadata-item">Correct: {correct_answer}</span>
                    <span class="metadata-item">Extracted: {extracted_answer}</span>
                </div>
                <button class="load-viz-btn" onclick="loadVisualizations({record_id}, {i})">
                    <i class="fas fa-chart-line"></i> Load Analysis
                </button>
            </div>
            
            <div class="example-content">
                <div class="question-section">
                    <h4>Question:</h4>
                    <p class="question-text">{question}</p>
                </div>
                
                <div class="visualizations-section" id="viz-section-{record_id}">
                    <div class="loading-placeholder">
                        <i class="fas fa-spinner fa-spin"></i>
                        <p>Click "Load Analysis" to generate visualizations</p>
                    </div>
                </div>
            </div>
        </div>
        """
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{dataset_name.replace('_', ' ').title()} - {model_name.upper()} Analysis</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }}
        
        .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
        
        .header {{
            background-color: rgba(255,255,255,0.1);
            color: white;
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 25px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .header h1 {{ font-size: 2.2rem; }}
        
        .back-btn {{
            background-color: rgba(255,255,255,0.2);
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1rem;
            transition: all 0.3s ease;
            text-decoration: none;
        }}
        
        .back-btn:hover {{ background-color: rgba(255,255,255,0.3); }}
        
        .model-stats {{
            background-color: white;
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 25px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        
        .stat-item {{
            text-align: center;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 10px;
        }}
        
        .stat-value {{
            font-size: 1.8rem;
            font-weight: bold;
            color: #007bff;
            margin-bottom: 8px;
        }}
        
        .examples-container {{
            background-color: white;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        }}
        
        .example-section {{
            border: 1px solid #e9ecef;
            border-radius: 12px;
            margin-bottom: 30px;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        
        .example-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid #dee2e6;
        }}
        
        .example-header h3 {{
            color: #495057;
            font-size: 1.4rem;
        }}
        
        .example-metadata {{
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }}
        
        .metadata-item {{
            background-color: #e9ecef;
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 0.9rem;
            color: #495057;
        }}
        
        .load-viz-btn {{
            background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.9rem;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .load-viz-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(40, 167, 69, 0.3);
        }}
        
        .load-viz-btn:disabled {{
            background: #6c757d;
            cursor: not-allowed;
            transform: none;
        }}
        
        .loading-placeholder {{
            text-align: center;
            padding: 40px;
            color: #6c757d;
            background-color: #f8f9fa;
            border-radius: 8px;
            border: 2px dashed #dee2e6;
        }}
        
        .loading-placeholder i {{
            font-size: 2rem;
            margin-bottom: 15px;
            color: #007bff;
        }}
        
        .loading-placeholder p {{
            font-size: 1.1rem;
            margin: 0;
        }}
        
        .viz-loading {{
            text-align: center;
            padding: 20px;
            color: #007bff;
        }}
        
        .viz-loading i {{
            font-size: 1.5rem;
            margin-right: 10px;
        }}
        
        .error-message {{
            text-align: center;
            padding: 20px;
            color: #dc3545;
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            border-radius: 8px;
        }}
        
        .error-message i {{
            font-size: 1.5rem;
            margin-right: 10px;
        }}
        
        .text-evolution-section {{
            margin-top: 30px;
            background-color: #f8f9fa;
            border-radius: 12px;
            padding: 20px;
            border: 1px solid #dee2e6;
        }}
        
        .text-evolution-section h4 {{
            color: #495057;
            margin-bottom: 20px;
            font-size: 1.3rem;
        }}
        
        .checkpoint-texts-container {{
            display: flex;
            flex-direction: column;
            gap: 15px;
            max-height: 600px;
            overflow-y: auto;
        }}
        
        .checkpoint-text-block {{
            background-color: white;
            border: 2px solid #dee2e6;
            border-radius: 8px;
            padding: 15px;
            transition: all 0.3s ease;
        }}
        
        .checkpoint-text-block:hover {{
            border-color: #007bff;
            box-shadow: 0 4px 12px rgba(0,123,255,0.1);
        }}
        
        .checkpoint-text-block.answer-checkpoint {{
            border-color: #28a745;
            background-color: #d4edda;
        }}
        
        .checkpoint-text-block.answer-checkpoint:hover {{
            border-color: #28a745;
            box-shadow: 0 4px 12px rgba(40,167,69,0.15);
        }}
        
        .checkpoint-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #dee2e6;
        }}
        
        .checkpoint-header h5 {{
            color: #007bff;
            margin: 0;
            font-size: 1.2rem;
        }}
        
        .text-length {{
            background-color: #e9ecef;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.8rem;
            color: #495057;
        }}
        
        .checkpoint-tokens {{
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            padding: 10px;
            margin-bottom: 15px;
            font-size: 0.9rem;
            color: #495057;
        }}
        
        .checkpoint-tokens strong {{
            color: #007bff;
        }}
        
        .checkpoint-text {{
            background-color: #f8f9fa;
            border-radius: 6px;
            padding: 15px;
            border-left: 4px solid #007bff;
        }}
        
        .checkpoint-text pre {{
            margin: 0;
            white-space: pre-wrap;
            word-wrap: break-word;
            font-family: 'Courier New', monospace;
            font-size: 0.9rem;
            line-height: 1.4;
            color: #495057;
        }}
        
        .no-data {{
            text-align: center;
            color: #6c757d;
            font-style: italic;
            padding: 20px;
        }}
        
        .example-content {{
            background-color: white;
            border-radius: 8px;
            padding: 20px;
        }}
        
        .question-section {{
            margin-bottom: 25px;
        }}
        
        .question-section h4 {{
            color: #007bff;
            margin-bottom: 10px;
        }}
        
        .question-text {{
            font-size: 1.1rem;
            line-height: 1.6;
            color: #495057;
            font-style: italic;
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #007bff;
        }}
        
        .visualizations-section h4 {{
            color: #28a745;
            margin-bottom: 15px;
        }}
        
        .viz-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        
        .viz-item {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid #dee2e6;
        }}
        
        .viz-item h5 {{
            margin-bottom: 12px;
            color: #495057;
            font-size: 1.1rem;
        }}
        
        .viz-image {{
            width: 100%;
            max-width: 100%;
            height: auto;
            border-radius: 6px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .summary-section {{
            background-color: rgba(255,255,255,0.1);
            color: white;
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 25px;
            text-align: center;
        }}
        
        @media (max-width: 768px) {{
            .container {{ padding: 15px; }}
            .header {{ flex-direction: column; gap: 15px; text-align: center; }}
            .viz-grid {{ grid-template-columns: 1fr; }}
            .example-header {{ flex-direction: column; gap: 10px; }}
            .example-metadata {{ justify-content: center; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1><i class="fas fa-database"></i> {dataset_name.replace('_', ' ').title()} - <i class="fas fa-microchip"></i> {model_name.upper()} Analysis</h1>
            <a href="/" class="back-btn">
                <i class="fas fa-arrow-left"></i> Back to Overview
            </a>
        </div>

        <div class="model-stats">
            <h2><i class="fas fa-chart-bar"></i> Model Performance</h2>
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-value">{model_data['total_records']}</div>
                    <div class="stat-label">Total Records</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{model_data['correct_records']}</div>
                    <div class="stat-label">Correct Answers</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{model_data['accuracy']:.1f}%</div>
                    <div class="stat-label">Accuracy</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{model_data['file_size_mb']} MB</div>
                    <div class="stat-label">Data Size</div>
                </div>
            </div>
        </div>

        <div class="summary-section">
            <h2><i class="fas fa-info-circle"></i> Analysis Summary</h2>
            <p>Showing {len(examples)} correct examples. Click "Load Analysis" on any example to generate detailed visualizations including top predictions evolution and extracted answer probability evolution across checkpoints.</p>
        </div>

        <div class="examples-container">
            <h2><i class="fas fa-list"></i> All Correct Examples Analysis</h2>
            <p style="margin-bottom: 25px; color: #666;">Detailed analysis of each correct example with comprehensive visualizations.</p>
            
            {examples_html}
        </div>
    </div>

    <script>
        console.log('Model analysis page loaded for {model_name} with {len(examples)} examples');
        
        // Add smooth scrolling for better UX
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {{
            anchor.addEventListener('click', function (e) {{
                e.preventDefault();
                document.querySelector(this.getAttribute('href')).scrollIntoView({{
                    behavior: 'smooth'
                }});
            }});
        }});
        
        // Function to load visualizations via AJAX
        function loadVisualizations(recordId, exampleIndex) {{
            const vizSection = document.getElementById(`viz-section-${{recordId}}`);
            const loadBtn = event.target.closest('.load-viz-btn');
            
            // Show loading state
            vizSection.innerHTML = `
                <div class="viz-loading">
                    <i class="fas fa-spinner fa-spin"></i>
                    Generating visualizations...
                </div>
            `;
            loadBtn.disabled = true;
            loadBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading...';
            
            // Make AJAX request
            fetch(`/api/visualizations/${{recordId}}`)
                .then(response => response.json())
                .then(data => {{
                    if (data.success) {{
                        vizSection.innerHTML = data.html;
                        loadBtn.innerHTML = '<i class="fas fa-check"></i> Loaded';
                        loadBtn.style.background = 'linear-gradient(135deg, #28a745 0%, #20c997 100%)';
                    }} else {{
                        vizSection.innerHTML = `
                            <div class="error-message">
                                <i class="fas fa-exclamation-triangle"></i>
                                Error loading visualizations: ${{data.error}}
                            </div>
                        `;
                        loadBtn.disabled = false;
                        loadBtn.innerHTML = '<i class="fas fa-chart-line"></i> Retry';
                    }}
                }})
                .catch(error => {{
                    vizSection.innerHTML = `
                        <div class="error-message">
                            <i class="fas fa-exclamation-triangle"></i>
                            Network error: ${{error.message}}
                        </div>
                    `;
                    loadBtn.disabled = false;
                    loadBtn.innerHTML = '<i class="fas fa-chart-line"></i> Retry';
                }});
        }}
    </script>
</body>
</html>"""

    return html_content

def create_empty_model_page_html(dataset_name, model_name, model_data):
    """Create a page for models with no correct examples"""
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{dataset_name.replace('_', ' ').title()} - {model_name.upper()} Analysis</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }}
        
        .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
        
        .header {{
            background-color: rgba(255,255,255,0.1);
            color: white;
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 25px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .header h1 {{ font-size: 2.2rem; }}
        
        .back-btn {{
            background-color: rgba(255,255,255,0.2);
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1rem;
            transition: all 0.3s ease;
            text-decoration: none;
        }}
        
        .back-btn:hover {{ background-color: rgba(255,255,255,0.3); }}
        
        .no-examples-section {{
            background-color: white;
            padding: 50px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        }}
        
        .no-examples-icon {{
            font-size: 4rem;
            color: #dc3545;
            margin-bottom: 20px;
        }}
        
        .no-examples-title {{
            font-size: 2rem;
            color: #495057;
            margin-bottom: 15px;
        }}
        
        .no-examples-text {{
            font-size: 1.1rem;
            color: #6c757d;
            line-height: 1.6;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1><i class="fas fa-database"></i> {dataset_name.replace('_', ' ').title()} - <i class="fas fa-microchip"></i> {model_name.upper()} Analysis</h1>
            <a href="/" class="back-btn">
                <i class="fas fa-arrow-left"></i> Back to Overview
            </a>
        </div>

        <div class="no-examples-section">
            <div class="no-examples-icon">
                <i class="fas fa-exclamation-triangle"></i>
            </div>
            <div class="no-examples-title">No Correct Examples Found</div>
            <div class="no-examples-text">
                This model has no correct answers in the dataset, so no detailed analysis can be performed.<br>
                The model achieved {model_data['accuracy']:.1f}% accuracy on {model_data['total_records']} total records.
            </div>
        </div>
    </div>
</body>
</html>"""
    
    return html_content

@app.route('/')
def index():
    """Main overview page"""
    try:
        datasets_data = get_datasets_data()
        if not datasets_data:
            return "No data found. Please ensure checkpoint files exist in output/ckpt/", 404
        
        return create_main_page_html(datasets_data)
    except Exception as e:
        return f"Error loading data: {str(e)}", 500

@app.route('/model/<dataset_name>/<model_name>')
def model_analysis(dataset_name, model_name):
    """Individual model analysis page"""
    try:
        datasets_data = get_datasets_data()
        if not datasets_data or dataset_name not in datasets_data:
            return f"Dataset {dataset_name} not found", 404
        
        if model_name not in datasets_data[dataset_name]['models']:
            return f"Model {model_name} not found in dataset {dataset_name}", 404
        
        model_data = datasets_data[dataset_name]['models'][model_name]
        
        # Store examples in global cache for AJAX access
        examples = model_data['correct_examples']
        for record in examples:
            examples_cache[id(record)] = record
        
        return create_model_page_html(dataset_name, model_name, model_data)
    except Exception as e:
        return f"Error loading model page: {str(e)}", 500

@app.route('/api/visualizations/<int:record_id>')
def get_visualizations(record_id):
    """AJAX endpoint to generate visualizations for a specific record"""
    try:
        if record_id not in examples_cache:
            return jsonify({'success': False, 'error': 'Record not found'}), 404
        
        record = examples_cache[record_id]
        
        # Generate visualizations with error handling
        viz1_img, answer_checkpoint = None, None
        viz4_img, first_correct_cp = None, None
        
        try:
            viz1_img, answer_checkpoint = track_top_predictions_base64(record)
        except Exception as viz1_error:
            print(f"Error generating top predictions viz: {viz1_error}")
            import traceback
            traceback.print_exc()
        
        try:
            viz4_img, first_correct_cp = track_extracted_answer_probability_base64(record)
        except Exception as viz4_error:
            print(f"Error generating extracted answer viz: {viz4_error}")
            import traceback
            traceback.print_exc()
        
        # Generate text evolution section
        try:
            text_evolution_html = generate_text_evolution_html(record)
        except Exception as text_error:
            print(f"Error generating text evolution: {text_error}")
            import traceback
            traceback.print_exc()
            text_evolution_html = """
            <div class="text-evolution-section">
                <h4>Text Evolution Across Checkpoints</h4>
                <p class="no-data">Error loading text data: """ + str(text_error) + """</p>
            </div>
            """
        
        # Create HTML for visualizations
        viz_html = f"""
        <h4>Analysis Visualizations:</h4>
        <div class="viz-grid">
        """
        
        if viz1_img:
            viz_html += f"""
            <div class="viz-item">
                <h5>Top Predictions Evolution</h5>
                <img src="data:image/png;base64,{viz1_img}" class="viz-image" alt="Top Predictions">
            </div>
            """
        else:
            viz_html += """
            <div class="viz-item">
                <h5>Top Predictions Evolution</h5>
                <p class="no-data">Error generating visualization</p>
            </div>
            """
        
        if viz4_img:
            viz_html += f"""
            <div class="viz-item">
                <h5>Extracted Answer Probability Evolution</h5>
                <img src="data:image/png;base64,{viz4_img}" class="viz-image" alt="Extracted Answer Probability">
            </div>
            """
        else:
            viz_html += """
            <div class="viz-item">
                <h5>Extracted Answer Probability Evolution</h5>
                <p class="no-data">Error generating visualization</p>
            </div>
            """
        
        viz_html += """
        </div>
        """
        
        # Add text evolution section
        viz_html += text_evolution_html
        
        return jsonify({
            'success': True,
            'html': viz_html,
            'answer_checkpoint': answer_checkpoint,
            'first_correct_checkpoint': first_correct_cp
        })
        
    except Exception as e:
        print(f"Error in get_visualizations: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask Checkpoint Analysis Dashboard!")
    print("Loading data from output/ckpt directory...")
    print("Open your browser to http://localhost:5000 to view the dashboard")
    
    # Preload data on startup
    try:
        datasets_data = get_datasets_data()
        if datasets_data:
            total_datasets = len(datasets_data)
            total_models = sum(len(dataset['models']) for dataset in datasets_data.values())
            total_records = sum(dataset['total_records'] for dataset in datasets_data.values())
            total_correct = sum(dataset['total_correct'] for dataset in datasets_data.values())
            print(f"Loaded {total_datasets} datasets, {total_models} models, {total_records} total records, {total_correct} correct answers")
        else:
            print("Warning: No data found in output/ckpt directory")
    except Exception as e:
        print(f"Warning: Error preloading data: {e}")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000) 
