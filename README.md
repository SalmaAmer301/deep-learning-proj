import os
import json
from datetime import datetime
import pandas as pd
import gradio as gr
from openai import OpenAI

#  OpenAI Client ----------
client = OpenAI(api_key="") 

#  Logs Folder ----------
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

#  Default Dataset ----------
DEFAULT_DATA = [
    ("ما هو التعليم؟","التعليم هو عملية نقل المعرفة والمهارات والقيم."),
    ("ما هو الذكاء الاصطناعي؟","فرع من علوم الحاسوب يحاكي الذكاء البشري."),
    ("ما هو التعلم الآلي؟","أحد فروع الذكاء الاصطناعي يعتمد على البيانات."),
    ("ما هو Gradio؟","مكتبة لبناء واجهات لتطبيقات الذكاء الاصطناعي."),
    ("ما هي عاصمة مصر؟","القاهرة هي عاصمة مصر.")
]

df = pd.DataFrame(DEFAULT_DATA, columns=["question","answer"])

# Backend Functions ----------
def load_uploaded_dataset(file):
    global df
    if file:
        df = pd.read_csv(file.name)

def search_dataset(question):
    match = df[df["question"].str.contains(question, case=False, na=False)]
    return match.iloc[0]["answer"] if not match.empty else None

def generate_gpt_answer(question):
    prompt = f"أجب باللغة العربية بإيجاز على السؤال التالي: {question}"
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=150
        )
        answer = response.choices[0].message.content
        return answer
    except Exception as e:
        return " Error calling GPT: " + str(e)

def save_history(username, question, answer):
    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    record = {"user": username, "question": question, "answer": answer, "time": time_now}

    csv_path = f"{LOG_DIR}/{username}_chat.csv"
    json_path = f"{LOG_DIR}/{username}_chat.json"

    # Save CSV
    pd.DataFrame([record]).to_csv(
        csv_path,
        mode="a",
        header=not os.path.exists(csv_path),
        index=False,
        encoding="utf-8-sig"
    )

    # Save JSON
    history = []
    if os.path.exists(json_path):
        history = json.load(open(json_path, encoding="utf-8"))
    history.append(record)
    json.dump(history, open(json_path,"w",encoding="utf-8"),
              ensure_ascii=False, indent=2)

# Main Assistant Function ----------
def assistant(username, question):
    if not username.strip():
        return " Enter username", ""

    # 1. Search dataset
    answer = search_dataset(question)

    # 2. If not in dataset, call GPT
    if not answer:
        answer = generate_gpt_answer(question)

    # 3. Save history
    save_history(username, question, answer)

    # 4. Return HTML with avatar
    avatar_html = f"""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">
        <img src="https://i.ibb.co/9y0n0m6/avatar.png" width="50px" style="border-radius:50%">
        <div style="background:#ede9fe; padding:10px; border-radius:12px;">
            {answer}
        </div>
    </div>
    """
    return avatar_html, " Question saved with timestamp"

#  Clear History ----------
def clear_history(username):
    for ext in ["csv","json"]:
        path = f"{LOG_DIR}/{username}_chat.{ext}"
        if os.path.exists(path):
            os.remove(path)
    return " Chat history cleared"

#  Download History ----------
def download_csv(username):
    path = f"{LOG_DIR}/{username}_chat.csv"
    return path if os.path.exists(path) else None

def download_json(username):
    path = f"{LOG_DIR}/{username}_chat.json"
    return path if os.path.exists(path) else None

# CSS ----------
css = """
body {background: linear-gradient(to right, #ede9fe, #f5f3ff); font-family:Arial;}
h1,h2 {color:#6a0dad;text-align:center;}
.gr-button {border-radius:14px !important;font-size:16px !important;}
textarea {border-radius:12px !important;}
"""

# Gradio Frontend ----------
with gr.Blocks(css=css) as app:

    gr.Markdown("##  LLM-Based Arabic Education Assistant with GPT ")
    gr.Markdown("### Smart Arabic Educational Chatbot")

    with gr.Row():
        username = gr.Textbox(label=" Username")
        dataset_file = gr.File(label=" Upload Dataset (CSV)")

    dataset_file.change(load_uploaded_dataset, dataset_file, None)

    question = gr.Textbox(label=" Enter your question", lines=2)
    answer_html = gr.HTML(label=" Answer")
    status = gr.Textbox(label=" Status")

    with gr.Row():
        ask_btn = gr.Button(" Ask")
        clear_btn = gr.Button(" Clear History")

    with gr.Row():
        csv_btn = gr.Button("⬇Download CSV")
        json_btn = gr.Button("⬇ Download JSON")

    file_out = gr.File()

    ask_btn.click(assistant, [username, question], [answer_html, status])
    clear_btn.click(clear_history, username, status)
    csv_btn.click(download_csv, username, file_out)
    json_btn.click(download_json, username, file_out)

# Launch the app
app.launch(share=True)  

    
