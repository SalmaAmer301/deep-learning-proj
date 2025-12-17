!pip install gradio pandas openai


import os
import json
from datetime import datetime
import pandas as pd
import gradio as gr
import openai


openai.api_key = "YOUR_OPENAI_API_KEY"


dataset_path = "arabic_edu_dataset.csv"

data = [
    ("ما هو التعليم؟","التعليم هو عملية نقل المعرفة والمهارات والقيم من جيل إلى آخر."),
    ("ما هو الذكاء الاصطناعي؟","الذكاء الاصطناعي هو فرع من علوم الحاسوب يهدف إلى إنشاء أنظمة تحاكي الذكاء البشري."),
    ("ما هو التعلم الآلي؟","التعلم الآلي هو أحد فروع الذكاء الاصطناعي يعتمد على البيانات لتعلم الأنماط."),
    ("ما هو التعلم العميق؟","التعلم العميق هو نوع من التعلم الآلي يعتمد على الشبكات العصبية."),
    ("ما هي الشبكة العصبية؟","الشبكة العصبية هي نموذج رياضي مستوحى من الدماغ البشري."),
    ("ما هو الحاسوب؟","الحاسوب جهاز إلكتروني لمعالجة البيانات."),
    ("ما هي البرمجة؟","البرمجة هي كتابة أوامر للحاسوب."),
    ("ما هو بايثون؟","بايثون لغة سهلة وتستخدم في الذكاء الاصطناعي."),
    ("ما هو Gradio؟","مكتبة لبناء واجهات لتطبيقات الذكاء الاصطناعي."),
    ("ما هي عاصمة مصر؟","القاهرة هي عاصمة مصر.")
]

df = pd.DataFrame(data, columns=["question","answer"])
df.to_csv(dataset_path, index=False, encoding="utf-8-sig")


df = pd.read_csv(dataset_path)


LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)



def search_dataset(question):
    match = df[df["question"].str.contains(question, case=False, na=False)]
    if not match.empty:
        return match.iloc[0]["answer"]
    return None

def call_llm(question):
    response = openai.ChatCompletion.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": question}],
        temperature=0.5
    )
    return response["choices"][0]["message"]["content"]

def save_history(username, question, answer):
    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    record = {
        "username": username,
        "question": question,
        "answer": answer,
        "time": time_now
    }

    csv_path = f"{LOG_DIR}/{username}_chat.csv"
    json_path = f"{LOG_DIR}/{username}_chat.json"

    # CSV
    pd.DataFrame([record]).to_csv(
        csv_path,
        mode="a",
        header=not os.path.exists(csv_path),
        index=False,
        encoding="utf-8-sig"
    )

    # JSON
    history = []
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            history = json.load(f)

    history.append(record)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def assistant(username, question):
    if not username.strip():
        return " أدخل اسم المستخدم", ""

    answer = search_dataset(question)
    if answer is None:
        answer = call_llm(question)

    save_history(username, question, answer)
    return answer, " تم حفظ السؤال مع التاريخ والوقت"

def clear_history(username):
    for ext in ["csv","json"]:
        path = f"{LOG_DIR}/{username}_chat.{ext}"
        if os.path.exists(path):
            os.remove(path)
    return " تم حذف سجل المستخدم"

def download_csv(username):
    path = f"{LOG_DIR}/{username}_chat.csv"
    return path if os.path.exists(path) else None

def download_json(username):
    path = f"{LOG_DIR}/{username}_chat.json"
    return path if os.path.exists(path) else None


css = """
body { background: #f5f3ff; }
h1 { color: #6a0dad; text-align:center; }
button { border-radius: 12px; font-size: 16px; }
"""


with gr.Blocks(css=css) as app:

    gr.Markdown("##  LLM-Based Arabic Education Assistant")

    username = gr.Textbox(label=" اسم المستخدم")
    question = gr.Textbox(label="? السؤال", lines=2)

    answer = gr.Textbox(label=" الإجابة")
    status = gr.Textbox(label=" الحالة")

    with gr.Row():
        ask_btn = gr.Button(" اسأل")
        clear_btn = gr.Button(" Clear History")

    with gr.Row():
        csv_btn = gr.Button("⬇ Download CSV")
        json_btn = gr.Button("⬇ Download JSON")

    file_out = gr.File()

    ask_btn.click(assistant, [username, question], [answer, status])
    clear_btn.click(clear_history, username, status)
    csv_btn.click(download_csv, username, file_out)
    json_btn.click(download_json, username, file_out)

app.launch()
