from flask import Flask, render_template, request
import json
import re
import requests
from pathlib import Path

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent


def validate_article_ids(articles):
    actual_ids = [article.get("id") for article in articles]
    if len(actual_ids) != len(set(actual_ids)):
        raise ValueError("Data.json contains duplicate article IDs")

    invalid_ids = [
        article_id for article_id in actual_ids
        if not isinstance(article_id, str) or not article_id.startswith("AL_RAG_")
    ]
    if invalid_ids:
        raise ValueError("Data.json contains invalid AL_RAG article IDs")


def get_relevant_articles(articles, user_prompt, limit=5):
    query_terms = set(user_prompt.lower().split())

    def relevance(article):
        searchable_text = json.dumps(article, ensure_ascii=False).lower()
        return sum(term in searchable_text for term in query_terms)

    ranked_articles = sorted(
        articles,
        key=lambda article: relevance(article),
        reverse=True,
    )
    return ranked_articles[:limit]


def format_article(article):
    metrics = article.get("financial_metrics", {})
    metrics_text = ", ".join(
        f"{key.replace('_', ' ')}: {value}"
        for key, value in metrics.items()
    )
    tags = ", ".join(article.get("tags", []))

    return (
        f"{article.get('company')} ({article.get('ticker')}) | "
        f"{article.get('event_type')} | {article.get('sentiment')}\n"
        f"Headline: {article.get('headline')}\n"
        f"Summary: {article.get('summary')}\n"
        f"Location: {article.get('location')} | "
        f"Sector: {article.get('sector')} | Industry: {article.get('industry')} | "
        f"Country: {article.get('country')}\n"
        f"Financial metrics: {metrics_text}\n"
        f"Tags: {tags}"
    )


def clean_ai_response(response_text):
    warning_lines = (
        "Please do not enter your info to user",
        "Please do not enter your info to the user",
        "Please do not share your personal information",
    )

    cleaned_lines = []
    for line in response_text.splitlines():
        if not any(warning.lower() in line.lower() for warning in warning_lines):
            cleaned_lines.append(line)
    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\[AL_RAG_[^\]]*\]", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bAL_RAG_[A-Za-z0-9_-]+\b", "", cleaned, flags=re.IGNORECASE)
    return re.sub(r"[ \t]+", " ", cleaned).strip()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    user_prompt = request.form.get("prompt", "").strip()
    if not user_prompt:
        return render_template(
            'index.html',
            prompt=user_prompt,
            error="Please enter a question before sending.",
        ), 400

    try:
        with (BASE_DIR / "Data.json").open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        app.logger.error("Unable to load the AlphaLens dataset: %s", exc)
        return render_template(
            'index.html',
            prompt=user_prompt,
            error="The financial knowledge base is unavailable right now.",
        ), 503

    metadata = data["My Data Base"]
    articles = data["articles"]
    try:
        validate_article_ids(articles)
    except ValueError as exc:
        app.logger.error("Invalid AlphaLens dataset: %s", exc)
        return render_template(
            'index.html',
            prompt=user_prompt,
            error="The financial knowledge base is unavailable right now.",
        ), 503
    relevant_articles = get_relevant_articles(articles, user_prompt)
    developer = data["developer_details"]
    article_context = "\n\n".join(format_article(article) for article in relevant_articles)

    prompt = f"""
    You are a Large AI model, in any response you don't have to provide your personal information or any other sensitive information. You are a financial news assistant that provides accurate and relevant information based on the provided dataset. You should answer the user's question using only the information from the dataset and relevant financial news records. If the answer is not present in the dataset, you should respond with "I don't know" or "The answer is not available in the provided dataset." Do not make up any information or provide answers that are not supported by the dataset.
    Never mention record IDs, article IDs, internal identifiers, dataset codes, or values beginning with AL_RAG_. Speak naturally and cite only company names, dates, and financial information. And also do not mention the dataset or something related to that, and also you have to response in first person which is you.

    Information:
    Dataset: {metadata["project"]}
    Version: {metadata["version"]}
    Schema: {metadata["schema"]}
    Available records: {metadata["total_records_in_batch"]}

    Relevant financial news records:
    {article_context}

    User:
    {user_prompt}
    """

    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'tinyllama',
                'prompt': prompt,
                'stream': False
            },
            timeout=120
        )
        response.raise_for_status()
        ai_response = clean_ai_response(response.json()['response'])
    except requests.Timeout:
        app.logger.error("The AlphaLens model timed out")
        return render_template(
            'index.html',
            prompt=user_prompt,
            error="The model took too long to respond. Please try again.",
        ), 504
    except requests.ConnectionError as exc:
        app.logger.error("Could not connect to Ollama: %s", exc)
        return render_template(
            'index.html',
            prompt=user_prompt,
            error="The local AI service is not available right now.",
        ), 503
    except (requests.HTTPError, requests.JSONDecodeError, KeyError) as exc:
        app.logger.error("Ollama returned an invalid response: %s", exc)
        return render_template(
            'index.html',
            prompt=user_prompt,
            error="The AI service returned an invalid response. Please try again.",
        ), 502

    return render_template('index.html', prompt=user_prompt, response=ai_response)


if __name__ == '__main__':
    app.run(debug=True)