import re

import pandas as pd
import requests
from bs4 import BeautifulSoup
from langdetect import detect
from tqdm import tqdm

from secret import S2_API_KEY


def clean(text):
    return re.sub(r"\s+", " ", text).strip()

def load_survey():
    path = "../data/philpapers_survey.html"
    with open(path) as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("div", class_="results-wrapper")[:100]
    headings = soup.find_all("h5")[:100]
    questions = []
    options = []
    responses = []
    for question, table in zip(headings, tables):
        questions += [question.text.strip()]
        option = []
        for o in table.find_all("div", class_="option-name"):
            if ":" in o.text:
                option += [clean(o.text.split(":")[1])]
            else:
                option += [clean(o.text).lower()]
        options += [option]
        response = []
        for r in table.find_all("div", class_="option-value"):
            if "(" in r.text:
                response += [float(r.text.split("(")[1].strip()[:-2]) / 100]
            else:
                response += [float(r.text.strip()[:-1]) / 100]
        responses += [response]
    return questions, options, responses

def load_correlations():
    path = "../data/philpapers_correlations.html"
    with open(path) as f:
        html = f.read()
    question_pairs = []
    option_pairs = []
    response_tables = []
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table", class_="table")[:100]
    for table in tables:
        a = table.find("span", class_="colname").text.split(":")
        question_a = clean(a[0].strip())
        option_a = clean(a[1].strip())
        b = table.find("td", class_="rowname").text.split(":")
        question_b = clean(b[0].strip())
        option_b = clean(b[1].strip())
        question_pairs += [(question_a, question_b)]
        option_pairs += [(option_a, option_b)]
        response_table = table.find_all("td")[1:]
        response_table = [int(r.text) for r in response_table]
        response_tables += [response_table]
    return question_pairs, option_pairs, response_tables

def fetch_paper_ids(questions, load=None, save=None):
    if load:
        ids = pd.read_feather(load)
        return ids
    categories = []
    for question in questions:
        q = question.split(":")[0].lower()
        if "(" in q:
            q = q.split("(")[0].strip()
        categories += [q]
    ids = set()
    for category in categories:
        print(f"Fetching papers from {category}...")
        offset = 0
        while True:
            url = "https://api.semanticscholar.org/graph/v1/paper/search"
            limit = 100
            params = {
                "query": category,
                "offset": offset,
                "fieldsOfStudy": "Philosophy",
                "limit": limit,
            }
            headers = {"x-api-key": S2_API_KEY}
            r = requests.get(url, params=params, headers=headers).json()
            if "data" in r:
                ids.update([p["paperId"] for p in r["data"]])
                if "next" in r:
                    print(
                        f"Next: {r['next']}, Total: {r['total']}, Limit: {limit}, Offset: {offset}")
                    offset = r["next"]
                    limit = min(r["total"] - offset, 100)
                    if offset < r["total"]:
                        continue
            break
    ids = pd.DataFrame(list(ids)).rename(columns={0: "id"})
    if save:
        ids.to_feather(save)
    return ids

def detect_language(data):
    is_english = data["abstract"].values.tolist()
    for i in tqdm(range(len(is_english)), total=len(is_english)):
        try:
            is_english[i] = detect(is_english[i]) == "en"
        except:
            pass
    return is_english

def fetch_abstracts(ids, load=None, save=None):
    if load:
        data = pd.read_feather(load)
        return data
    data = []
    ids = ids["id"].values.tolist()
    batch_size = 500
    for i in tqdm(range(0, len(ids), batch_size), total=len(ids)//batch_size):
        batch = ids[i:i+batch_size]
        url = "https://api.semanticscholar.org/graph/v1/paper/batch"
        params = {
            "fields": "title,abstract,year,venue,authors,citationCount,citations,references"
        }
        headers = {"x-api-key": S2_API_KEY}
        jsn = {"ids": batch}
        r = requests.post(url, params=params, headers=headers, json=jsn).json()
        for paper in r:
            if paper and "abstract" in paper and paper["abstract"]:
                author_ids = [x["authorId"] for x in paper["authors"] if "authorId" in x]
                paper["authors"] = author_ids
                data += [paper]
    data = pd.DataFrame(data).dropna()
    data = (
        data.loc[(data["abstract"].str.len() > 100) & (data["year"] > 1980)]
        .reset_index(drop=True)
    )
    data["is_english"] = detect_language(data)
    data = (
        data.loc[data["is_english"] == True]
        .reset_index(drop=True)
        .rename(columns={"paperId": "id", "citationCount": "citations"})
    )
    del data["is_english"]
    data["year"] = data["year"].astype(int)
    if save:
        data.to_feather(save)
    return data