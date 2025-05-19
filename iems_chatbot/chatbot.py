import pandas as pd
import numpy as np
import sqlite3
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class IEMSCourseChatbot:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.df = self.load_data()
        if self.df is not None and not self.df.empty:
            self.clean_and_embed()
        else:
            self.df = pd.DataFrame()

    def load_data(self):
        try:
            conn = sqlite3.connect("data/courses.db")
            df = pd.read_sql_query(
                "SELECT course_code, course_name, pre_requisites, what_it_satisfies, description FROM courses",
                conn
            )
            conn.close()
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def clean_and_embed(self):
        self.df["course_code"] = (
            self.df["course_code"]
            .fillna("")
            .str.upper()
            .str.replace("-", " ", regex=False)
            .str.strip()
        )
        self.df["combined_text"] = (
            self.df["course_code"] + " " +
            self.df["course_name"].fillna("") + " " +
            self.df["description"].fillna("")
        )
        self.df["embedding"] = self.df["combined_text"].apply(lambda x: self.model.encode(x))

    def semantic_search(self, query, threshold=0.35):
        query_cleaned = query.lower()
        query_cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', query_cleaned)
        query_cleaned = query_cleaned.replace("-", " ").strip()
        query_vec = self.model.encode(query_cleaned)

        similarities = [cosine_similarity([query_vec], [emb])[0][0] for emb in self.df["embedding"]]

        # Boost similarity if course code appears
        course_code = self.extract_course_code(query)
        if course_code:
            self.df["boost"] = self.df["course_code"].apply(lambda c: 1.0 if course_code in c else 0.0)
            similarities = [sim + 0.15 * boost for sim, boost in zip(similarities, self.df["boost"])]

        print(f"Semantic Search for Query: '{query_cleaned}'")
        for i, score in enumerate(similarities):
            print(f"{self.df.iloc[i]['course_code']}: {score:.3f}")

        best_idx = int(np.argmax(similarities))
        if similarities[best_idx] >= threshold:
            return self.df.iloc[best_idx]
        return None

    def keyword_fallback(self, query):
        query = query.lower().replace("-", " ")
        mask = self.df["combined_text"].str.lower().str.contains(query)
        matches = self.df[mask]
        return matches.iloc[0] if not matches.empty else None

    def classify_intent(self, query):
        query = query.lower()
        if "prerequisite" in query:
            return "prerequisites"
        elif "satisfies" in query or "requirement" in query:
            return "requirements"
        elif "compare" in query or "better" in query:
            return "comparison"
        return "general_info"

    def extract_course_code(self, query):
        query = query.upper().replace("-", " ")
        match = re.search(r"\b([A-Z]{2,4})\s?(\d{2,3})\b", query)
        return f"{match.group(1)} {match.group(2)}" if match else None

    def respond_to_query(self, query):
        if not query or len(query) < 3:
            return "Please enter a query with at least 3 characters."

        if self.df.empty:
            return "Sorry, I couldn’t load course data. Please check the database setup or try again later."

        intent = self.classify_intent(query)
        course = self.semantic_search(query)

        if course is None:
            course = self.keyword_fallback(query)

        if course is None:
            print("NO MATCH:", query)
            return "Sorry, I couldn’t find a matching course. Try a different query or check spelling."

        if intent == "prerequisites":
            value = course["pre_requisites"] if pd.notna(course["pre_requisites"]) else "None"
            return f"Prerequisites for {course['course_code']}: {value}"
        elif intent == "requirements":
            value = course["what_it_satisfies"] if pd.notna(course["what_it_satisfies"]) else "No information"
            return f"What {course['course_code']} satisfies: {value}"
        elif intent == "general_info":
            return f"{course['course_code']} - {course['course_name']}: {course['description']}"
        elif intent == "comparison":
            return "Course comparison features are under development. Please ask about one course at a time."
        else:
            return "I'm not sure how to help with that yet. Try asking about prerequisites or what a course satisfies."
