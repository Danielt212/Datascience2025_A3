import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import sys
import os
import subprocess
from tqdm import tqdm
import gc
import pickle
import json

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Week10.load_data import load_data

class FeatureEngineering:
    def __init__(self):
        # Load the data
        print("Loading data files...")
        self.dfs = load_data()
        self.train_df = self.dfs['train']
        self.attributes_df = self.dfs['attributes']
        self.product_descriptions_df = self.dfs['product_descriptions']
        
        # Print number of unique queries
        unique_queries = self.train_df['search_term'].nunique()
        print(f"\nNumber of unique queries: {unique_queries}")
        
        # Load spaCy model
        print("\nLoading spaCy model...")
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except:
            print("Downloading spaCy model...")
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
            self.nlp = spacy.load('en_core_web_sm')
        
        print("Initializing TF-IDF vectorizer...")
        # Initialize TF-IDF vectorizer with limited features
        self.tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        
        # Create features DataFrame
        self.features_df = self.train_df[['search_term', 'product_uid', 'relevance']].copy()
        self.features_df['tfidf_similarity'] = 0.0
        self.features_df['spacy_similarity'] = 0.0
        self.features_df['attribute_matches'] = 0.0
        
        # Create directory for intermediate results
        os.makedirs('intermediate_results', exist_ok=True)
    
    def prepare_product_text(self, product_id):
        # Get product description
        description = self.product_descriptions_df[
            self.product_descriptions_df['product_uid'] == product_id
        ]['product_description'].iloc[0]
        
        # Get product attributes
        attributes = self.attributes_df[
            self.attributes_df['product_uid'] == product_id
        ]
        
        # Combine description and attributes
        attribute_text = ' '.join([
            f"{row['name']} {row['value']}"
            for _, row in attributes.iterrows()
        ])
        
        return f"{description} {attribute_text}"
    
    def calculate_tfidf_similarity(self, queries, products, batch_size=100):
        print("Calculating TF-IDF similarities...")
        # Fit TF-IDF vectorizer on all product texts
        all_product_texts = [self.prepare_product_text(pid) for pid in products['product_uid'].unique()]
        self.tfidf.fit(all_product_texts)
        
        # Process queries in batches
        for i in tqdm(range(0, len(queries), batch_size), desc="Processing TF-IDF batches"):
            batch_queries = queries.iloc[i:i+batch_size]
            batch_products = products[products['product_uid'].isin(batch_queries['product_uid'])]
            
            # Prepare texts
            query_texts = batch_queries['search_term'].tolist()
            product_texts = [self.prepare_product_text(pid) for pid in batch_products['product_uid']]
            
            # Transform texts to TF-IDF vectors
            query_vectors = self.tfidf.transform(query_texts)
            product_vectors = self.tfidf.transform(product_texts)
            
            # Calculate similarities
            similarities = cosine_similarity(query_vectors, product_vectors)
            
            # Update the main DataFrame with calculated similarities
            for idx, (_, row) in enumerate(batch_queries.iterrows()):
                product_idx = batch_products[batch_products['product_uid'] == row['product_uid']].index[0]
                self.features_df.loc[
                    (self.features_df['search_term'] == row['search_term']) & 
                    (self.features_df['product_uid'] == row['product_uid']),
                    'tfidf_similarity'
                ] = similarities[idx, product_idx]
            
            # Clear memory
            del query_vectors, product_vectors, similarities
            gc.collect()
    
    def calculate_spacy_similarity(self, queries, products, batch_size=50):
        print("Calculating SpaCy similarities...")
        # Process in batches
        for i in tqdm(range(0, len(queries), batch_size), desc="Processing SpaCy batches"):
            batch = queries.iloc[i:i+batch_size]
            
            # Calculate similarities
            similarities = []
            for _, row in batch.iterrows():
                query = row['search_term']
                product_id = row['product_uid']
                product_text = self.prepare_product_text(product_id)
                
                # Process texts with spaCy
                query_doc = self.nlp(query)
                product_doc = self.nlp(product_text)
                
                # Calculate similarity
                similarity = query_doc.similarity(product_doc)
                similarities.append(similarity)
            
            # Update the main DataFrame with calculated similarities
            for idx, (_, row) in enumerate(batch.iterrows()):
                self.features_df.loc[
                    (self.features_df['search_term'] == row['search_term']) & 
                    (self.features_df['product_uid'] == row['product_uid']),
                    'spacy_similarity'
                ] = similarities[idx]
            
            # Clear memory
            del similarities
            gc.collect()
    
    def calculate_attribute_matches(self, queries, products, attributes, batch_size=100):
        print("Calculating attribute matches...")
        # Process in batches
        for i in tqdm(range(0, len(queries), batch_size), desc="Processing attribute matches"):
            batch = queries.iloc[i:i+batch_size]
            
            # Calculate matches
            matches = []
            for _, row in batch.iterrows():
                query = row['search_term'].lower()
                product_id = row['product_uid']
                
                # Get product attributes
                product_attrs = attributes[attributes['product_uid'] == product_id]
                
                # Count matches in attribute names and values
                name_matches = sum(1 for name in product_attrs['name'] if any(term in name.lower() for term in query.split()))
                value_matches = sum(1 for value in product_attrs['value'] if any(term in str(value).lower() for term in query.split()))
                
                matches.append(name_matches + value_matches)
            
            # Update the main DataFrame with calculated matches
            for idx, (_, row) in enumerate(batch.iterrows()):
                self.features_df.loc[
                    (self.features_df['search_term'] == row['search_term']) & 
                    (self.features_df['product_uid'] == row['product_uid']),
                    'attribute_matches'
                ] = matches[idx]
            
            # Clear memory
            del matches
            gc.collect()
    
    def evaluate_features(self, features_df):
        print("\nFeature Evaluation:")
        print("------------------")
        
        # Calculate correlations with relevance score
        correlations = features_df[['tfidf_similarity', 'spacy_similarity', 'attribute_matches', 'relevance']].corr()['relevance']
        print("\nCorrelations with Relevance:")
        print(correlations)
        
        # Calculate feature statistics
        feature_stats = features_df[['tfidf_similarity', 'spacy_similarity', 'attribute_matches']].describe()
        print("\nFeature Statistics:")
        print(feature_stats)
        
        return correlations, feature_stats
    
    def run_feature_engineering(self):
        print("Starting feature engineering process...")
        
        # Calculate features
        self.calculate_tfidf_similarity(self.train_df, self.product_descriptions_df)
        self.calculate_spacy_similarity(self.train_df, self.product_descriptions_df)
        self.calculate_attribute_matches(self.train_df, self.product_descriptions_df, self.attributes_df)
        
        # Evaluate features
        correlations, feature_stats = self.evaluate_features(self.features_df)
        
        # Save features
        print("\nSaving engineered features...")
        self.features_df.to_csv('engineered_features.csv', index=False)
        print("Features saved to engineered_features.csv")
        
        return self.features_df, correlations, feature_stats

if __name__ == "__main__":
    # Create feature engineering instance
    fe = FeatureEngineering()
    
    # Run feature engineering
    features_df, correlations, feature_stats = fe.run_feature_engineering() 