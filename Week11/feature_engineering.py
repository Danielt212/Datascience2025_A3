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
        
    def prepare_product_text(self):
        """Combine product title, description, and attributes into a single text field"""
        print("\nPreparing product text...")
        # Create a dictionary of product descriptions
        product_texts = {}
        
        # Add product descriptions
        for _, row in self.product_descriptions_df.iterrows():
            product_texts[row['product_uid']] = row['product_description']
        
        # Add attributes
        for _, row in self.attributes_df.iterrows():
            product_uid = row['product_uid']
            if product_uid in product_texts:
                product_texts[product_uid] += f" {row['name']} {row['value']}"
            else:
                product_texts[product_uid] = f"{row['name']} {row['value']}"
        
        return product_texts
    
    def calculate_tfidf_similarity(self, product_texts):
        """Calculate TF-IDF similarity between queries and products"""
        print("\nCalculating TF-IDF similarities...")
        # Get unique queries and products
        queries = self.train_df['search_term'].unique()
        products = list(product_texts.keys())
        
        # Take a sample of queries for testing (e.g., first 1000)
        sample_size = 1000  # Adjust this number as needed
        queries = queries[:sample_size]
        print(f"Processing {len(queries)} queries...")
        
        # Process in smaller batches
        batch_size = 100
        chunk_size = 1000  # Save results every 1000 queries
        
        # Create directory for intermediate results if it doesn't exist
        os.makedirs('intermediate_results', exist_ok=True)
        
        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:i + batch_size]
            print(f"Processing queries {i} to {i + len(batch_queries)}...")
            
            # Create TF-IDF matrix for batch of queries
            query_tfidf = self.tfidf.fit_transform(batch_queries)
            
            # Create TF-IDF matrix for products
            product_texts_list = [product_texts[pid] for pid in products]
            product_tfidf = self.tfidf.transform(product_texts_list)
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query_tfidf, product_tfidf)
            
            # Save results for this batch
            batch_results = {}
            for j, query in enumerate(batch_queries):
                batch_results[query] = dict(zip(products, similarities[j]))
            
            # Save batch results
            chunk_num = i // chunk_size
            with open(f'intermediate_results/tfidf_chunk_{chunk_num}.json', 'a') as f:
                json.dump(batch_results, f)
                f.write('\n')
            
            # Clear memory
            del query_tfidf, product_tfidf, similarities, batch_results
            gc.collect()
        
        return None  # Results are saved in files
    
    def load_tfidf_similarities(self, query):
        """Load TF-IDF similarities for a specific query from saved files"""
        chunk_num = 0
        while True:
            try:
                with open(f'intermediate_results/tfidf_chunk_{chunk_num}.json', 'r') as f:
                    for line in f:
                        chunk_data = json.loads(line)
                        if query in chunk_data:
                            return chunk_data[query]
            except FileNotFoundError:
                break
            chunk_num += 1
        return {}
    
    def calculate_spacy_similarity(self, product_texts):
        """Calculate semantic similarity using spaCy"""
        print("\nCalculating spaCy similarities...")
        
        # Take a sample of the training data
        sample_size = 1000  # Adjust this number as needed
        sample_df = self.train_df.head(sample_size)
        print(f"Processing {len(sample_df)} rows...")
        
        # Process in smaller batches
        batch_size = 50
        chunk_size = 1000  # Save results every 1000 rows
        
        for i in tqdm(range(0, len(sample_df), batch_size)):
            batch_df = sample_df.iloc[i:i + batch_size]
            batch_results = {}
            
            for _, row in batch_df.iterrows():
                query = row['search_term']
                product_uid = row['product_uid']
                
                if product_uid in product_texts:
                    query_doc = self.nlp(query)
                    product_doc = self.nlp(product_texts[product_uid])
                    similarity = query_doc.similarity(product_doc)
                    batch_results[(query, product_uid)] = similarity
            
            # Save batch results
            chunk_num = i // chunk_size
            with open(f'intermediate_results/spacy_chunk_{chunk_num}.json', 'a') as f:
                json.dump(batch_results, f)
                f.write('\n')
            
            # Clear memory
            del batch_results
            gc.collect()
        
        return None  # Results are saved in files
    
    def load_spacy_similarities(self, query, product_uid):
        """Load spaCy similarities for a specific query-product pair from saved files"""
        chunk_num = 0
        while True:
            try:
                with open(f'intermediate_results/spacy_chunk_{chunk_num}.json', 'r') as f:
                    for line in f:
                        chunk_data = json.loads(line)
                        if (query, product_uid) in chunk_data:
                            return chunk_data[(query, product_uid)]
            except FileNotFoundError:
                break
            chunk_num += 1
        return 0
    
    def calculate_attribute_matches(self):
        """Calculate number of query terms that match product attributes"""
        print("\nCalculating attribute matches...")
        
        # Take a sample of the training data
        sample_size = 1000  # Adjust this number as needed
        sample_df = self.train_df.head(sample_size)
        print(f"Processing {len(sample_df)} rows...")
        
        # Process in smaller batches
        batch_size = 100
        chunk_size = 1000  # Save results every 1000 rows
        
        for i in tqdm(range(0, len(sample_df), batch_size)):
            batch_df = sample_df.iloc[i:i + batch_size]
            batch_results = {}
            
            for _, row in batch_df.iterrows():
                query = row['search_term'].lower()
                product_uid = row['product_uid']
                
                # Get attributes for this product
                product_attrs = self.attributes_df[self.attributes_df['product_uid'] == product_uid]
                
                # Count matches in attribute names and values
                name_matches = sum(1 for name in product_attrs['name'].str.lower() 
                                 if any(term in name for term in query.split()))
                value_matches = sum(1 for value in product_attrs['value'].str.lower() 
                                  if any(term in value for term in query.split()))
                
                batch_results[(query, product_uid)] = name_matches + value_matches
            
            # Save batch results
            chunk_num = i // chunk_size
            with open(f'intermediate_results/attribute_chunk_{chunk_num}.json', 'a') as f:
                json.dump(batch_results, f)
                f.write('\n')
            
            # Clear memory
            del batch_results
            gc.collect()
        
        return None  # Results are saved in files
    
    def load_attribute_matches(self, query, product_uid):
        """Load attribute matches for a specific query-product pair from saved files"""
        chunk_num = 0
        while True:
            try:
                with open(f'intermediate_results/attribute_chunk_{chunk_num}.json', 'r') as f:
                    for line in f:
                        chunk_data = json.loads(line)
                        if (query, product_uid) in chunk_data:
                            return chunk_data[(query, product_uid)]
            except FileNotFoundError:
                break
            chunk_num += 1
        return 0
    
    def evaluate_features(self):
        """Evaluate the effectiveness of each feature"""
        print("\nStarting feature evaluation...")
        
        # Calculate features if not already done
        if not os.path.exists('intermediate_results'):
            print("Calculating initial features...")
            product_texts = self.prepare_product_text()
            self.calculate_tfidf_similarity(product_texts)
            self.calculate_spacy_similarity(product_texts)
            self.calculate_attribute_matches()
            del product_texts
            gc.collect()
        
        print("\nLoading similarities into memory...")
        # Load all similarities into memory
        tfidf_similarities = {}
        spacy_similarities = {}
        attribute_matches = {}
        
        # Load TF-IDF similarities
        chunk_num = 0
        while True:
            try:
                with open(f'intermediate_results/tfidf_chunk_{chunk_num}.json', 'r') as f:
                    for line in f:
                        chunk_data = json.loads(line)
                        tfidf_similarities.update(chunk_data)
            except FileNotFoundError:
                break
            chunk_num += 1
        
        # Load spaCy similarities
        chunk_num = 0
        while True:
            try:
                with open(f'intermediate_results/spacy_chunk_{chunk_num}.json', 'r') as f:
                    for line in f:
                        chunk_data = json.loads(line)
                        spacy_similarities.update(chunk_data)
            except FileNotFoundError:
                break
            chunk_num += 1
        
        # Load attribute matches
        chunk_num = 0
        while True:
            try:
                with open(f'intermediate_results/attribute_chunk_{chunk_num}.json', 'r') as f:
                    for line in f:
                        chunk_data = json.loads(line)
                        attribute_matches.update(chunk_data)
            except FileNotFoundError:
                break
            chunk_num += 1
        
        print("\nCreating feature DataFrame...")
        # Use the same sample size as in other methods
        sample_size = 50  # Small sample for testing
        sample_df = self.train_df.head(sample_size)
        print(f"Processing {len(sample_df)} rows...")
        
        # Process in smaller batches
        batch_size = 10  # Smaller batch size for more frequent updates
        features = []
        
        for i in range(0, len(sample_df), batch_size):
            print(f"Processing batch {i//batch_size + 1} of {(len(sample_df) + batch_size - 1)//batch_size}")
            batch_df = sample_df.iloc[i:i + batch_size]
            
            for _, row in batch_df.iterrows():
                query = row['search_term']
                product_uid = row['product_uid']
                relevance = row['relevance']
                
                # Get similarities from memory
                tfidf_sim = tfidf_similarities.get(query, {}).get(product_uid, 0)
                spacy_sim = spacy_similarities.get((query, product_uid), 0)
                attr_matches = attribute_matches.get((query, product_uid), 0)
                
                feature_dict = {
                    'query': query,
                    'product_uid': product_uid,
                    'relevance': relevance,
                    'tfidf_similarity': tfidf_sim,
                    'spacy_similarity': spacy_sim,
                    'attribute_matches': attr_matches
                }
                features.append(feature_dict)
            
            # Save intermediate results
            if len(features) >= batch_size:
                temp_df = pd.DataFrame(features)
                temp_df.to_csv('engineered_features_temp.csv', mode='a', header=not os.path.exists('engineered_features_temp.csv'), index=False)
                features = []
                gc.collect()
        
        # Save any remaining features
        if features:
            temp_df = pd.DataFrame(features)
            temp_df.to_csv('engineered_features_temp.csv', mode='a', header=not os.path.exists('engineered_features_temp.csv'), index=False)
        
        # Load final results for correlation analysis
        print("\nCalculating feature correlations...")
        final_df = pd.read_csv('engineered_features_temp.csv')
        
        # Convert relevance to float and select only numeric columns for correlation
        numeric_cols = ['relevance', 'tfidf_similarity', 'spacy_similarity', 'attribute_matches']
        final_df[numeric_cols] = final_df[numeric_cols].astype(float)
        
        # Calculate correlations only for numeric columns
        correlations = final_df[numeric_cols].corr()['relevance'].sort_values(ascending=False)
        print("\nFeature correlations with relevance:")
        print(correlations)
        
        return final_df

if __name__ == "__main__":
    print("Starting feature engineering process...")
    fe = FeatureEngineering()
    features_df = fe.evaluate_features()
    
    # Save features to CSV
    print("\nSaving features to CSV...")
    features_df.to_csv('engineered_features.csv', index=False)
    print("\nFeatures saved to engineered_features.csv") 