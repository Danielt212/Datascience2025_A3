import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from load_data import load_data

def analyze_data():
    dfs = load_data()
    
    train_df = dfs['train']
    attributes_df = dfs['attributes']
    
    # 1. Total number of product-query pairs
    total_pairs = len(train_df)
    print(f"1. Total number of product-query pairs: {total_pairs}")
    
    # 2. Number of unique products
    unique_products = train_df['product_uid'].nunique()
    print(f"2. Number of unique products: {unique_products}")
    
    # 3. Two most occurring products
    product_counts = train_df['product_uid'].value_counts()
    top_products = product_counts.head(2)
    print("\n3. Two most occurring products:")
    for product_id, count in top_products.items():
        print(f"   Product ID {product_id}: {count} occurrences")
    
    # 4. Descriptive statistics for relevance values
    relevance_stats = train_df['relevance'].describe()
    print("\n4. Descriptive statistics for relevance values:")
    print(f"   Mean: {relevance_stats['mean']:.3f}")
    print(f"   Median: {relevance_stats['50%']:.3f}")
    print(f"   Standard Deviation: {relevance_stats['std']:.3f}")
    
    # 5. Histogram of relevance values
    plt.figure(figsize=(10, 6))
    sns.histplot(data=train_df, x='relevance', bins=20)
    plt.title('Distribution of Relevance Values')
    plt.xlabel('Relevance')
    plt.ylabel('Count')
    plt.savefig('relevance_distribution.png')
    plt.close()
    
    # 6. Top-5 most occurring brand names
    brand_related_attributes = attributes_df[attributes_df['name'].str.contains('brand|manufacturer|maker', case=False, na=False)]
    print("\nSample of brand-related attributes:")
    print(brand_related_attributes[['name', 'value']].head(10))
    
    brand_counts = brand_related_attributes['value'].value_counts().head(6)
    print("\n6. Top-5 most occurring brand names:")
    for brand, count in brand_counts.items():
        print(f"   {brand}: {count} occurrences")


if __name__ == "__main__":
    analyze_data() 