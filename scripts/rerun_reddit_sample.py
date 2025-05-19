import pandas as pd
import os
import random

# Current list of cities
CITIES = [
    "southbend", "rockford", "kzoo", "scranton", "fayetteville",
    "sanfrancisco", "portland", "baltimore", "buffalo", "elpaso"
]

BASE_PATH = "data"
OUTPUT_PATH = "output/sampled_reddit_comments_by_city.csv"
COMMENTS_PER_CITY = 50

# Expected columns in output
REQUIRED_COLUMNS = [
    "Submission Title", "Submission Score", "Submission URL", "Submission Timestamp",
    "Comment", "Comment Score", "Comment Timestamp", "City"
]

def load_city_data(city):
    """Load filtered comments for a city."""
    filtered_path = os.path.join(BASE_PATH, city, "reddit", "filtered_comments.csv")
    
    if not os.path.exists(filtered_path):
        print(f"Warning: No filtered comments found for {city}")
        return None
    
    try:
        df = pd.read_csv(filtered_path)
        df["City"] = city
        
        # Clean up the data
        df["Comment Timestamp"] = pd.to_datetime(df["Comment Timestamp"], errors='coerce')
        df = df.dropna(subset=["Comment Timestamp"])
        
        # Filter out removed/deleted comments
        df = df[~df["Comment"].isin(["[removed]", "[deleted]"])]
        
        # Filter out very short comments (less than 5 words)
        df = df[df["Comment"].str.split().apply(lambda x: len([w for w in x if w.strip()])) >= 5]
        
        # Remove duplicates
        df = df.drop_duplicates(subset=["Comment"])
        
        return df
    except Exception as e:
        print(f"Warning: Could not load filtered comments for {city}: {e}")
        return None

def get_existing_comments():
    """Load existing comments from output file if it exists."""
    if not os.path.exists(OUTPUT_PATH):
        return pd.DataFrame()
    
    try:
        return pd.read_csv(OUTPUT_PATH)
    except Exception as e:
        print(f"Warning: Could not read existing output file: {e}")
        return pd.DataFrame()

def main():
    # Load existing comments
    existing_df = get_existing_comments()
    
    # Filter out cities not in current list
    existing_df = existing_df[existing_df["City"].isin(CITIES)]
    
    all_samples = []
    
    for city in CITIES:
        print(f"\nProcessing {city}...")
        
        # Get existing comments for this city
        city_existing = existing_df[existing_df["City"] == city]
        
        # Apply the same filtering to existing comments
        if not city_existing.empty:
            # Filter out removed/deleted comments
            city_existing = city_existing[~city_existing["Comment"].isin(["[removed]", "[deleted]"])]
            # Filter out very short comments
            city_existing = city_existing[city_existing["Comment"].str.split().apply(lambda x: len([w for w in x if w.strip()])) >= 5]
        
        # Load city data
        city_df = load_city_data(city)
        if city_df is None:
            print(f"Warning: No valid data found for {city}. Skipping.")
            continue
        
        # Get available comments (not already in existing set)
        available_comments = city_df[~city_df["Comment"].isin(city_existing["Comment"])]
        print(f"Available new comments: {len(available_comments)}")
        
        # Start with existing comments
        city_sample = city_existing.copy()
        
        # Keep adding random comments until we have COMMENTS_PER_CITY
        while len(city_sample) < COMMENTS_PER_CITY:
            if len(available_comments) == 0:
                print(f"Warning: Not enough valid comments available for {city}. Only got {len(city_sample)} comments.")
                break
                
            # Get a random comment from available ones
            new_comment = available_comments.sample(n=1, random_state=random.randint(0, 10000))
            # Add it to our sample
            city_sample = pd.concat([city_sample, new_comment], ignore_index=True)
            # Remove it from available comments
            available_comments = available_comments[~available_comments["Comment"].isin(new_comment["Comment"])]
            print(f"Added comment {len(city_sample)} of {COMMENTS_PER_CITY}")
        
        all_samples.append(city_sample)
        print(f"Successfully sampled {len(city_sample)} comments for {city}")
    
    if not all_samples:
        print("Error: No valid samples were collected for any city.")
        return
    
    # Combine all samples and save
    final_df = pd.concat(all_samples, ignore_index=True)
    
    # Ensure all required columns are present and in correct order
    for col in REQUIRED_COLUMNS:
        if col not in final_df.columns:
            print(f"Error: Missing required column {col}")
            return
    
    final_df = final_df[REQUIRED_COLUMNS]
    
    final_df.to_csv(OUTPUT_PATH, index=False)
    
    # Print summary
    print("\nSampling complete!")
    city_counts = final_df["City"].value_counts()
    successful_cities = city_counts[city_counts == COMMENTS_PER_CITY].index.tolist()
    
    print(f"Successfully sampled {COMMENTS_PER_CITY} comments each for {len(successful_cities)} cities:")
    for city in successful_cities:
        print(f"- {city}")
    
    skipped_cities = set(CITIES) - set(successful_cities)
    if skipped_cities:
        print("\nThe following cities were skipped or had insufficient data:")
        for city in skipped_cities:
            count = city_counts.get(city, 0)
            print(f"- {city}: {count} comments")

if __name__ == "__main__":
    main()
