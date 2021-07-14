def my_food_recommendation(food_name):
    import pandas as pd
    import numpy as np
    # Importing db of food items across all canteens registered on the platform df1=pd.read_csv('/Users/sahajchawla/Documents/AI/deeplizard/myfood1.csv')
    df1.columns = ['food_id','title','canteen_id','price', 'num_orders', 'category', 'avg_rating', 'num_rating', 'tags', 'imgsrc'] # mean of average ratings of all items
    C= df1['avg_rating'].mean()
    # the minimum number of votes required to appear in recommendation list, i.e, 60th percentile among 'num_rating' m= df1['num_rating'].quantile(0.6)
    # items that qualify the criteria of minimum num of votes q_items = df1.copy().loc[df1['num_rating'] >= m]
    # Calculation of weighted rating based on the IMDB formula

def weighted_rating(x, m=m, C=C):
    v = x['num_rating']
    R = x['avg_rating']
    return (v/(v+m) * R) + (m/(m+v) * C)
# Applying weighted_rating to qualified items 
q_items['score'] = q_items.apply(weighted_rating, axis=1)

# Shortlisting the top rated items and popular items 
top_rated_items = q_items.sort_values('score', ascending=False) 
pop_items= df1.sort_values('num_orders', ascending=False)

# Display results of demographic filtering 
#top_rated_items[['title', 'num_rating', 'avg_rating', 'score']].head() 
#pop_items[['title', 'num_orders']].head()

def create_referal(x):
    tags = x['tags'].lower().split(', ') 
    tags.extend(x['title'].lower().split()) 
    ztags.extend(x['category'].lower().split())
    return " ".join(sorted(set(tags), key=tags.index))
df1['referal name'] = df1.apply(create_referal, axis=1) #df1.head(15)
# Import CountVectorizer and create the count matrix
from sklearn.feature_extraction.text import CountVectorizer count = CountVectorizer(stop_words='english')
# df1['referal']
count_matrix = count.fit_transform(df1['referal name'])
# Compute the Cosine Similarity matrix based on the count_matrix
from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(count_matrix, count_matrix)
indices_from_title = pd.Series(df1.index, index=df1['title'])
indices_from_food_id = pd.Series(df1.index, index=df1['food_id'])
# Function that takes in food title or food id as input and outputs most similar dishes 
def get_recommendations(title="", cosine_sim=cosine_sim, idx=-1):
    # Get the index of the item that matches the title 
    if idx == -1 and title != "":
        idx = indices_from_title[title]
    # Get the pairwsie similarity scores of all dishes with that dish 
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort the dishes based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True) # Get the scores of the 10 most similar dishes
    sim_scores = sim_scores[1:3]
    # Get the food indices
    food_indices = [i[0] for i in sim_scores]
    # Return the top 10 most similar dishes return food_indices
    ans=df1.loc[get_recommendations(title=food_name)] a=ans.iloc[:,1].values
    b=ans.iloc[:,9].values
    c=np.concatenate((a,b))
    return c