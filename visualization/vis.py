import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

def load_recipe_data():
    """Load and return the recipes dataset."""
    df = pd.read_csv('shuyangli94/food-com-recipes-and-user-interactions/versions/2/RAW_recipes.csv')
    # Convert submitted date to datetime
    df['submitted'] = pd.to_datetime(df['submitted'])
    return df

def plot_top_tags(df, top_n=10):
    """Create a bar plot of the most common recipe tags."""
    plt.figure(figsize=(12, 6))
    all_tags = df['tags'].apply(eval).explode()
    tag_counts = all_tags.value_counts().head(top_n)
    
    sns.barplot(x=tag_counts.values, y=tag_counts.index)
    plt.title(f'Top {top_n} Recipe Tags')
    plt.xlabel('Number of Recipes')
    plt.ylabel('Tag')
    plt.tight_layout()
    plt.savefig('top_tags.png')
    plt.close()

def plot_submissions_over_time(df):
    """Create a line plot showing recipe submissions over time."""
    plt.figure(figsize=(15, 6))
    submissions_by_month = df.resample('M', on='submitted')['id'].count()
    
    plt.plot(submissions_by_month.index, submissions_by_month.values)
    plt.title('Recipe Submissions Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Recipes Submitted')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('submissions_over_time.png')
    plt.close()

def plot_nutrition_distribution(df):
    """Create box plots for nutritional information."""
    plt.figure(figsize=(15, 6))
    nutrition_df = pd.DataFrame(df['nutrition'].apply(eval).tolist(),
                              columns=['calories', 'total_fat', 'sugar', 'sodium',
                                     'protein', 'saturated_fat', 'carbohydrates'])
    
    # Create box plot for each nutritional value
    sns.boxplot(data=nutrition_df)
    plt.title('Distribution of Nutritional Values')
    plt.xticks(rotation=45)
    plt.ylabel('Amount')
    plt.tight_layout()
    plt.savefig('nutrition_distribution.png')
    plt.close()

def plot_recipe_complexity_distribution(df):
    """Create a joint plot showing the relationship between steps and ingredients."""
    plt.figure(figsize=(10, 6))
    df['n_ingredients'] = df['ingredients'].apply(lambda x: len(eval(x)))
    
    g = sns.jointplot(
        data=df,
        x='n_steps',
        y='n_ingredients',
        kind='hex',
        height=10,
        joint_kws={'gridsize': 20}
    )
    g.fig.suptitle('Recipe Complexity: Steps vs Ingredients', y=1.02)
    g.ax_joint.set_xlabel('Number of Steps')
    g.ax_joint.set_ylabel('Number of Ingredients')
    plt.tight_layout()
    plt.savefig('recipe_complexity.png')
    plt.close()

def plot_preparation_time_by_ingredients(df):
    """Create a violin plot showing cooking time distribution by number of ingredients."""
    plt.figure(figsize=(15, 6))
    df['n_ingredients'] = df['ingredients'].apply(lambda x: len(eval(x)))
    df['minutes'] = df['minutes'].clip(0, 300)  # Clip extreme values
    
    # Create ingredient bins
    df['ingredient_bins'] = pd.qcut(df['n_ingredients'], q=6, labels=['Very Few', 'Few', 'Medium-Low', 
                                                                     'Medium-High', 'Many', 'Very Many'])
    
    sns.violinplot(data=df, x='ingredient_bins', y='minutes')
    plt.title('Cooking Time Distribution by Number of Ingredients')
    plt.xlabel('Number of Ingredients')
    plt.ylabel('Cooking Time (minutes)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('cooking_time_by_ingredients.png')
    plt.close()

if __name__ == "__main__":
    # Load the data
    df = load_recipe_data()
    
    # Create all visualizations
    plot_top_tags(df)
    plot_submissions_over_time(df)
    plot_nutrition_distribution(df)
    plot_recipe_complexity_distribution(df)
    plot_preparation_time_by_ingredients(df)
