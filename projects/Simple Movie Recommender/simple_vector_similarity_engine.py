import numpy as np

# Get user genre preference
genres = ("Romance", "Action", "Comedy", "Psychological", "Horror")
user_genres = list()

print("What is your favorite movie genre, rate from -10 to 10: ")
for i in range(len(genres)):
    data = float(input(f"{genres[i]}: "))
    user_genres.append(data)

user_genres = np.array(user_genres)

# 1. "Interstellar" (Heavy Psychological/Sci-Fi thriller vibe, low comedy, minor romance)
# 2. "Superbad" (Pure Comedy, zero Horror/Psychological, low Action)
# 3. "La La Land" (Peak Romance, decent Comedy, absolutely zero Horror)
# 4. "John Wick" (Pure Action, high Thriller/Psychological tension, low Romance)
# 5. "The Conjuring" (Max Horror, high Psychological dread, negative Romance)

movie_names = ["Interstellar", "Superbad", "La La Land", "John Wick", "The Conjuring"]

# Movie genre scoring
movie_matrix = np.array([
    [-2,  4, -5,  9,  3],  # Interstellar
    [-6,  1, 10, -8, -10], # Superbad
    [10, -4,  6, -3, -10], # La La Land
    [-3, 10, -2,  7,  2],  # John Wick
    [-8,  2, -7,  8, 10]   # The Conjuring
])

highest_score = 0
lowest_score = 0

for i in range(len(movie_names)):
    magnitude_A = np.sqrt(np.sum(np.power(user_genres, 2)))
    magnitude_B = np.sqrt(np.sum(np.power(movie_matrix[i], 2)))

    cosine_similarity = np.dot(user_genres, movie_matrix[i])/(magnitude_A * magnitude_B)

    if cosine_similarity > highest_score:
        highest_score = i # Set the movie the user might like
    if cosine_similarity < lowest_score:
        lowest_score = i # Set the movie the user might not like; just cosmetics lol

print("\n=== Output ===")
print(f"I recommend this '{movie_names[highest_score]}' movie. You might like it.")
print(f"I recommend you avoid this '{movie_names[lowest_score]}' movie...")

input("\nPress Enter to exit...")
