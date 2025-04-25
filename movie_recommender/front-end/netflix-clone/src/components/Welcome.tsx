import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { FaStar } from 'react-icons/fa';
import loginBg from '../assets/login_bg.png';
import yamrLogo from '../assets/yamr.png';
import { DynamoDBClient } from '@aws-sdk/client-dynamodb';
import { DynamoDBDocumentClient, ScanCommand, PutCommand, DeleteCommand } from '@aws-sdk/lib-dynamodb';

const dbClient = new DynamoDBClient({
  region: process.env.REACT_APP_AWS_REGION || 'us-east-1',
  credentials: {
    accessKeyId: process.env.REACT_APP_AWS_ACCESS_KEY_ID || '',
    secretAccessKey: process.env.REACT_APP_AWS_SECRET_ACCESS_KEY || ''
  }
});

const docClient = DynamoDBDocumentClient.from(dbClient);

const WelcomeContainer = styled.div`
  padding: 20px;
  max-width: 800px;
  margin: 0 auto;
  position: relative;
  z-index: 2;

  @media (min-width: 600px) {
    padding: 40px;
  }
`;

const LogoContainer = styled.div`
  display: flex;
  justify-content: center;
  align-items: center;
  margin-bottom: 40px;
`;

const Logo = styled.img`
  width: 140px;
  margin-bottom: 30px;
  display: block;
  margin-left: auto;
  margin-right: auto;
`;

const BackgroundWrapper = styled.div`
  min-height: 100vh;
  background-color: #141414;
  position: relative;
  
  &:before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-image: url(${loginBg});
    background-size: cover;
    background-position: center;
    opacity: 0.5;
    z-index: 1;
  }
`;

const Title = styled.h1`
  color: white;
  font-size: 32px;
  margin-bottom: 30px;
  text-align: center;
  font-weight: 700;
  text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
  position: relative;
  
  @media (min-width: 600px) {
    font-size: 48px;
  }

  &:after {
    content: '';
    position: absolute;
    bottom: -10px;
    left: 50%;
    transform: translateX(-50%);
    width: 100px;
    height: 4px;
    background-color: #e50914;
    border-radius: 2px;
  }
`;

const MovieList = styled.div`
  display: grid;
  gap: 20px;
  margin-bottom: 30px;
  width: 100%;

  /* Default for mobile: 1 column */
  grid-template-columns: 1fr;

  /* Small tablets: 2 columns */
  @media (min-width: 600px) {
    grid-template-columns: repeat(2, 1fr);
  }

  /* Tablets: 3 columns */
  @media (min-width: 900px) {
    grid-template-columns: repeat(3, 1fr);
  }

  /* Small desktops: 4 columns */
  @media (min-width: 1200px) {
    grid-template-columns: repeat(4, 1fr);
  }

  /* Large desktops: 5 columns */
  @media (min-width: 1500px) {
    grid-template-columns: repeat(5, 1fr);
  }
`;

const MovieCard = styled.div`
  background-color: rgba(0, 0, 0, 0.75);
  border-radius: 4px;
  padding: 20px;
  color: white;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  min-height: 120px;
  transition: transform 0.2s;
  
  &:hover {
    transform: scale(1.05);
    background-color: rgba(30, 30, 30, 0.9);
  }
`;

const MovieTitle = styled.h3`
  margin-bottom: 10px;
  font-size: 1rem;
  font-weight: 500;
  text-align: center;
  // Ensure long titles don't break layout
  overflow: hidden;
  text-overflow: ellipsis;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
`;

const RatingContainer = styled.div`
  display: flex;
  gap: 5px;
  justify-content: center;
  margin-top: auto;  // Push to bottom of card
`;

const StarButton = styled.button<{ active: boolean }>`
  background: none;
  border: none;
  color: ${props => props.active ? '#ffd700' : '#666'};
  cursor: pointer;
  font-size: 20px;
  padding: 0;
  
  &:hover {
    color: #ffd700;
  }
`;

const SubmitButton = styled.button`
  background-color: #e50914;
  color: white;
  border: none;
  border-radius: 4px;
  padding: 12px 24px;
  font-size: 16px;
  font-weight: 500;
  cursor: pointer;
  margin: 30px auto;
  display: block;
  min-width: 200px;
  transition: background-color 0.2s, transform 0.1s, opacity 0.2s;
  
  &:hover {
    background-color: #f40612;
    transform: scale(1.02);
  }

  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
  }
`;

const RecommendationsContainer = styled.div`
  margin-top: 20px;
  padding: 30px;
  background-color: rgba(0, 0, 0, 0.75);
  border-radius: 4px;
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
  max-width: 600px;
  margin-left: auto;
  margin-right: auto;

  ${Logo} {
    background: none;
    margin-bottom: 30px;
  }
`;

const RecommendationsTitle = styled.h2`
  color: white;
  margin-bottom: 20px;
  text-align: center;
  font-size: 1.8rem;
  font-weight: 700;
`;

const RecommendationCard = styled.div`
  background-color: rgba(20, 20, 20, 0.8);
  border-radius: 4px;
  padding: 12px;
  margin-bottom: 8px;
  color: white;
  transition: transform 0.2s;
  cursor: pointer;
  text-align: center;
  width: 100%;
  
  &:hover {
    transform: scale(1.02);
    background-color: rgba(30, 30, 30, 0.9);
  }
`;

const RecommendationTitle = styled.h3`
  margin-bottom: 5px;
  font-size: 1.1rem;
  font-weight: 500;
  text-align: center;
`;

const RateAgainButton = styled.button`
  background-color: #e50914;
  color: white;
  border: none;
  border-radius: 4px;
  padding: 10px 20px;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  margin-top: 20px;
  display: block;
  margin-left: auto;
  margin-right: auto;
  
  &:hover {
    background-color: #f40612;
  }
`;

interface Movie {
  movieId: number;
  title: string;
}

interface DynamoDBMovie {
  movieId: number;
  title: string;
  [key: string]: any;
}

interface Recommendation {
  movie_id: string;
  title: string;
  predicted_rating: number;
}

const Welcome: React.FC = () => {
  const [ratings, setRatings] = useState<{ [key: number]: number }>({});
  const [movies, setMovies] = useState<Movie[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const userId = 611; // Hardcoded user ID

  useEffect(() => {
    fetchRandomMovies();
  }, []);

  const fetchRandomMovies = async () => {
    try {
      const scanCommand = new ScanCommand({
        TableName: 'movie-recommender-movies',
        ProjectionExpression: 'movieId, title'
      });

      const response = await docClient.send(scanCommand);
      
      if (!response.Items || response.Items.length === 0) {
        setError('No movies found');
        setLoading(false);
        return;
      }

      // Shuffle the array and take 10 random movies
      const shuffledMovies = (response.Items as DynamoDBMovie[])
        .sort(() => Math.random() - 0.5)
        .slice(0, 10)
        .map(item => ({
          movieId: item.movieId,
          title: item.title
        }));

      setMovies(shuffledMovies);
      setLoading(false);
    } catch (err) {
      console.error('Error fetching movies:', err);
      setError('Failed to fetch movies');
      setLoading(false);
    }
  };

  const handleRating = (movieId: number, rating: number) => {
    setRatings(prev => ({
      ...prev,
      [movieId]: rating
    }));
  };

  const deleteExistingRatings = async () => {
    try {
      // First, scan to find all ratings for this user
      const scanCommand = new ScanCommand({
        TableName: 'movie-recommender-ratings',
        FilterExpression: 'userId = :userId',
        ExpressionAttributeValues: {
          ':userId': userId
        }
      });

      const response = await docClient.send(scanCommand);
      
      if (!response.Items || response.Items.length === 0) {
        return; // No ratings to delete
      }

      // Delete each rating
      const deletePromises = response.Items.map(item => {
        const deleteCommand = new DeleteCommand({
          TableName: 'movie-recommender-ratings',
          Key: {
            userId: item.userId,
            movieId: item.movieId
          }
        });
        return docClient.send(deleteCommand);
      });

      await Promise.all(deletePromises);
      console.log('Deleted existing ratings for user:', userId);
    } catch (err) {
      console.error('Error deleting existing ratings:', err);
      throw err;
    }
  };

  const fetchRecommendations = async (userId: number) => {
    try {
      const response = await fetch('https://yamrinf.storybookllm.com:5000/recommend', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: userId,
          top_n: 3
        })
      });

      if (!response.ok) {
        throw new Error('Failed to fetch recommendations');
      }

      const data = await response.json();
      setRecommendations(data.recommendations);
    } catch (err) {
      console.error('Error fetching recommendations:', err);
      setError('Failed to fetch recommendations');
    }
  };

  const handleSubmit = async () => {
    setSubmitting(true);
    try {
      // Format ratings for the inference container
      const formattedRatings = Object.entries(ratings).map(([movieId, rating]) => ({
        movie_id: parseInt(movieId),
        rating: rating
      }));

      // Send ratings to inference container
      const response = await fetch('https://yamrinf.storybookllm.com:5000/rate/batch', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: userId,
          ratings: formattedRatings
        })
      });

      if (!response.ok) {
        throw new Error('Failed to submit ratings to inference container');
      }

      const result = await response.json();
      console.log('Ratings submitted successfully:', result);
      
      // Fetch recommendations after successful rating submission
      await fetchRecommendations(userId);
      
      // Clear the movies list since we're showing recommendations now
      setMovies([]);
      setRatings({});  // Clear ratings
    } catch (err) {
      console.error('Error submitting ratings:', err);
      setError('Failed to submit ratings. Please try again.');
    } finally {
      setSubmitting(false);
    }
  };

  const handleRateAgain = () => {
    setRecommendations([]);
    fetchRandomMovies();
  };

  if (loading) {
    return (
      <BackgroundWrapper>
        <WelcomeContainer>
          <LogoContainer>
            <Logo src={yamrLogo} alt="YAMR Logo" />
          </LogoContainer>
          <Title>Loading movies...</Title>
        </WelcomeContainer>
      </BackgroundWrapper>
    );
  }

  if (error) {
    return (
      <BackgroundWrapper>
        <WelcomeContainer>
          <LogoContainer>
            <Logo src={yamrLogo} alt="YAMR Logo" />
          </LogoContainer>
          <Title>Error: {error}</Title>
        </WelcomeContainer>
      </BackgroundWrapper>
    );
  }

  return (
    <BackgroundWrapper>
      <WelcomeContainer>
        {movies.length > 0 ? (
          <>
            <LogoContainer>
              <Logo src={yamrLogo} alt="YAMR Logo" />
            </LogoContainer>
            <Title>Rate 10 Movies</Title>
            <MovieList>
              {movies.map(movie => (
                <MovieCard key={movie.movieId}>
                  <MovieTitle>{movie.title}</MovieTitle>
                  <RatingContainer>
                    {[1, 2, 3, 4, 5].map(star => (
                      <StarButton
                        key={star}
                        active={ratings[movie.movieId] ? ratings[movie.movieId] >= star : false}
                        onClick={() => handleRating(movie.movieId, star)}
                      >
                        {FaStar({ size: 20 })}
                      </StarButton>
                    ))}
                  </RatingContainer>
                </MovieCard>
              ))}
            </MovieList>
            <SubmitButton 
              onClick={handleSubmit} 
              disabled={submitting || Object.keys(ratings).length === 0}
            >
              {submitting ? 'Submitting...' : 'Submit Ratings'}
            </SubmitButton>
          </>
        ) : (
          recommendations.length > 0 && (
            <>
              <LogoContainer>
                <Logo src={yamrLogo} alt="YAMR Logo" />
              </LogoContainer>
              <RecommendationsContainer>
                <RecommendationsTitle>Recommended Movies For You</RecommendationsTitle>
                {recommendations.map(rec => (
                  <RecommendationCard key={rec.movie_id}>
                    <RecommendationTitle>{rec.title}</RecommendationTitle>
                  </RecommendationCard>
                ))}
                <RateAgainButton onClick={handleRateAgain}>
                  Rate More Movies
                </RateAgainButton>
              </RecommendationsContainer>
            </>
          )
        )}
      </WelcomeContainer>
    </BackgroundWrapper>
  );
};

export default Welcome; 