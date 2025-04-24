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
  padding: 40px;
  max-width: 1200px;
  margin: 0 auto;
  position: relative;
  z-index: 2;
`;

const Logo = styled.img`
  width: 180px;
  margin-bottom: 40px;
  display: block;
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
  font-size: 48px;
  margin-bottom: 40px;
  text-align: center;
  font-weight: 700;
  text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
  position: relative;
  
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
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 20px;
`;

const MovieCard = styled.div`
  background-color: rgba(0, 0, 0, 0.75);
  border-radius: 4px;
  padding: 20px;
  color: white;
`;

const MovieTitle = styled.h3`
  margin-bottom: 10px;
`;

const RatingContainer = styled.div`
  display: flex;
  gap: 5px;
  margin-top: 10px;
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
  margin-top: 40px;
  
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

const Welcome: React.FC = () => {
  const [ratings, setRatings] = useState<{ [key: number]: number }>({});
  const [movies, setMovies] = useState<Movie[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
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

  const handleSubmit = async () => {
    setSubmitting(true);
    try {
      // Delete existing ratings before storing new ones
      await deleteExistingRatings();

      const timestamp = Math.floor(Date.now() / 1000);
      
      // Create an array of rating promises
      const ratingPromises = Object.entries(ratings).map(([movieId, rating]) => {
        const putCommand = new PutCommand({
          TableName: 'movie-recommender-ratings',
          Item: {
            userId: userId,
            movieId: parseInt(movieId),
            rating: rating,
            timestamp: timestamp
          }
        });
        return docClient.send(putCommand);
      });

      // Execute all rating updates in parallel
      await Promise.all(ratingPromises);
      
      console.log('Ratings submitted successfully!');
      alert('Thanks for rating the movies!');
      
      // Fetch new movies after successful submission
      await fetchRandomMovies();
      setRatings({});  // Clear ratings
    } catch (err) {
      console.error('Error submitting ratings:', err);
      alert('Failed to submit ratings. Please try again.');
    } finally {
      setSubmitting(false);
    }
  };

  if (loading) {
    return (
      <BackgroundWrapper>
        <WelcomeContainer>
          <Logo src={yamrLogo} alt="YAMR Logo" />
          <Title>Loading movies...</Title>
        </WelcomeContainer>
      </BackgroundWrapper>
    );
  }

  if (error) {
    return (
      <BackgroundWrapper>
        <WelcomeContainer>
          <Logo src={yamrLogo} alt="YAMR Logo" />
          <Title>Error: {error}</Title>
        </WelcomeContainer>
      </BackgroundWrapper>
    );
  }

  return (
    <BackgroundWrapper>
      <WelcomeContainer>
        <Logo src={yamrLogo} alt="YAMR Logo" />
        <Title>Rate 10 Movies</Title>
        <MovieList>
          {movies.map(movie => (
            <MovieCard key={movie.movieId}>
              <MovieTitle>{movie.title}</MovieTitle>
              <RatingContainer>
                {[1, 2, 3, 4, 5].map(star => (
                  <StarButton
                    key={star}
                    active={ratings[movie.movieId] >= star}
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
      </WelcomeContainer>
    </BackgroundWrapper>
  );
};

export default Welcome; 