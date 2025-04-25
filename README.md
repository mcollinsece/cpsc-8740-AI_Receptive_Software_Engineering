# CPSC-8740 AI Receptive Software Engineering

This repository contains projects developed for the CPSC-8740 AI Receptive Software Engineering course.

## Final Project: YAMR (Yet Another Movie Recommender)

The movie recommender system is located in the `movie_recommender/` directory. It's a full-stack application that provides personalized movie recommendations using collaborative filtering and machine learning techniques.

### Features
- Personalized movie recommendations based on user ratings
- Netflix-style user interface for rating movies
- RESTful API for recommendation service
- Hybrid recommendation system combining collaborative and content-based filtering
- Support for new users with cold-start handling

### Tech Stack
- Frontend: React with TypeScript, styled-components
- Backend: Python, Flask, PyTorch
- Machine Learning: Collaborative Filtering, Neural Networks
- Deployment: Docker, AWS

### Project Structure
```
movie_recommender/
├── front-end/          # React frontend application
├── models/            # ML model implementations
├── data/             # Data processing and management
├── scripts/          # Utility scripts
├── web/             # Web service templates
├── app.py           # Flask application
├── recommender_service.py  # Main recommendation service
├── train.py         # Model training script
└── requirements.txt  # Python dependencies
```

## Module Projects

### Module 4 - GUI Applications
Located in `module4/`, this directory contains Python GUI applications built with Tkinter:
- Calculator with basic arithmetic operations
- Todo List for task management
- Tic-tac-toe game with two-player support


## Contributing

This is a course project repository. While it's not open for direct contributions, feel free to fork the repository and adapt the code for your own projects.

## License

This project is part of academic coursework and is provided as-is for educational purposes.