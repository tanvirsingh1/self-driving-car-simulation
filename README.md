# Self-Driving Car Simulation (AI Focused)

## Overview
This project implements a convolutional neural network (CNN) to control a self-driving car in a simulated environment.  
The goal is to predict the appropriate steering angle from images captured by the vehicle’s front-facing camera, ensuring the vehicle remains on the road.

---

## Features
- Real-time steering prediction from camera feed
- Data preprocessing and augmentation for robust training
- CNN model designed specifically for image-based steering control
- Integration with simulation environment for testing
- Modular, well-documented Python code

---

## Installation & Setup
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd self-driving-car-simulation
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate      # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the training script:
   ```bash
   python model/train_model.py
   ```

5. Start the simulation test:
   ```bash
   python test/TestSimulation.py
   ```

---

## Usage
- **Training**: Uses pre-recorded driving data to train the CNN to predict steering angles.
- **Simulation**: The trained model can be used in the simulation to steer the car autonomously.

---

## Dataset
The dataset consists of images and steering angle measurements collected while driving in the simulator.  
Data augmentation techniques such as flipping, brightness adjustment, and translation were applied to improve generalization.

---

## Model Architecture
- Input: 66x200 RGB images
- Convolutional layers for feature extraction
- Fully connected layers for steering prediction
- Dropout layers to prevent overfitting
- Optimizer: Adam
- Loss Function: Mean Squared Error (MSE)

---

## Contributors
This project was completed collaboratively by all three team members.  
Each member contributed equally to all major parts of the project, including:
- Data preprocessing and augmentation  
- CNN model design and training  
- Simulation integration and testing  
- Debugging and performance tuning  
- Documentation and report preparation  

---

## License
This project is for educational purposes only and not intended for real-world autonomous vehicle deployment.

## Demo Video
[Watch the simulation result on YouTube](https://www.youtube.com/watch?v=Jb8ngjwQ7T8&ab_channel=GazalGarg)