
# 🚀 Motion Analysis Project 🎥

Welcome to the **Motion Analysis Project**! This project detects and analyzes motion in video frames. Using custom Convolutional Neural Networks (CNN) and optical flow analysis, it detects moving objects, tracks them, and presents real-time analytics such as object speed, object count, and more—all within an interactive Flask web dashboard. 🖥️

## ✨ Features

- **Motion Detection**: Classifies moving vs. non-moving objects 🟩🟥
- **Speed Calculation**: Displays the speed of detected moving objects 🏃💨
- **Heatmap Visualization**: Shows motion regions in real-time 🌡️
- **Object Count & Classification**: Detects different types of objects 🎯
- **Analytics Dashboard**: Real-time data visualization 📊

## 📋 Prerequisites

Before running this project, make sure you have the following installed:

- **Python 3.x** (Python 3.7+ recommended) 🐍
- **Git** (for version control and cloning the repository) 📦
- **Conda** (optional, for virtual environment management) 🍃

### ✅ Dependencies

- **TensorFlow**: Deep learning framework for building the motion detection model 🤖
- **Flask**: Web framework for creating the dashboard 💻
- **OpenCV**: Library for real-time computer vision 🧐
- **NumPy**: Numerical computing library for data handling 🧮
- **Matplotlib**: For data visualization 📊

### 🔧 Installation Steps

1. **Clone the Repository**

   Clone the project repository to your local machine:

   ```bash
   git clone https://github.com/Kshitiz726/motion_analyser.git
   ```

2. **Navigate to the Project Directory**

   Move into the project folder:

   ```bash
   cd motion_analyser
   ```

3. **Set Up a Virtual Environment (Optional but Recommended)**

   It's best to create a virtual environment to manage dependencies:

   ```bash
   conda create --name motion_analysis python=3.8
   conda activate motion_analysis
   ```

4. **Install Dependencies**

   Install all required packages using pip:

   ```bash
   pip install -r requirements.txt
   ```

5. **Prepare the Dataset**

   Download the required dataset (e.g., UCF101 or DAVIS-2017 dataset for motion detection). Store it in the `data/` directory, structured as:

   ```plaintext
   data/
       raw/
       processed/
       training/
       test/
   ```

6. **Set Up Git LFS (For Large Files)**

   If you're dealing with large files, like videos or images, consider setting up Git Large File Storage (LFS). Follow [this guide](https://git-lfs.github.com/) for more details.

## 🏃‍♂️ Running the Program

Once everything is set up, you can start the project by training the model and running the Flask app.

### 1. Train the Model

To train the motion detection model, run:

```bash
python src/train_model.py
```

This will use the dataset in the `data/training/` folder to train the model.

### 2. Start the Flask Application

After training the model, run the Flask application:

```bash
python src/app.py
```

This starts the local server at `http://127.0.0.1:5000/`, where you can access the dashboard.

## 🌐 Accessing the Dashboard

### Main Dashboard

Once the app is running, open your web browser and visit:

```
http://127.0.0.1:5000/
```

The dashboard will display:

- **Motion Detection**: Moving objects will be highlighted in green 🟩, non-moving in red 🟥.
- **Speed Calculation**: The average speed of moving objects will be displayed 🚗💨.
- **Heatmap**: Visualizes areas with motion 🔥.

### Analytics Page

To view more detailed analytics, navigate to:

```
http://127.0.0.1:5000/analytics
```

Here you’ll find:

- **Average Speed of objects** 🚀
- **Object Count** (e.g., total number of detected moving objects)
- **Zone-wise Activity graphs** 🗺️
- **Speed Distribution graphs** 📊

## 🛠️ Usage

The project is built to work in real-time. Once the model is trained and the server is running, you can upload videos to analyze their motion.

- **Motion Detection**: Moving objects are detected and displayed in real-time 🏃‍♂️
- **Speed Calculation**: Displays the average speed of moving objects over time 🏎️
- **Heatmap**: A real-time heatmap of motion regions 🔥
- **Object Classification**: Detects and classifies objects in the video 🎯

## 🤝 Contributing

We welcome contributions to improve this project! Feel free to:

1. Fork the repository 💡
2. Create an issue to report bugs 🐞
3. Submit a pull request with your changes 🔄

For any contributions, please follow the [GitHub contributing guide](CONTRIBUTING.md).

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Thanks to [UCF101](https://www.crcv.ucf.edu/data/UCF101.php) and [DAVIS-2017](https://davischallenge.org/davis2017/code.html) for providing the datasets 📂
- Special thanks to all the contributors who helped make this project a reality ✨

---

🚀 Enjoy using the Motion Analysis Project! 🎉
