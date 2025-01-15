import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for non-GUI rendering
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import os
import io
import base64

app = Flask(__name__)

# Load and preprocess the dataset
courses_path = 'coursera-courses.csv'
if not os.path.exists(courses_path):
    raise FileNotFoundError(f"The file {courses_path} does not exist. Please ensure it is correctly uploaded.")

courses_df = pd.read_csv(courses_path)
courses_df['skills'] = courses_df['skills'].apply(
    lambda x: [skill.lower() for skill in eval(x)] if isinstance(x, str) else []
)
courses_df.reset_index(inplace=True)  # Add an index column for unique course IDs
courses_df.rename(columns={'index': 'course_id'}, inplace=True)

class AntColony:
    def __init__(self, courses, num_ants, num_iterations, alpha, beta, evaporation_rate):
        self.courses = courses
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.pheromones = np.ones(len(courses))
        self.skills_array = np.array(courses['skills'])

    def heuristic(self, user_skills_set):
        course_skills = [set(skills) for skills in self.skills_array]
        matching_skills = [len(user_skills_set.intersection(skills)) for skills in course_skills]
        return np.array([match / len(skills) if len(skills) > 0 else 0 for match, skills in zip(matching_skills, course_skills)])

    def run(self, user_skills):
        user_skills_set = set(skill.lower() for skill in user_skills)
        scores = np.zeros(len(self.courses))

        for _ in range(self.num_iterations):
            heuristic_values = self.heuristic(user_skills_set)
            iteration_scores = np.zeros(len(self.courses))

            for _ in range(self.num_ants):
                probabilities = (self.pheromones ** self.alpha) * (heuristic_values ** self.beta)
                prob_sum = probabilities.sum()

                if prob_sum > 0:
                    probabilities /= prob_sum
                    selected_index = np.random.choice(len(self.courses), p=probabilities)
                    iteration_scores[selected_index] += 1

            self.pheromones = (1 - self.evaporation_rate) * self.pheromones + iteration_scores / self.num_ants
            scores += iteration_scores

        non_zero_indices = np.where(scores > 0)[0]
        filtered_courses = self.courses.iloc[non_zero_indices]
        filtered_scores = scores[non_zero_indices]
        sorted_indices = np.argsort(-filtered_scores)
        return filtered_courses.iloc[sorted_indices], filtered_scores[sorted_indices]

# Store user feedback and accuracy history
user_feedback = {}
accuracy_history = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    user_skills = [skill.strip().lower() for skill in data.get('skills', '').split(',') if skill.strip()]

    if not user_skills:
        return jsonify({"error": "No skills provided."}), 400

    ant_colony = AntColony(
        courses=courses_df,
        num_ants=500,
        num_iterations=1,
        alpha=1.0,
        beta=2.0,
        evaporation_rate=0.1
    )

    ranked_courses, scores = ant_colony.run(user_skills)
    top_courses = []
    for i in range(min(5, len(ranked_courses))):
        course = ranked_courses.iloc[i]
        course_id = int(course['course_id'])
        top_courses.append({
            "course_id": course_id,
            "course_name": course['course_name'],
            "course_provided_by": course['course_provided_by'],
            "skills": ', '.join(course['skills']),
            "course_rating": course.get('course_rating', 'N/A'),
            "course_url": course.get('course_url', 'N/A'),
            "match_score": f"{scores[i]:.2f}"
        })

        # Initialize feedback entry for the recommendation
        if course_id not in user_feedback:
            user_feedback[course_id] = {"positive": 0, "total": 0}

    return jsonify({"recommendations": top_courses})

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    data = request.json
    course_id = data.get('course_id')
    feedback = data.get('feedback')

    if course_id is None or feedback is None:
        return jsonify({"error": "Feedback data is incomplete."}), 400

    feedback_value = 1 if str(feedback).strip().lower() in ['1', 'positive', 'yes', 'true'] else 0

    if course_id in user_feedback:
        user_feedback[course_id]['positive'] += feedback_value
        user_feedback[course_id]['total'] += 1
    else:
        user_feedback[course_id] = {"positive": feedback_value, "total": 1}

    return jsonify({"message": "Feedback received!"})

@app.route('/accuracy_graph')
def accuracy_graph():
    global accuracy_history

    if not user_feedback:
        # If no feedback exists, return a default graph
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([], [], marker='o')
        ax.set_title('Accuracy of Recommendations Over Time')
        ax.set_xlabel('Time (Requests)')
        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim(0, 100)

        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
        plt.close(fig)

        return jsonify({"graph": img_base64, "accuracy": 0, "history": []})

    # Calculate current accuracy
    total_positive = sum(feedback['positive'] for feedback in user_feedback.values())
    total_feedback = sum(feedback['total'] for feedback in user_feedback.values())
    accuracy = (total_positive / total_feedback) * 100 if total_feedback > 0 else 0

    # Update accuracy history
    accuracy_history.append(accuracy)

    # Plot the accuracy over time as a line graph
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(accuracy_history, marker='o', color='blue')
    ax.set_title('Accuracy of Recommendations Over Time')
    ax.set_xlabel('Time (Requests)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 100)

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close(fig)

    return jsonify({"graph": img_base64, "accuracy": accuracy, "history": accuracy_history})

if __name__ == '__main__':
    app.run(debug=True)
